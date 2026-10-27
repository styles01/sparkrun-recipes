"""A chat's files: `<data>/workspaces/<conversation id>/`, mounted as `/workspace` in its sandbox.

chatd reads and writes them on the host, so a read, a write, a search or the files pane needs no
running container. The tree is the agent's, and the agent changes it while chatd works in it, so
nothing here trusts it (SEC-1):

- every path is resolved under the workspace root; `..` and an absolute path outside `/workspace`
  are refused (`PathError`, a 400);
- every access walks from the root one directory at a time, each opened with O_NOFOLLOW relative
  to the one before (`SandboxDir`). A symlink anywhere on the way is refused, and a directory the
  sandbox swaps for a symlink after a check cannot redirect the access: there is no check apart
  from the open itself. A file is opened with O_NOFOLLOW and O_NONBLOCK and read only when its
  fstat says it is a regular file (a FIFO would block chatd);
- a write goes to a new temporary file (O_EXCL) in the verified directory and is renamed over the
  name, so it never writes into a file that is already there;
- nothing here executes anything from the tree. git runs only inside the container: a `.git/config`
  the agent wrote (core.fsmonitor, hooks) would otherwise run on the host as chatd's user.

The sandbox's home (`<data>/homes/<id>`, /home/agent) is a `SandboxDir` too (chatd/owui.py).

The `etag` of a file is its git blob hash (sha1 of `blob <size>\\0<bytes>`), so the editor's
If-Match and git agree on what "this version" is.
"""

from __future__ import annotations

import errno
import fnmatch
import hashlib
import os
import re
import secrets
import shutil
import stat

MOUNT = "/workspace"
MAX_READ = 2 * 1024 * 1024          # a read the tools hand the model at once (the pane streams)
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", ".tandem"}

_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_NEW = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


class PathError(ValueError):
    pass


def etag(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class SandboxDir:
    """A folder the sandbox writes into, seen from the host: `rel` paths under `root` ("a/b.txt"),
    every one reached through directory fds opened with O_NOFOLLOW. `mount` is where the sandbox
    sees the root; messages name paths that way."""

    def __init__(self, root: str, mount: str):
        self.root = os.path.realpath(root)
        self.mount = mount

    def ensure(self) -> None:
        os.makedirs(self.root, mode=0o755, exist_ok=True)

    def show(self, rel: str) -> str:
        return f"{self.mount}/{rel}" if rel else self.mount

    def _parts(self, rel: str) -> list[str]:
        parts = [s for s in rel.split("/") if s not in ("", ".")]
        if any(s == ".." or "\0" in s for s in parts):
            raise PathError(f"{self.show(rel)}: `..` is not allowed")
        return parts

    def _refused(self, dfd: int, seg: str, label: str) -> PathError:
        try:
            link = stat.S_ISLNK(os.stat(seg, dir_fd=dfd, follow_symlinks=False).st_mode)
        except OSError:
            link = False
        return PathError(f"{label}: a symlink on the way is not followed" if link else
                         f"{label}: {seg} is not a directory")

    def _dir(self, parts: list[str], label: str, make: bool = False) -> int:
        """An fd of the folder `parts`, each step opened from the one before with O_NOFOLLOW (and
        made when `make`); the caller closes it."""
        fd = os.open(self.root, _DIR)
        try:
            for seg in parts:
                try:
                    try:
                        nxt = os.open(seg, _DIR, dir_fd=fd)
                    except FileNotFoundError:
                        if not make:
                            raise PathError(f"{label}: no such file") from None
                        try:
                            os.mkdir(seg, 0o755, dir_fd=fd)
                        except FileExistsError:
                            pass
                        nxt = os.open(seg, _DIR, dir_fd=fd)
                except OSError as exc:
                    if exc.errno in (errno.ENOTDIR, errno.ELOOP, errno.EMLINK):
                        raise self._refused(fd, seg, label) from None
                    raise
                os.close(fd)
                fd = nxt
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _open_in(self, dfd: int, name: str, label: str):
        """(fd, fstat) of the regular file `name` in the folder `dfd`."""
        try:
            fd = os.open(name, _FILE, dir_fd=dfd)
        except FileNotFoundError:
            raise PathError(f"{label}: no such file") from None
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.EMLINK):
                raise PathError(f"{label}: a symlink on the way is not followed") from None
            raise
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            os.close(fd)
            raise PathError(f"{label}: is a directory" if stat.S_ISDIR(st.st_mode) else
                            f"{label}: not a file")
        return fd, st

    def open(self, rel: str, label: str | None = None):
        """(binary file object, its fstat) of a regular file; the caller closes it. The stat is
        the open file's, so a file replaced after the open is read as it was."""
        label = label or self.show(rel)
        parts = self._parts(rel)
        if not parts:
            raise PathError(f"{label}: is a directory")
        dfd = self._dir(parts[:-1], label)
        try:
            fd, st = self._open_in(dfd, parts[-1], label)
        finally:
            os.close(dfd)
        return os.fdopen(fd, "rb"), st

    def read(self, rel: str, limit: int | None = None, label: str | None = None) -> bytes:
        fh, _ = self.open(rel, label)
        with fh:
            return fh.read(limit) if limit else fh.read()

    def stat(self, rel: str, label: str | None = None) -> os.stat_result:
        """The lstat of `rel` (a symlink at the end is itself)."""
        label = label or self.show(rel)
        parts = self._parts(rel)
        if not parts:
            return os.stat(self.root)
        dfd = self._dir(parts[:-1], label)
        try:
            return os.stat(parts[-1], dir_fd=dfd, follow_symlinks=False)
        except FileNotFoundError:
            raise PathError(f"{label}: no such file") from None
        finally:
            os.close(dfd)

    def write(self, rel: str, data: bytes, mode: int = 0o644, label: str | None = None,
              over_link: bool = False, if_etag: str | None = None) -> None:
        """Create or replace `rel` atomically (its folders made as needed): the bytes go to a new
        temporary file in the verified folder, renamed over the name. A symlink at the name is
        refused, or replaced itself with `over_link`; `if_etag`: the etag the file must still have
        (ChangedError)."""
        label = label or self.show(rel)
        parts = self._parts(rel)
        if not parts:
            raise PathError(f"{label}: is a directory")
        name = parts[-1]
        dfd = self._dir(parts[:-1], label, make=True)
        try:
            try:
                st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
            except FileNotFoundError:
                st = None
            if st is not None and stat.S_ISDIR(st.st_mode):
                raise PathError(f"{label}: is a directory")
            if st is not None and stat.S_ISLNK(st.st_mode) and not over_link:
                raise PathError(f"{label}: a symlink on the way is not followed")
            if if_etag is not None and st is not None:
                fd, _ = self._open_in(dfd, name, label)
                with os.fdopen(fd, "rb") as fh:
                    if etag(fh.read()) != if_etag:
                        raise ChangedError(label)
            tmp = f".tandem-{secrets.token_hex(6)}.tmp"
            fd = os.open(tmp, _NEW, mode, dir_fd=dfd)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, name, src_dir_fd=dfd, dst_dir_fd=dfd)
            except BaseException:
                try:
                    os.unlink(tmp, dir_fd=dfd)
                except OSError:
                    pass
                raise
        finally:
            os.close(dfd)

    def create(self, rel: str, data: bytes, mode: int = 0o644, label: str | None = None) -> None:
        """A new file at `rel` (its folders made as needed); FileExistsError when the name is
        taken by anything, a dangling symlink included."""
        label = label or self.show(rel)
        parts = self._parts(rel)
        if not parts:
            raise PathError(f"{label}: is a directory")
        dfd = self._dir(parts[:-1], label, make=True)
        try:
            fd = os.open(parts[-1], _NEW, mode, dir_fd=dfd)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
            except BaseException:
                try:
                    os.unlink(parts[-1], dir_fd=dfd)
                except OSError:
                    pass
                raise
        finally:
            os.close(dfd)

    def delete(self, rel: str, label: str | None = None) -> None:
        """A file, a symlink (itself) or a folder with everything in it."""
        label = label or self.show(rel)
        parts = self._parts(rel)
        if not parts:
            raise PathError(f"{label}: the workspace itself is not deleted")
        dfd = self._dir(parts[:-1], label)
        try:
            try:
                st = os.stat(parts[-1], dir_fd=dfd, follow_symlinks=False)
            except FileNotFoundError:
                raise PathError(f"{label}: no such file") from None
            if stat.S_ISDIR(st.st_mode):
                shutil.rmtree(parts[-1], dir_fd=dfd)
            else:
                os.unlink(parts[-1], dir_fd=dfd)
        finally:
            os.close(dfd)

    def scan(self, rel: str, label: str | None = None) -> list[tuple[str, os.stat_result]]:
        """(name, lstat) of each entry of the folder `rel`."""
        dfd = self._dir(self._parts(rel), label or self.show(rel))
        try:
            with os.scandir(dfd) as it:
                return [(e.name, e.stat(follow_symlinks=False)) for e in it]
        finally:
            os.close(dfd)

    def files(self, rel: str = "", skip=()):
        """The rel path of every regular file under `rel` (sorted, folders in `skip` left out), by
        os.fwalk from a verified fd: a symlinked folder is not entered, a symlink not listed."""
        parts = self._parts(rel)
        dfd = self._dir(parts, self.show(rel))
        try:
            for dirpath, dirnames, filenames, dirfd in os.fwalk(".", dir_fd=dfd,
                                                                 follow_symlinks=False):
                dirnames[:] = sorted(d for d in dirnames if d not in skip)
                base = parts + [s for s in dirpath.split("/") if s not in ("", ".")]
                for f in sorted(filenames):
                    try:
                        st = os.stat(f, dir_fd=dirfd, follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.S_ISREG(st.st_mode):
                        yield "/".join(base + [f])
        finally:
            os.close(dfd)


class Workspace:
    def __init__(self, root: str):
        self.dir = SandboxDir(root, MOUNT)
        self.root = self.dir.root

    def ensure(self) -> None:
        self.dir.ensure()

    # ------------------------------------------------------------------ paths
    def rel(self, path: str) -> str:
        """`path` (relative, or absolute under /workspace) as a clean relative path; PathError
        for anything that leaves the root."""
        p = (path or "").strip()
        if p.startswith(MOUNT):
            p = p[len(MOUNT):]
        elif p.startswith("/"):
            raise PathError(f"{path}: outside {MOUNT}; the file tools read and write {MOUNT} only "
                            f"(copy the file there with bash first)")
        parts = []
        for seg in p.replace("\\", "/").split("/"):
            if seg in ("", "."):
                continue
            if seg == ".." or "\0" in seg:
                raise PathError(f"{path}: `..` is not allowed")
            parts.append(seg)
        return "/".join(parts)

    def resolve(self, path: str, must_exist: bool = False) -> str:
        """The host path, for show() and checks only: every existing component on the way must be
        a real directory or file, not a symlink. Nothing opens what it returns; the reads and
        writes below go through SandboxDir."""
        rel = self.rel(path)
        cur = self.root
        for seg in rel.split("/") if rel else []:
            cur = os.path.join(cur, seg)
            try:
                st = os.lstat(cur)
            except FileNotFoundError:
                if must_exist:
                    raise PathError(f"{path}: no such file") from None
                break
            if stat.S_ISLNK(st.st_mode):
                raise PathError(f"{path}: a symlink on the way is not followed")
        full = os.path.join(self.root, rel) if rel else self.root
        real = os.path.realpath(full)
        if real != self.root and not real.startswith(self.root + os.sep):
            raise PathError(f"{path}: outside {MOUNT}; the file tools read and write {MOUNT} only "
                            f"(copy the file there with bash first)")
        return full

    def show(self, full: str) -> str:
        """A host path as the agent sees it."""
        rel = os.path.relpath(full, self.root)
        return MOUNT if rel == "." else f"{MOUNT}/{rel}"

    # ------------------------------------------------------------------ files
    def read(self, path: str, limit: int | None = None) -> bytes:
        return self.dir.read(self.rel(path), limit, label=path)

    def open(self, path: str):
        """(binary file object, its fstat) for a streamed read; the caller closes it. The stat is
        the open file's, so a file replaced after the open is read as it was."""
        return self.dir.open(self.rel(path), label=path)

    def stat(self, path: str) -> os.stat_result:
        return self.dir.stat(self.rel(path), label=path)

    def write(self, path: str, data: bytes, if_match: str | None = None) -> str:
        """Create or replace; `if_match`: the etag the caller read (a 412 when it moved)."""
        self.dir.write(self.rel(path), data, label=path, if_etag=if_match)
        return etag(data)

    def delete(self, path: str) -> None:
        self.dir.delete(self.rel(path), label=path)

    def list(self, path: str = MOUNT) -> list[dict]:
        rel = self.rel(path)
        out = []
        for name, st in self.dir.scan(rel, label=path):
            kind = "dir" if stat.S_ISDIR(st.st_mode) else "file"
            out.append({"path": self.dir.show(f"{rel}/{name}" if rel else name), "kind": kind,
                        "bytes": 0 if kind == "dir" else int(st.st_size),
                        "modified_at": _iso(st.st_mtime)})
        return sorted(out, key=lambda f: (f["kind"] != "dir", f["path"]))

    def size(self) -> int:
        total = 0
        for dirpath, dirnames, filenames, dirfd in os.fwalk(self.root, follow_symlinks=False):
            for f in filenames:
                try:
                    total += os.stat(f, dir_fd=dirfd, follow_symlinks=False).st_size
                except OSError:
                    pass
        return total

    # ------------------------------------------------------------------ search
    def walk(self, start: str = MOUNT):
        """The host path of every regular file under `start` (SandboxDir.files: no symlink, no
        SKIP_DIRS); open them through read() or open(), never by the path."""
        for rel in self.dir.files(self.rel(start), SKIP_DIRS):
            yield os.path.join(self.root, rel)

    def glob(self, pattern: str, start: str = MOUNT, limit: int = 500) -> list[str]:
        pat = pattern.lstrip("/")
        if pat.startswith("workspace/"):
            pat = pat[len("workspace/"):]
        rx = _glob_rx(pat)
        out = []
        for p in self.walk(start):
            rel = os.path.relpath(p, self.root)
            if rx.fullmatch(rel) or ("/" not in pat and fnmatch.fnmatch(os.path.basename(p), pat)):
                out.append(self.show(p))
                if len(out) >= limit:
                    break
        return out

    def grep(self, pattern: str, start: str = MOUNT, include: str | None = None,
             ignore_case: bool = False, limit: int = 200,
             skipped: list | None = None) -> list[str]:
        """`file:line: text` of the lines that match, at most `limit`. `start` is a folder (every
        file under it, a file over MAX_READ left out and named in `skipped` when given) or one
        file, searched whole, line by line, whatever its size."""
        try:
            rx = re.compile(pattern, re.I if ignore_case else 0)
        except re.error as exc:
            raise PathError(f"bad pattern: {exc}") from None
        if stat.S_ISREG(self.stat(start).st_mode):
            # one file: the path the caller named, not a folder to walk (it said "is not a
            # directory" before)
            fh, _ = self.open(start)
            out: list[str] = []
            with fh:
                if b"\0" in fh.read(4096):
                    return out
                fh.seek(0)
                for n, raw in enumerate(fh, 1):
                    line = raw.decode("utf-8", "replace").rstrip("\r\n")
                    if rx.search(line):
                        out.append(f"{self.show(self.resolve(start))}:{n}: {line[:300]}")
                        if len(out) >= limit:
                            break
            return out
        out = []
        for p in self.walk(start):
            if include and not fnmatch.fnmatch(os.path.basename(p), include):
                continue
            try:
                fh, st = self.dir.open(os.path.relpath(p, self.root))
                with fh:
                    if st.st_size > MAX_READ:
                        if skipped is not None:
                            skipped.append((self.show(p), st.st_size))
                        continue
                    data = fh.read(MAX_READ + 1)
            except (PathError, OSError):
                continue
            if b"\0" in data[:4096]:
                continue
            for n, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                if rx.search(line):
                    out.append(f"{self.show(p)}:{n}: {line[:300]}")
                    if len(out) >= limit:
                        return out
        return out


class ChangedError(Exception):
    """The file moved on since the caller read it (the editor's 412)."""


def _glob_rx(pat: str) -> re.Pattern:
    """`**` crosses directories, `*` and `?` do not."""
    out, i = "", 0
    while i < len(pat):
        if pat.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pat.startswith("**", i):
            out += ".*"
            i += 2
        elif pat[i] == "*":
            out += "[^/]*"
            i += 1
        elif pat[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pat[i])
            i += 1
    return re.compile(out)


def _iso(t: float) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
