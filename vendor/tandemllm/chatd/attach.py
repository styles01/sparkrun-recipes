"""A safety net for attachments: none is ever inlined past the window (SRV-487).

Live (2026-10-15, chat 01M4ZSH3DEEX2G3X11YK1RND30): a 9.5 MB Medium article saved as HTML, its
figures embedded as base64 `data:` URIs, went into the prompt whole (2,363,997 tokens), and the run
stopped with "This conversation is about 2,875,364 tokens ... Start a new chat", which blamed the
conversation for one file.

    save_upload(ws, name, data)  every attachment of a message goes to the chat's workspace under
                                 uploads/ when the message is sent (a free name, at most 255 bytes
                                 of UTF-8); the sandbox's tools work on it from there. The sandbox
                                 owns that folder, so the file is created through SandboxDir: no
                                 symlink on the way, a new file only (O_EXCL). When it cannot be
                                 placed (uploads/ a symlink, the disk full) the message is still
                                 sent: the attachment keeps its blob, `unplaced` says why, and the
                                 prompt says so in one line (unplaced_line)
    budget(window, used)         the most tokens one message's text attachments may put in the
                                 prompt: a quarter of the window's free room, at most
                                 INLINE_MAX_TOKENS
    line(a)                      what the prompt gets instead of a text file over the budget: one
                                 line with its name, size, tokens and path

The placement is decided once, at the send (`inline`, `path` on the attachment), so every later turn
renders the same bytes (chatd/render.py).
"""

from __future__ import annotations

import errno
import os

from chatd.workspace import MOUNT, PathError, SandboxDir

INLINE_MAX_TOKENS = 32_000
INLINE_SHARE = 0.25
LINE_TOKENS = 64                    # what the line costs in the prompt, about
UPLOADS = "uploads"
NAME_MAX = 255                      # bytes of one file name (Linux, APFS)
EXT_MAX = 32                        # a longer "extension" is part of the name
FREE_TRIES = 1000


def budget(window: int, used: int) -> int:
    """The tokens one message's text attachments may take: a quarter of the free room, at most
    INLINE_MAX_TOKENS; INLINE_MAX_TOKENS when the window is unknown."""
    if not window:
        return INLINE_MAX_TOKENS
    return max(0, min(INLINE_MAX_TOKENS, int((window - used) * INLINE_SHARE)))


def size_words(n: int) -> str:
    return (f"{n / 1024 / 1024:.1f} MB" if n >= 1024 * 1024 else f"{n / 1024:.0f} KB"
            if n >= 1024 else f"{n} bytes")


def line(a: dict) -> str:
    """The prompt's line for a text attachment over the budget."""
    where = (f"It is saved at {a['path']}." if a.get("path") else
             f"It could not be saved to {MOUNT}/{UPLOADS}/ ({a['unplaced']})." if
             a.get("unplaced") else
             "It was attached before attachments were saved to the workspace.")
    return (f"\n\n[Attached file {a.get('name') or 'file'}: "
            f"{size_words(int(a.get('bytes') or 0))}, about {int(a.get('tokens') or 0):,} tokens: "
            f"too long to include. {where}]")


def unplaced_line(a: dict) -> str:
    """The prompt's line for an attachment that is not in uploads/ (its text or image, if any,
    still goes in the prompt)."""
    return (f"\n\n[Attached file {a.get('name') or 'file'}: it could not be saved to "
            f"{MOUNT}/{UPLOADS}/ ({a['unplaced']}), so the sandbox's tools do not have it.]")


def unplaced_words(exc: Exception) -> str:
    """Why save_upload failed, for `unplaced`."""
    if isinstance(exc, PathError):
        return str(exc)
    return getattr(exc, "strerror", None) or type(exc).__name__


def fit_name(stem: str, tail: str) -> str:
    """`stem` cut on a character boundary so that stem + tail is at most NAME_MAX bytes."""
    room = NAME_MAX - len(tail.encode())
    return stem.encode()[:room].decode("utf-8", "ignore") + tail


def save_upload(ws_dir: str, name: str, data: bytes) -> str:
    """The file under the workspace's uploads/ with a free name; its path as the sandbox sees it.
    PathError (a symlink on the way) or OSError when it cannot be placed."""
    d = SandboxDir(ws_dir, MOUNT)
    base = os.path.basename(name).replace("\\", "_").replace("\0", "")
    base = base.encode("utf-8", "replace").decode()           # no lone surrogate
    if base in ("", ".", ".."):
        base = "file"
    stem, ext = os.path.splitext(base)
    if len(ext.encode()) > EXT_MAX:
        stem, ext = base, ""
    for i in range(1, FREE_TRIES + 1):
        pick = fit_name(stem, (f"-{i}" if i > 1 else "") + ext)
        try:
            d.create(f"{UPLOADS}/{pick}", data)
        except FileExistsError:
            continue
        return f"{MOUNT}/{UPLOADS}/{pick}"
    raise FileExistsError(errno.EEXIST, f"no free name for {base}")
