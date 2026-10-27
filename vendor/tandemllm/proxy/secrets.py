"""API keys at rest: AES-256-GCM with a master key that never sits in the config database.

A backend's key (an OpenAI or DeepSeek key, the bearer of a remote vLLM) is stored in the config
database's `backend_secrets` table as ciphertext, a 12-byte nonce and the master key's id. The
backend id is the associated data, so a ciphertext copied onto another backend's row does not
decrypt. The key is decrypted into memory for the relay's `Authorization` header and nowhere else.

THE MASTER KEY, 32 bytes, from the first of these that answers:

    env       TANDEM_SECRETS_KEY, base64 of 32 bytes (a docker deploy, a CI run)
    keychain  the OS keychain, read only: `security find-generic-password -s tandemllm
              -a secrets-key -w` on macOS, `secret-tool lookup service tandemllm key secrets-key`
              on Linux; each is optional, and a missing tool or entry is a quiet miss
    file      `secrets.key` beside the config database (`~/.tandemllm/secrets.key` by default),
              written once with 32 random bytes at mode 0600

The Backends screen names the source. Encryption at rest protects a copied or backed-up database;
it does not protect against a process running as the same user, which can read the key file or
the environment as easily as the proxy can. A backup of the whole directory carries the key file
too: keep the master key in the environment or the keychain when backups leave the machine.

FAIL CLOSED. A ciphertext made under another master key (the key id differs) or one that fails
its tag is never sent as a key: `decrypt` raises `SecretError` and the backend reports "key
unreadable". A lost master key means the keys are typed in again, nothing else.

WHAT LEAVES THIS MODULE. `hint()` gives the API and the GUI `key_set`, the last 4 characters and
the first 6 hex digits of the key's sha256. Never the key: no log line, journal row, ledger row or
API body holds one (tests/test_proxy_secrets.py greps every output for every fixture key).
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets as _rand
import subprocess
import sys

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:                                                # pragma: no cover
    AESGCM = None

ENV = "TANDEM_SECRETS_KEY"
KEYCHAIN_SERVICE = "tandemllm"
KEYCHAIN_ACCOUNT = "secrets-key"
KEY_FILE = "secrets.key"
NONCE_BYTES = 12


class SecretError(Exception):
    """A key that cannot be read: wrong master key, a broken ciphertext, no cryptography."""


def key_id(master: bytes) -> str:
    """The master key's id stored beside each ciphertext: 8 hex digits of its sha256."""
    return hashlib.sha256(b"tandemllm-master-v1" + master).hexdigest()[:8]


def hint(key: str | None) -> dict:
    """What the API and the GUI may show of a key."""
    if not key:
        return {"key_set": False, "last4": None, "sha256_6": None}
    return {"key_set": True, "last4": key[-4:] if len(key) >= 8 else None,
            "sha256_6": hashlib.sha256(key.encode()).hexdigest()[:6]}


def _decode_master(text: str, where: str) -> bytes:
    try:
        raw = base64.b64decode(text.strip(), validate=True)
    except (ValueError, TypeError):
        raise SecretError(f"{where} is not base64")
    if len(raw) != 32:
        raise SecretError(f"{where} must hold 32 bytes, it holds {len(raw)}")
    return raw


def _keychain(cmd: list[str] | None = None, timeout: float = 3.0) -> str | None:
    """The master key from the OS keychain, or None. Read only; a missing tool is a miss."""
    if cmd is None:
        if sys.platform == "darwin":
            cmd = ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE,
                   "-a", KEYCHAIN_ACCOUNT, "-w"]
        elif sys.platform.startswith("linux"):
            cmd = ["secret-tool", "lookup", "service", KEYCHAIN_SERVICE, "key", KEYCHAIN_ACCOUNT]
        else:
            return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    out = r.stdout.strip()
    return out if r.returncode == 0 and out else None


class Vault:
    """The master key and the two operations on it. One per process."""

    def __init__(self, master: bytes, source: str):
        if AESGCM is None:
            raise SecretError("the `cryptography` package is missing; install it to store keys")
        self._aead = AESGCM(master)
        self.key_id = key_id(master)
        self.source = source                     # env | keychain | file | test

    @classmethod
    def load(cls, key_dir: str, env=None, keychain_cmd: list[str] | None = None,
             use_keychain: bool = True) -> "Vault":
        """The master key by the order above. The key file is created when nothing else answers."""
        env = os.environ if env is None else env
        text = env.get(ENV)
        if text:
            return cls(_decode_master(text, ENV), "env")
        if use_keychain:
            got = _keychain(keychain_cmd)
            if got:
                return cls(_decode_master(got, "the keychain entry"), "keychain")
        path = os.path.join(os.path.expanduser(key_dir), KEY_FILE)
        if os.path.exists(path):
            with open(path) as f:
                return cls(_decode_master(f.read(), path), "file")
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        master = _rand.token_bytes(32)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(base64.b64encode(master).decode() + "\n")
        return cls(master, "file")

    def encrypt(self, backend_id: str, key: str) -> tuple[bytes, bytes, str]:
        """(ciphertext, nonce, key id) of `key` for the row of `backend_id`."""
        nonce = _rand.token_bytes(NONCE_BYTES)
        return self._aead.encrypt(nonce, key.encode(), backend_id.encode()), nonce, self.key_id

    def decrypt(self, backend_id: str, ciphertext: bytes, nonce: bytes, kid: str) -> str:
        if kid != self.key_id:
            raise SecretError("the key was stored under another master key; enter it again")
        try:
            return self._aead.decrypt(nonce, ciphertext, backend_id.encode()).decode()
        except Exception:                                          # noqa: BLE001
            raise SecretError("the stored key does not decrypt (tag mismatch); enter it again")
