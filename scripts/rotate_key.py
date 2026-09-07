#!/usr/bin/env python3
"""Re-encrypt the whole store under a new APP_SECRET.

A secret you cannot rotate is a secret you will not rotate — including after you
suspect it has leaked. This makes rotation a five-minute operation.

    # See what would change, without touching anything:
    python3 scripts/rotate_key.py --dry-run

    # Do it:
    python3 scripts/rotate_key.py --new-secret "$(python3 -c 'import secrets;print(secrets.token_urlsafe(48))')"

How it works: a new primary key is derived from the new secret, while every
previous key is retained as a read-only fallback. Each document is decrypted with
whichever key wrote it and rewritten under the new one. Because old keys stay
readable throughout, an interrupted run leaves a store that still opens — restart
it and the remaining documents move.

Afterwards, put the new secret in ``.streamlit/secrets.toml`` (or your
environment) **before** starting the app again. The old secret can then be
destroyed.

One thing this does NOT re-encrypt: each user's personal API keys. Those are
sealed under a key derived from that user's password, which is the whole point —
this script has no way to read them, and it does not need to. They keep working
untouched.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.keymgmt import (  # noqa: E402
    KEYRING_PATH,
    Keyring,
    KeyError_,
    load_or_create_keyring,
)
from src.storage import Cipher, StorageError, _make_backend  # noqa: E402


def load_secrets() -> dict[str, str]:
    """Read configuration the same way the app does."""
    values: dict[str, str] = {}
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    for name in (
        "APP_SECRET", "DATA_DIR", "DROPBOX_APP_KEY", "DROPBOX_APP_SECRET",
        "DROPBOX_REFRESH_TOKEN", "DROPBOX_FOLDER",
    ):
        values[name] = os.environ.get(name, "")

    try:  # Streamlit secrets, when present, win — that is what the app uses.
        import streamlit as st

        for name in list(values):
            if name in st.secrets:
                values[name] = str(st.secrets[name])
    except Exception:
        pass
    return values


def reencrypt_inner(
    path: str, document: dict, old: Cipher, new: Cipher
) -> tuple[dict, int]:
    """Re-wrap credentials stored as tokens *inside* a document.

    Rewriting the outer envelope is not enough. Issued API keys and the
    management key are encrypted a second time, under their own subkeys, so they
    have to be decrypted with the old key and re-encrypted with the new one — or
    they would still need the old secret after rotation, and the rotation would
    be a lie.

    Users' personal API keys are deliberately absent from this list: they are
    sealed under each user's password, this script cannot read them, and they
    need no rotation.
    """
    count = 0

    if path.startswith("users/u/"):
        user = document.get("user")
        if isinstance(user, dict):
            issued = user.get("issued_keys")
            if isinstance(issued, dict):
                for provider, token in list(issued.items()):
                    plaintext = old.decrypt_text(token)
                    if plaintext:
                        issued[provider] = new.encrypt_text(plaintext)
                        count += 1

    elif path == "config":
        values = document.get("values")
        if isinstance(values, dict):
            for name, token in list(values.items()):
                if not name.startswith("secret:"):
                    continue
                plaintext = old.decrypt_config(token)
                if plaintext:
                    values[name] = new.encrypt_config(plaintext)
                    count += 1

    return document, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--new-secret", help="the replacement APP_SECRET")
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be rewritten"
    )
    args = parser.parse_args()

    secrets = load_secrets()
    old_secret = secrets.get("APP_SECRET", "")
    if not old_secret:
        print("APP_SECRET is not set — nothing to rotate.", file=sys.stderr)
        return 2

    if not args.dry_run and not args.new_secret:
        print("Pass --new-secret, or --dry-run to preview.", file=sys.stderr)
        return 2
    if args.new_secret and args.new_secret == old_secret:
        print("The new secret is the same as the old one.", file=sys.stderr)
        return 2
    if args.new_secret and len(args.new_secret) < 32:
        print(
            "That secret is short. Generate one:\n"
            "  python3 -c \"import secrets; print(secrets.token_urlsafe(48))\"",
            file=sys.stderr,
        )
        return 2

    # Open the store with the current key.
    reader_backend, _ = _make_backend(secrets, Cipher.__new__(Cipher))
    header = reader_backend.read_plain(KEYRING_PATH)
    try:
        old_ring, created = load_or_create_keyring(old_secret, header)
    except KeyError_ as exc:
        print(f"Cannot open the store: {exc}", file=sys.stderr)
        return 1

    source, _ = _make_backend(secrets, Cipher(old_secret, old_ring))
    # Everything, including the rollback watermarks — that document is encrypted
    # like any other, and leaving it behind would make the store unreadable under
    # the new key.
    paths = source.list_paths()

    if not paths:
        print("No documents found. Is DATA_DIR / Dropbox configured correctly?")
        return 1

    print(f"Store: {source.describe()}")
    print(f"Documents: {len(paths)}")
    for path in paths:
        print(f"  - {path}")

    if args.dry_run:
        print("\nDry run — nothing was changed.")
        if created:
            print("Note: no keyring header exists yet; one will be written on rotation.")
        return 0

    new_ring: Keyring = old_ring.rotated(args.new_secret)
    destination, _ = _make_backend(secrets, Cipher(args.new_secret, new_ring))

    print(
        f"\nRotating to key version {new_ring.primary.key_version} "
        f"({new_ring.primary.kdf})…"
    )
    old_cipher = Cipher(old_secret, old_ring)
    new_cipher = Cipher(args.new_secret, new_ring)

    moved, failed, inner = 0, 0, 0
    for path in paths:
        try:
            document = source.read(path)
            if document is None:
                continue
            document, rewrapped = reencrypt_inner(path, document, old_cipher, new_cipher)
            inner += rewrapped
            # Preserve the rollback version — rewriting must not look like a
            # brand-new document to the guard.
            destination.write(path, document)
            moved += 1
            print(f"  ✓ {path}" + (f"  ({rewrapped} credential(s) re-wrapped)" if rewrapped else ""))
        except StorageError as exc:
            failed += 1
            print(f"  ✗ {path}: {exc}", file=sys.stderr)

    if failed:
        print(
            f"\n{failed} document(s) failed. The old secret still opens them — "
            "fix the cause and run again.",
            file=sys.stderr,
        )
        return 1

    # Only once every document has moved is it safe to drop the old readers.
    final = Keyring(secret=args.new_secret, primary=new_ring.primary, legacy=[])
    destination.write_plain(KEYRING_PATH, final.to_dict())

    print(f"\nRotated {moved} document(s); re-wrapped {inner} stored credential(s).")
    print("Now set APP_SECRET to the new value before restarting the app.")
    print("Once it starts cleanly, destroy the old secret.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
