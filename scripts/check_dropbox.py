#!/usr/bin/env python3
"""Verify Dropbox storage before you trust it with a semester of work.

The failure this exists to prevent: Dropbox credentials that look right, an app
that starts without complaint, and a discovery three weeks later that every
account and transcript has been living on a container disk that gets wiped on
restart. The app falls back to local files when Dropbox does not work, which is
the right behaviour at runtime and a terrible way to find out.

So this script does the full round trip — connect, write a real encrypted
document, read it back, confirm the bytes on Dropbox are not readable, clean up —
and tells you exactly which step failed and what to change.

    python3 scripts/check_dropbox.py

It reads configuration the same way the app does (``.env``, environment,
Streamlit secrets), and it writes only to a scratch path it removes afterwards.
Nothing it does touches your accounts, settings or saved lectures.
"""

from __future__ import annotations

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.keymgmt import KEYRING_PATH, KeyError_, load_or_create_keyring  # noqa: E402
from src.storage import Cipher, DropboxStore, StorageError  # noqa: E402

REQUIRED_SCOPES = (
    "files.metadata.read",
    "files.metadata.write",
    "files.content.read",
    "files.content.write",
)

OK = "  ✓ "
BAD = "  ✗ "


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

    try:  # Streamlit secrets win, because that is what the app reads.
        import streamlit as st

        for name in list(values):
            if name in st.secrets:
                values[name] = str(st.secrets[name])
    except Exception:
        pass
    return values


def fail(message: str, *hints: str) -> int:
    print(BAD + message)
    for hint in hints:
        print("      " + hint)
    print("\nDropbox is NOT ready. See docs/DROPBOX.md.")
    return 1


def main() -> int:
    secrets = load_secrets()
    print("Checking Dropbox storage\n")

    # 1. Configuration present -------------------------------------------- #
    missing = [
        name
        for name in ("DROPBOX_APP_KEY", "DROPBOX_APP_SECRET", "DROPBOX_REFRESH_TOKEN")
        if not secrets.get(name)
    ]
    if missing:
        return fail(
            "Missing: " + ", ".join(missing),
            "Add them to .streamlit/secrets.toml (or .env for local runs).",
            "Without all three the app silently uses local files instead.",
        )
    print(OK + "All three Dropbox values are set")

    for name in ("DROPBOX_APP_KEY", "DROPBOX_APP_SECRET", "DROPBOX_REFRESH_TOKEN"):
        value = secrets[name]
        if value != value.strip():
            return fail(
                f"{name} has whitespace around it",
                "A copied token often picks up a trailing space or newline.",
            )
    print(OK + "No stray whitespace in the credentials")

    if not secrets.get("APP_SECRET"):
        return fail(
            "APP_SECRET is not set",
            "Dropbox stores encrypted blobs; without APP_SECRET there is no key.",
            'Generate one: python3 -c "import secrets; print(secrets.token_urlsafe(48))"',
        )
    print(OK + "APP_SECRET is set")

    # 2. Connect ---------------------------------------------------------- #
    # Connect before deriving keys, because the keyring header lives on Dropbox
    # and a check that invented its own salt would test a key path the app never
    # takes. A placeholder cipher is enough to read a plaintext header.
    folder = secrets.get("DROPBOX_FOLDER") or "/lecture-quiz-builder"
    placeholder = Cipher.__new__(Cipher)  # never used for crypto
    try:
        store = DropboxStore(
            placeholder,  # type: ignore[arg-type]
            secrets["DROPBOX_APP_KEY"],
            secrets["DROPBOX_APP_SECRET"],
            secrets["DROPBOX_REFRESH_TOKEN"],
            folder,
        )
    except StorageError as exc:
        return fail(str(exc), "pip install -r requirements.txt")

    if not store.available():
        return fail(
            f"Could not list the app folder — {store.last_error or 'no detail'}",
            "Required scopes: " + ", ".join(REQUIRED_SCOPES),
            "After ticking scopes you must generate a NEW refresh token;",
            "an existing token keeps whatever scopes it was issued with.",
        )
    print(OK + f"Connected, app folder {folder} is readable")

    # 3. Encryption, using the store's own keyring where one exists -------- #
    try:
        header = store.read_plain(KEYRING_PATH)
    except StorageError as exc:
        return fail(f"Could not read the keyring header: {exc}")

    try:
        keyring, created = load_or_create_keyring(secrets["APP_SECRET"], header)
        cipher = Cipher(secrets["APP_SECRET"], keyring)
    except KeyError_ as exc:
        return fail(
            f"Encryption could not start: {exc}",
            "If this deployment has data already, APP_SECRET must be the same",
            "value it was written with. Restore it from your password manager.",
        )
    store.cipher = cipher
    print(
        OK
        + (
            "Encryption initialised (no existing keyring — this is a fresh store)"
            if created
            else "Encryption initialised from the existing keyring on Dropbox"
        )
    )

    # 4. Round trip ------------------------------------------------------- #
    probe_path = f"_healthcheck/{uuid.uuid4().hex[:12]}"
    marker = uuid.uuid4().hex
    payload = {"probe": marker, "note": "written by scripts/check_dropbox.py"}

    try:
        store.write(probe_path, payload)
    except StorageError as exc:
        return fail(
            f"Write failed: {exc}",
            "Reading works but writing does not — that is usually a missing",
            "files.content.write or files.metadata.write scope.",
        )
    print(OK + "Wrote a test document")

    try:
        readback = store.read(probe_path)
    except StorageError as exc:
        return fail(f"Read-back failed: {exc}")

    if not readback or readback.get("probe") != marker:
        return fail("The document read back did not match what was written")
    print(OK + "Read it back intact")

    # 5. Confirm it is actually encrypted up there ------------------------ #
    try:
        raw = store._download(store._path(probe_path))  # noqa: SLF001
    except StorageError as exc:
        raw = None
        print(f"  ~ Could not re-download the raw blob to inspect it ({exc})")
    if raw is not None:
        if marker.encode() in raw:
            return fail(
                "The stored file contains plaintext",
                "This should be impossible; do not use this deployment.",
            )
        print(OK + "Stored bytes are encrypted, not plaintext")

    # 6. Clean up --------------------------------------------------------- #
    # ``Store`` has no delete — nothing in the app ever removes a document, it
    # writes tombstones instead. So reach past the interface for this one call.
    try:
        store._client.files_delete_v2(store._path(probe_path))  # noqa: SLF001
        print(OK + "Cleaned up the test document")
    except Exception:
        print(f"  ~ Left {folder}/{probe_path}.enc behind; delete it by hand if you like")

    print(
        "\nDropbox is ready. Accounts, saved lectures and settings will survive "
        "a restart.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
