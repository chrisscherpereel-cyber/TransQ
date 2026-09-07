#!/usr/bin/env python3
"""Walk through Dropbox authorization and print the secrets block to paste.

Steps 4 and 5 of docs/DROPBOX.md are where this goes wrong. They ask you to
hand-assemble a URL, copy a code out of a web page, and run a curl command with
two credentials in it — while the code expires in minutes. Miss
``token_access_type=offline`` and you get a token that dies in four hours;
generate the code before submitting the permissions and you get one with the
wrong scopes. Neither mistake announces itself: the app just quietly stores
everything on a disk that gets wiped.

So this does those steps for you. Run it, paste two values from the App Console,
approve in the browser, paste the code back:

    python3 scripts/setup_dropbox.py

It prints a finished secrets block. It never writes to your app or your
Dropbox — the output is yours to paste where you want it.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AUTHORIZE = "https://www.dropbox.com/oauth2/authorize"
TOKEN = "https://api.dropboxapi.com/oauth2/token"

REQUIRED_SCOPES = (
    "files.metadata.read",
    "files.metadata.write",
    "files.content.read",
    "files.content.write",
)

RULE = "─" * 68


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        raise SystemExit(1)


def fail(message: str, *hints: str) -> None:
    print(f"\n✗ {message}")
    for hint in hints:
        print(f"  {hint}")
    raise SystemExit(1)


def exchange(app_key: str, app_secret: str, code: str) -> dict:
    """Trade the one-time authorization code for a long-lived refresh token."""
    data = urllib.parse.urlencode(
        {"code": code, "grant_type": "authorization_code"}
    ).encode()
    request = urllib.request.Request(TOKEN, data=data, method="POST")

    import base64

    credentials = base64.b64encode(f"{app_key}:{app_secret}".encode()).decode()
    request.add_header("Authorization", f"Basic {credentials}")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        if "invalid_grant" in body:
            fail(
                "Dropbox rejected the authorization code.",
                "Codes are single-use and expire within minutes.",
                "Run this script again and paste the code promptly.",
            )
        if "invalid_client" in body:
            fail(
                "Dropbox rejected the app key or secret.",
                "Re-copy both from the App Console Settings tab.",
                "Watch for a trailing space.",
            )
        fail(f"Dropbox returned {exc.code}: {body[:300]}")
    except Exception as exc:  # pragma: no cover - network shapes vary
        fail(f"Could not reach Dropbox: {exc}")
    return {}


def main() -> int:
    print(RULE)
    print("Dropbox setup for Lecture Quiz Builder")
    print(RULE)
    print(
        "\nBefore starting, in the Dropbox App Console "
        "(dropbox.com/developers/apps):\n"
        "  1. Create app → Scoped access → App folder → name it\n"
        "  2. Permissions tab → tick these four:\n"
        + "".join(f"       {s}\n" for s in REQUIRED_SCOPES)
        + "  3. Press Submit\n"
        "\nStep 3 matters more than it looks. A token carries whatever scopes\n"
        "it had when issued, so a token made before Submit keeps the old ones\n"
        "forever — and this script would hand you credentials that fail later\n"
        "for reasons that point nowhere near here.\n"
    )
    if ask("Have you ticked all four and pressed Submit? [y/N] ").lower() not in (
        "y",
        "yes",
    ):
        print("\nDo that first, then run this again. Nothing has been changed.")
        return 1

    print("\nFrom the App Console → Settings tab:")
    app_key = ask("  App key:    ")
    app_secret = ask("  App secret: ")
    if not app_key or not app_secret:
        fail("Both values are required.")
    # Surrounding whitespace is stripped rather than rejected: a copied
    # credential picks up a trailing space constantly, and that is this script's
    # problem to absorb, not the user's to notice.

    url = f"{AUTHORIZE}?" + urllib.parse.urlencode(
        {
            "client_id": app_key,
            "response_type": "code",
            # Without this Dropbox returns only a 4-hour access token, and the
            # app breaks over lunch with an expiry nobody connects to setup.
            "token_access_type": "offline",
        }
    )

    print(f"\n{RULE}\nOpen this and click Allow:\n\n{url}\n{RULE}")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    code = ask("\nPaste the authorization code Dropbox showed you: ")
    if not code:
        fail("No code entered.")

    print("\nExchanging it for a refresh token…")
    payload = exchange(app_key, app_secret, code)

    refresh_token = payload.get("refresh_token", "")
    if not refresh_token:
        fail(
            "Dropbox did not return a refresh token.",
            "That happens without token_access_type=offline — but this script",
            "always sends it, so try again with a freshly generated code.",
        )

    granted = set((payload.get("scope") or "").split())
    missing = [s for s in REQUIRED_SCOPES if s not in granted]
    if missing:
        fail(
            "The token is missing permissions: " + ", ".join(missing),
            "Tick them in the App Console → Permissions, press Submit,",
            "then run this script again — this token cannot be upgraded.",
        )
    print("✓ All four permissions granted")

    print(f"\n{RULE}\nPaste this into your Streamlit secrets\n"
          "(Manage app → Settings → Secrets), replacing the empty placeholders:\n")
    print(f'DROPBOX_APP_KEY = "{app_key}"')
    print(f'DROPBOX_APP_SECRET = "{app_secret}"')
    print(f'DROPBOX_REFRESH_TOKEN = "{refresh_token}"')
    print(f"\n{RULE}")
    print(
        "Keep APP_SECRET as it is — that is your encryption key and is unrelated\n"
        "to the Dropbox 'App secret' above.\n\n"
        "The app restarts automatically after you save. The sidebar should then\n"
        "read 'Storage: Dropbox'. To test the credentials before deploying them:\n"
        "  python3 scripts/check_dropbox.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
