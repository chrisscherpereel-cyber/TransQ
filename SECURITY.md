# Security: what this app protects, and what it does not

All five recommendations from the original assessment are now implemented. This
document records what each one does, what it deliberately costs, and what is
still true that you should know before sharing the URL.

| # | Recommendation | Status |
|---|---|---|
| 1 | Issue capped, revocable keys instead of protecting unlimited ones | ✅ `src/provisioning.py` |
| 2 | Envelope encryption keyed on the user's password | ✅ `src/keymgmt.py`, `src/accounts.py` |
| 3 | Fix the key derivation, purpose separation, and rotation | ✅ `src/keymgmt.py`, `scripts/rotate_key.py` |
| 4 | Split the store per user; detect rollback | ✅ `src/accounts.py`, `src/storage.py` |
| 5 | Close the operational gaps | ✅ lockout, timeout, audit, self-revoke, masking |

---

## The threat this was always about

The original weakness was not the cipher. It was that a single `APP_SECRET`
decrypted everything and lived in the same process as the code using it. Anyone
who could read the Streamlit dashboard's App settings could decrypt every saved
API key; so could any leaked stack trace or compromised dependency.

The fix was not a stronger algorithm. It was changing **who can decrypt**.

---

## 1. Issued keys — capped and revocable

An administrator holds one **management key** and mints a separate OpenRouter key
per person, with a hard credit ceiling that resets monthly.

- A leaked key costs at most that person's cap, not your balance.
- Revocation is one API call. Deleting an account revokes its key upstream first.
- Nobody has to hand you a personal credential, so you stop being a custodian.
- The key string is captured at creation (OpenRouter shows only a masked label
  afterwards) and stored with its `hash` in one write — a key can never be saved
  without the handle needed to revoke it.

Set it up under **Admin → Issued API keys**.

---

## 2. Envelope encryption — the app cannot read your personal key

```
password ──scrypt──► KEK (session memory only, never stored)
                       │ unwraps
                 wrapped_DEK ──► DEK ──► your personal API keys
                  (stored)     (session memory only)
```

Each account holds a random data key. It is stored only wrapped under a key
derived from that user's password, which is stored nowhere. Signing in unwraps it
into `st.session_state`; signing out or timing out discards it.

| Event | Before | Now |
|---|---|---|
| `APP_SECRET` leaks | Every personal key decryptable | None are |
| An administrator reads the store | Can decrypt everyone's keys | Cannot decrypt anyone's |
| The store is stolen | All keys exposed | Each password must be broken separately |
| **An admin resets a password** | Keys survived | **That user's personal keys are destroyed** |

That last row is the deliberate cost, and the admin UI states it before you
confirm. If a reset could recover the keys, so could an administrator — which is
the thing being prevented. There is no recovery copy wrapped under `APP_SECRET`,
because adding one would undo the entire change. A test asserts this so it is not
quietly "fixed" later.

**Issued keys work differently on purpose.** An administrator must be able to
create one for a user who is not present, so those are encrypted under the app
key. That is a weaker guarantee, acceptable only because an issued key is capped
and revocable — which is the argument for preferring them over personal keys.

---

## 3. Key derivation, purpose separation, rotation

Four changes, all in `src/keymgmt.py`:

- **scrypt (n=2¹⁵), not PBKDF2.** Memory-hard, so a GPU no longer helps much.
- **A random 16-byte salt per deployment**, recorded in a plaintext `keyring`
  document beside the data. A salt is not secret; what it buys is that
  precomputation against one deployment buys nothing against another. The old
  fixed salt made the key a pure function of `APP_SECRET` everywhere.
- **HKDF subkeys per purpose** — documents, API keys, and config each get their
  own. Previously the "second layer" around API keys used the same key as the
  file around them, so it protected against nothing. It is now real, and a test
  proves each subkey refuses the others' ciphertext.
- **A key version on every ciphertext**, with old keys retained as read-only
  fallbacks.

### Rotating the secret

```bash
python3 scripts/rotate_key.py --dry-run
python3 scripts/rotate_key.py --new-secret "$(python3 -c 'import secrets;print(secrets.token_urlsafe(48))')"
```

Every document is decrypted with whichever key wrote it and rewritten under the
new one — including credentials stored *inside* documents (issued keys, the
management key), which the outer envelope alone would have missed. Because old
keys stay readable during the run, an interruption leaves a store that still
opens; rerun it. Only when everything has moved are the old readers dropped, and
the old secret stops working.

Users' personal keys are untouched: this script cannot read them, and does not
need to.

**Upgrading an existing store is automatic.** A store written by the old scheme
has no keyring; one is created with modern parameters and the legacy PBKDF2 spec
is kept as a fallback reader, so existing data opens and every subsequent write
uses the new key.

---

## 4. Per-user documents and rollback detection

**One document per account** (`users/u/<username>` plus an index). Saving your
settings no longer rewrites everybody's record, which removes the Dropbox
write-collision problem for the common case and shrinks what a partial write can
damage. A pre-existing single-document store is migrated on first read.

**Rollback detection.** Fernet proves a blob is authentic; it cannot prove it is
*current*, and an old ciphertext stays valid forever. Every document now carries
a monotonic version, and the highest version seen is recorded separately.
Restoring an old copy — to re-enable a disabled account, or by an accidental
"restore previous version" in Dropbox — is refused rather than accepted.

This is a tripwire, not a vault: someone who rolls back both the document and the
watermark record defeats it. It catches the realistic case.

---

## 5. Operational

| Gap | What now happens |
|---|---|
| Unlimited password guessing | 5 failures locks the account for 15 minutes; an admin can unlock. The correct password is refused while locked |
| Username enumeration | Wrong password and unknown user return the identical message |
| Sessions never expiring | Signed out after 8 hours idle, discarding the data key with it |
| No record of key events | `src/audit.py` logs sign-ins, lockouts, account changes and key add/issue/revoke — never the credential. Admins see it under **Security log**, exportable as CSV |
| No self-service revocation | **My security** panel: change your password (keys survive), or forget every personal key at once |
| Keys rendered back into the UI | Only the last four characters are shown; input fields are never pre-filled with a real key |

---

## What is still true

**Losing `APP_SECRET` still loses the store.** Rotation is now possible, but only
while you still have the current secret. Back it up where you back up passwords.

**Issued keys and the management key are recoverable by the app.** That is by
design — an administrator has to mint keys for absent users. It also means the
management key is now the single most valuable credential in the deployment: it
can create and revoke keys on your OpenRouter account. Keep caps low, review
`openrouter.ai/settings/keys` occasionally, and rotate `APP_SECRET` if you ever
suspect exposure.

**Dropbox still has no transactions.** Per-user documents remove most collisions,
and revision-checked writes detect the rest, but this is file storage. Past
roughly ten active users, move to Postgres — `Store` in `src/storage.py` is a
small interface precisely so that is one new class.

**A determined attacker with full host access wins.** They can read the process
memory of a signed-in session and take that user's data key. Envelope encryption
raises the cost from "read one config value" to "compromise the running host at
the moment someone is signed in"; it does not make the app invulnerable.

**Local files do not survive Streamlit Community Cloud restarts.** Use Dropbox
there, or accounts vanish.

---

## If you think something leaked

1. **Revoke the issued keys.** Admin → each account → Revoke, or delete them at
   `openrouter.ai/settings/keys`. This stops the spend immediately.
2. **Rotate the management key** at OpenRouter and save the new one.
3. **Rotate `APP_SECRET`** with `scripts/rotate_key.py`.
4. **Tell users to re-enter personal keys** if you believe a password was
   compromised — those are sealed per user, so only that user's keys are at risk.
5. **Read the Security log** for when the account was last used and by whom.

---

## Not doing

- **A recovery path for personal keys.** It would hand decryption back to the
  server and undo recommendation 2.
- **Storing anything in the audit log that could reconstruct a credential.** A
  log that leaks what it audits is worse than no log.
- **Treating "encrypted at rest" as sufficient on shared hosting.** If the
  consequences of exposure matter, run this on a machine you control.
