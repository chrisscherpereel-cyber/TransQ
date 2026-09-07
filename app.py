"""Lecture Quiz Builder — Streamlit app.

Lecture audio (or an existing transcript) -> local Whisper -> LLM summary ->
reviewable multiple-choice bank -> exports for Canvas, Excel, Word, and PDF.

Accounts, saved settings, per-user API keys and the usage ledger are persisted
through :mod:`src.storage` (encrypted local files, or Dropbox).

Run locally:   streamlit run app.py
Deploy:        push to GitHub, then point Streamlit Community Cloud at app.py
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile
import time

import pandas as pd
import streamlit as st

from src import __version__
from src.accounts import (
    LOCKOUT_MINUTES,
    ROLES,
    AuthError,
    Session,
    User,
    UserDirectory,
    mask_key,
    suggest_password,
)
from src.audit import AuditLog
from src.diagnostics import PHASE_GENERATE, PHASE_SUMMARY, RunReport
from src.library import LibraryEntry, LibraryError, TranscriptLibrary
from src.appconfig import AppConfig
from src.config import (
    AUDIO_EXTENSIONS,
    DEFAULT_PROVIDER,
    PROVIDERS,
    WHISPER_MODELS,
    AppSettings,
    get_provider,
    get_secret,
)
from src.exporters import (
    export_csv,
    export_docx,
    export_markdown,
    export_pdf,
    export_qti12_canvas,
    export_qti21,
    export_xlsx,
)
from src.exporters.transcript_formats import (
    export_srt,
    export_timestamped_txt,
    export_txt,
    export_vtt,
)
from src.llm import LLMClient, LLMError, PRICING, register_pricing
from src.chunking import chunk_transcript
from src.mcq import coverage_report, generate_replacement, generate_question_set, validate_all
from src.openrouter_catalog import ORModel, load_models, pricing_map, vendors
from src.provisioning import (
    DEFAULT_LIMIT,
    LIMIT_PRESETS,
    RESET_PERIODS,
    KeyInfo,
    ProvisioningClient,
    ProvisioningError,
    key_name_for,
)
from src.schema import OPTION_LETTERS, Quiz, QuizMeta, Summary, Transcript, format_timestamp
from src.storage import (
    DROPBOX_KEYS,
    Cipher,
    DropboxStore,
    StorageError,
    build_store,
    dropbox_credentials,
)
from src.summarize import summarize_transcript
from src.hostinfo import available_memory_gb, check_model_fits, describe_host
from src.transcribe import (
    TranscriptionError,
    estimate_transcription_minutes,
    load_model,
    natural_sort_key,
    probe_duration,
    transcribe_parts,
)
from src.transcript_import import TranscriptImportError, import_transcript
from src.usage import LiveMeter, UsageLog

st.set_page_config(
    page_title="Lecture Quiz Builder",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# One hue for every chart in this app. Each chart here is single-series
# (daily cost, cost per model, cost per user), so there is no categorical
# palette to separate — a second colour would encode nothing.
CHART_COLOR = "#00539B"

# Sign the user out after this long with no interaction. A signed-in tab left on
# a shared office machine is the realistic exposure, not a stolen password.
IDLE_TIMEOUT_MINUTES = 8 * 60
MAX_ATTEMPTS_HINT = 5

SECRET_KEYS = (
    "APP_SECRET",
    "DATA_DIR",
    "DROPBOX_APP_KEY",
    "DROPBOX_APP_SECRET",
    "DROPBOX_REFRESH_TOKEN",
    "DROPBOX_FOLDER",
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_whisper(model_size: str, compute_type: str):
    """Whisper weights are large; load them once per app process."""
    return load_model(model_size, compute_type)


@st.cache_data(ttl=3600, show_spinner="Loading the OpenRouter model list…")
def get_openrouter_models(_nonce: int = 0) -> tuple[list[ORModel], str | None]:
    """The live catalog, refreshed hourly.

    ``_nonce`` is not used by the function — bumping it is how the refresh button
    busts Streamlit's cache without waiting out the hour.
    """
    return load_models()


@st.cache_resource(show_spinner=False)
def get_backend():
    """Storage, the account directory, the usage ledger, and app config."""
    secrets = {key: get_secret(key) for key in SECRET_KEYS}
    store, warning = build_store(secrets)
    secret = secrets.get("APP_SECRET", "")
    cipher = Cipher(secret) if secret else None
    audit = AuditLog(store)
    return (
        store,
        UserDirectory(store, cipher, audit),
        UsageLog(store),
        AppConfig(store, cipher),
        warning,
        audit,
    )


def init_state() -> None:
    defaults = {
        "transcript": None,
        "transcript_note": "",
        "summary": None,
        "quiz": None,
        "quiz_versions": [],
        "active_version": 0,
        "chunks": [],
        "generation_notes": [],
        "run_report": None,
        "library_id": None,
        "source_filename": "",
        # A queued multi-part transcription: one part is done per script run,
        # so this survives between runs. See run_transcription for why.
        "job": None,
        "catalog_nonce": 0,
        "user": None,
        "dek": None,
        "last_seen": 0.0,
        "meter": LiveMeter(),
        "session_cost": 0.0,
        "session_tokens": 0,
        "session_priced": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def current_user() -> User | None:
    return st.session_state.get("user")


def adopt(session: Session) -> None:
    """Take a successful sign-in: the account, and its unwrapped data key.

    The data key lives in ``st.session_state`` and nowhere else — not on disk,
    not in the store. Signing out or timing out discards it, and with it the
    ability to read that person's personal API keys.
    """
    st.session_state.user = session.user
    st.session_state.dek = session.dek
    st.session_state.last_seen = time.time()


def sign_out(message: str = "") -> None:
    st.session_state.user = None
    st.session_state.dek = None
    st.session_state.last_seen = 0.0
    if message:
        st.session_state["_signout_notice"] = message


def session_expired() -> bool:
    """Has this session been idle too long? Touches the clock as a side effect."""
    if current_user() is None:
        return False
    last = float(st.session_state.get("last_seen") or 0.0)
    if last and (time.time() - last) > IDLE_TIMEOUT_MINUTES * 60:
        return True
    st.session_state.last_seen = time.time()
    return False


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #


def first_run_setup(directory: UserDirectory) -> None:
    """Create the first administrator. Only reachable while no accounts exist."""
    st.title("🎓 Lecture Quiz Builder")
    st.subheader("First-run setup")
    st.write(
        "No accounts exist yet. Create the administrator account — it can add "
        "everyone else later."
    )

    with st.form("bootstrap"):
        username = st.text_input("Username", value="admin")
        display_name = st.text_input("Display name (optional)")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create administrator", type="primary")

    if submitted:
        if password != confirm:
            st.error("The passwords do not match.")
            return
        try:
            session = directory.bootstrap_admin(username, password, display_name)
        except (AuthError, StorageError) as exc:
            st.error(str(exc))
            return
        adopt(session)
        st.rerun()

    st.caption(
        f"Suggested strong password: `{suggest_password()}` "
        "(refresh the page for another)"
    )


def login_screen(directory: UserDirectory) -> None:
    st.title("🎓 Lecture Quiz Builder")
    st.caption("Sign in to continue.")

    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")

    if submitted:
        try:
            adopt(directory.authenticate(username, password))
        except (AuthError, StorageError) as exc:
            st.error(str(exc))
            return
        st.rerun()

    st.caption(
        f"No account? Accounts are created by an administrator. "
        f"After {MAX_ATTEMPTS_HINT} failed attempts an account locks for "
        f"{LOCKOUT_MINUTES} minutes."
    )


def force_password_change(directory: UserDirectory, user: User) -> None:
    st.title("🎓 Lecture Quiz Builder")
    st.subheader("Choose a new password")
    st.info("This account was created with a temporary password. Set your own to continue.")

    with st.form("change_password"):
        password = st.text_input("New password", type="password")
        confirm = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Save password", type="primary")

    if submitted:
        if password != confirm:
            st.error("The passwords do not match.")
            return
        try:
            directory.set_password(
                user.username, password,
                current_dek=st.session_state.dek, actor=user.username,
            )
        except (AuthError, StorageError) as exc:
            st.error(str(exc))
            return
        adopt(directory.authenticate(user.username, password))
        st.toast("Password updated.", icon="✅")
        st.rerun()


def gate(directory: UserDirectory) -> bool:
    """True when a signed-in, ready user is available."""
    try:
        empty = directory.is_empty()
    except StorageError as exc:
        st.error(str(exc))
        return False

    if session_expired():
        sign_out(
            f"Signed out after {IDLE_TIMEOUT_MINUTES // 60} hours of inactivity."
        )

    notice = st.session_state.pop("_signout_notice", "")
    if notice:
        st.info(notice, icon="🔒")

    if empty:
        first_run_setup(directory)
        return False

    user = current_user()
    if user is None:
        login_screen(directory)
        return False
    if user.must_change_password:
        force_password_change(directory, user)
        return False
    return True


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #


def preference(user: User | None, key: str, default):
    """A saved setting for this account, falling back to the app default."""
    if user is None:
        return default
    return user.settings.get(key, default)


def openrouter_model_picker(default_slug: str) -> str:
    """Every model OpenRouter currently carries, A–Z, free ones marked.

    The list is fetched live rather than hardcoded: OpenRouter's roster turns
    over weekly, so a baked-in list would offer retired models and hide new ones.
    """
    catalog, warning = get_openrouter_models(st.session_state.catalog_nonce)
    register_pricing(pricing_map(catalog))

    if warning:
        st.warning(warning, icon="📶")

    free_only = st.checkbox(
        "Free models only",
        value=False,
        help="Models OpenRouter serves at $0. They are rate-limited and often "
        "smaller — fine for trying the app out, weaker at following the "
        "item-writing rules.",
    )
    chosen_vendors = st.multiselect(
        "Filter by vendor", vendors(catalog), default=[], placeholder="All vendors"
    )

    shown = [
        m
        for m in catalog
        if (not free_only or m.is_free)
        and (not chosen_vendors or m.vendor in chosen_vendors)
    ]
    if not shown:
        st.info("No models match those filters.")
        shown = catalog

    slugs = [m.id for m in shown]
    labels = {m.id: m.option_label for m in shown}
    index = slugs.index(default_slug) if default_slug in slugs else 0

    free_count = sum(1 for m in catalog if m.is_free)
    st.caption(
        f"{len(shown)} of {len(catalog)} models · {free_count} free · "
        f"{'bundled snapshot' if warning else 'live from openrouter.ai'}"
    )

    selected = st.selectbox(
        "Model",
        slugs,
        index=index,
        format_func=lambda slug: labels.get(slug, slug),
        help="Type to search. Sorted alphabetically; 🆓 marks models priced at $0.",
    )

    c1, c2 = st.columns(2)
    if c1.button("↻ Refresh list", use_container_width=True):
        st.session_state.catalog_nonce += 1
        get_openrouter_models.clear()
        st.rerun()
    custom = c2.text_input(
        "Or a slug", value="", placeholder="vendor/model",
        help="Anything not in the list — a brand-new model, or a variant.",
    ).strip()

    return custom or selected


def storage_status() -> None:
    """Say which backend is live, in the sidebar, always.

    A misconfigured Dropbox does not crash the app — it falls back to local
    files, which behave identically until a restart wipes them. That is the
    right runtime behaviour and a terrible way to find out, so the answer to
    "is my data actually going somewhere durable?" is on screen rather than
    inferred from the absence of a warning.
    """
    try:
        store, _, _, _, _, _ = get_backend()
    except StorageError:
        return

    name = getattr(store, "name", "") or "unknown"
    reason = getattr(store, "fallback_reason", "")

    if name == "Dropbox":
        st.caption("💾 Storage: **Dropbox** — survives restarts")
        return

    if "local" in name:
        st.caption("💾 Storage: **local encrypted files** — wiped on restart")
    else:
        st.caption(f"💾 Storage: **{name}**")

    # Point at the diagnostics panel rather than explaining here. This used to
    # be a collapsed expander in the sidebar and went unfound — a fourth
    # expander among three others, with a label unlike theirs. The panel on the
    # main page is where people actually look, so the explanation lives there
    # and this is a signpost.
    if reason:
        st.caption("Open **🩺 Diagnostics** on the main page to see why.")


def dropbox_credential_checklist() -> None:
    """Which of the three secrets the app can actually see.

    A tick list beats prose here: a misspelled key name in Streamlit secrets is
    invisible in every other view — the app simply behaves as though Dropbox was
    never configured — and one glance at this settles it.
    """
    present, missing = dropbox_credentials(
        {key: get_secret(key) for key in SECRET_KEYS}
    )
    st.markdown(
        "\n".join(
            f"- {'✅' if key in present else '❌'} `{key}`"
            for key in (*DROPBOX_KEYS,)
        )
    )
    if missing:
        st.caption(
            "❌ means the app cannot see that value at all — usually a "
            "misspelled name rather than a wrong secret. All three are required, "
            "spelled exactly as above. On Community Cloud they go in "
            "**Manage app → Settings → Secrets**; locally, in "
            "`.streamlit/secrets.toml`."
        )
    else:
        st.caption(
            "All three are present, so the names are right and the problem is "
            "the credentials themselves — most often a token generated before "
            "the four permissions were submitted in the Dropbox App Console. "
            "`python3 scripts/check_dropbox.py` tests them directly."
        )


def interrupted_run_detail(library: TranscriptLibrary, entry: LibraryEntry) -> None:
    """Per-part memory and timing for the run that died, with a reading of it.

    The point of recording these is that they discriminate. Memory rising part
    over part means the next part was always going to fail and the cause is
    inside the app. Memory flat means the app was healthy when something outside
    it stopped the process — a platform limit, a redeploy, a dropped connection —
    and no amount of tuning the model will help.
    """
    try:
        parts = library.load(entry.id).transcript.parts
    except (LibraryError, StorageError):
        return
    if not parts or not any(p.memory_gb for p in parts):
        st.caption(
            "This lecture was recorded before per-part telemetry existed, so "
            "there are no memory or timing figures for it. The next run will "
            "have them."
        )
        return

    st.markdown("**What the parts recorded before it stopped**")
    st.table(
        {
            "Part": [p.filename for p in parts],
            "Minutes taken": [f"{p.elapsed_seconds / 60:.1f}" for p in parts],
            "Memory after (GB)": [f"{p.memory_gb:.2f}" for p in parts],
            "Finished": [p.finished_at.replace("T", " ")[11:19] for p in parts],
        }
    )

    memories = [p.memory_gb for p in parts if p.memory_gb]
    budget = available_memory_gb()
    if len(memories) >= 2:
        growth = memories[-1] - memories[0]
        headroom = f" against {budget:.1f} GB available" if budget else ""
        if growth > 0.25:
            st.warning(
                f"Memory grew {growth:.2f} GB across {len(memories)} parts"
                f"{headroom}. That trend is the likely cause — the run was going "
                "to hit the ceiling eventually, and a longer recording would fail "
                "sooner. Transcribing fewer, shorter parts per run avoids it.",
                icon="📈",
            )
        else:
            st.info(
                f"Memory stayed flat (about {memories[-1]:.2f} GB{headroom}), so "
                "the app was healthy when it stopped. That points away from the "
                "model and towards something outside the app — a platform time "
                "limit, a redeploy, or a dropped browser connection. The Streamlit "
                "Cloud logs (**Manage app → logs**) will name it.",
                icon="📉",
            )

    total = sum(p.elapsed_seconds for p in parts)
    if total:
        st.caption(
            f"Total run time before it stopped: {total / 60:.0f} minutes across "
            f"{len(parts)} parts."
        )


def dropbox_live_probe() -> None:
    """Ask Dropbox, right now, and print exactly what it says.

    Everything else on this panel reports state captured when the app started
    and carried through several objects. That is one plumbing bug away from
    telling you nothing, which is precisely what happened: a deployment fell
    back to local files and the recorded explanation arrived empty, leaving the
    panel confidently silent about the only question that mattered.

    So this path is deliberately independent. It builds a client from the
    current secrets, makes a real call, and shows the raw result — no caching,
    no stored state, nothing to go stale between here and the failure.
    """
    if not st.button("🔌 Test the Dropbox connection now", use_container_width=True):
        return

    secrets = {key: get_secret(key) for key in SECRET_KEYS}
    present, missing = dropbox_credentials(secrets)
    if missing:
        st.error("Cannot test — missing: " + ", ".join(missing))
        return

    try:
        import dropbox  # noqa: F401
    except ImportError:
        st.error(
            "The `dropbox` package is not installed on this server, so the app "
            "cannot use Dropbox no matter how good the credentials are. It is "
            "listed in requirements.txt — if this is Streamlit Cloud, the "
            "install failed and the build logs will say why.",
            icon="📦",
        )
        return

    with st.spinner("Calling Dropbox…"):
        try:
            probe = DropboxStore(
                Cipher.__new__(Cipher),  # no crypto needed to test reachability
                secrets["DROPBOX_APP_KEY"].strip(),
                secrets["DROPBOX_APP_SECRET"].strip(),
                secrets["DROPBOX_REFRESH_TOKEN"].strip(),
                secrets.get("DROPBOX_FOLDER") or "/lecture-quiz-builder",
            )
            ok = probe.available()
        except Exception as exc:  # noqa: BLE001 - the raw text is the point
            st.error(f"The connection attempt raised: {exc}")
            return

    if ok:
        st.success(
            "Dropbox answered. The credentials are good — restart the app "
            "(**Manage app → Reboot**) and storage should switch over.",
            icon="✅",
        )
        return

    st.error(f"Dropbox refused: {probe.last_error or 'no detail returned'}", icon="🔌")
    st.caption(
        "That text comes straight from Dropbox. If it mentions a missing "
        "permission, the token was issued before the four scopes were submitted "
        "and must be regenerated — `python3 scripts/setup_dropbox.py` checks the "
        "scopes as it goes."
    )


def diagnostics_panel(settings: AppSettings, user: User) -> None:
    """Everything needed to explain a failed run, in one place.

    Written because two separate reports — "it says local files" and
    "transcription stops with no message" — turned out to be one situation.
    Checkpoints are written to storage; if storage is a container disk that the
    crash itself wipes, the evidence is destroyed by the same event that created
    it. Neither symptom is diagnosable alone, so they belong on one screen.
    """
    with st.expander("🩺 Diagnostics — why did my run fail?"):
        store = None
        try:
            store, _, _, _, _, _ = get_backend()
        except StorageError as exc:
            st.error(f"Storage could not start: {exc}")

        durable = getattr(store, "name", "") == "Dropbox"
        st.markdown(
            f"**Storage** · {getattr(store, 'name', 'unavailable')} "
            + ("✅ survives restarts" if durable else "❌ wiped on restart")
        )
        if not durable:
            reason = getattr(store, "fallback_reason", "")
            if reason:
                st.caption(reason)
            dropbox_credential_checklist()
            dropbox_live_probe()

        st.markdown(f"**This server** · {describe_host()}")
        verdict = check_model_fits(settings.whisper_model, settings.compute_type)
        icon = {"ok": "✅", "tight": "⚠️", "refused": "❌"}[verdict.level]
        st.markdown(
            f"**Whisper `{settings.whisper_model}`** · {icon} needs about "
            f"{verdict.needed_gb:.1f} GB"
        )
        if verdict.message:
            st.caption(verdict.message)

        library = get_library(user)
        unfinished = []
        if library is not None:
            try:
                unfinished = [e for e in library.entries() if not e.is_complete]
            except (LibraryError, StorageError):
                pass
        if unfinished:
            st.markdown("**Interrupted lectures**")
            for entry in unfinished:
                st.markdown(
                    f"- {entry.title} — {entry.progress_label}, "
                    f"stopped before `{entry.pending_parts[0]}`"
                )
            st.caption(
                "A run that stops here is where the process died. The part named "
                "is the one that was being transcribed."
            )
            interrupted_run_detail(library, unfinished[0])
        elif not durable:
            st.caption(
                "No interrupted lectures recorded — but with storage on a disk "
                "that a restart wipes, a crashed run erases its own evidence. "
                "Configure Dropbox and the next failure will leave a trail."
            )
        else:
            st.caption("No interrupted lectures.")


def sidebar(directory: UserDirectory, user: User) -> AppSettings:
    s = AppSettings()
    with st.sidebar:
        st.markdown(f"### 👤 {user.label}")
        st.caption(f"{user.role.title()} · signed in")
        storage_status()

        with st.expander("Transcription", expanded=False):
            whisper_default = preference(user, "whisper_model", "small")
            names = list(WHISPER_MODELS)
            s.whisper_model = st.selectbox(
                "Whisper model",
                names,
                index=names.index(whisper_default) if whisper_default in names else 2,
                help="Ignored when you import a transcript instead of audio.",
            )
            st.caption(WHISPER_MODELS[s.whisper_model])
            st.caption(f"This server: {describe_host()}")

            # Say it here, at the moment of choosing, as well as at run time.
            # Finding out that a model does not fit after uploading 80 MB of
            # audio and waiting ten minutes is the failure this replaces.
            verdict = check_model_fits(s.whisper_model, s.compute_type)
            if verdict.level == "refused":
                st.error(verdict.message, icon="🧠")
            elif verdict.level == "tight":
                st.warning(verdict.message, icon="🧠")

            languages = ["Auto-detect", "en", "es", "fr", "de", "zh", "hi", "pt"]
            lang = st.selectbox("Language", languages, index=0)
            s.language = None if lang == "Auto-detect" else lang
            s.vad_filter = st.checkbox("Skip silence (voice-activity filter)", value=True)
            s.beam_size = st.slider("Beam size", 1, 5, 1)
            st.checkbox(
                "Continue to the next part automatically",
                value=True,
                key="auto_continue",
                help="Each part is transcribed in its own short run, which is what "
                "keeps a long lecture from being cut off. Uncheck to press "
                "Continue yourself between parts.",
            )

        with st.expander("Question generation", expanded=True):
            provider_keys = list(PROVIDERS)
            saved_provider = preference(user, "provider", DEFAULT_PROVIDER)
            s.provider = st.selectbox(
                "LLM provider",
                provider_keys,
                index=provider_keys.index(saved_provider)
                if saved_provider in provider_keys
                else provider_keys.index(DEFAULT_PROVIDER),
                format_func=lambda k: PROVIDERS[k].label,
            )
            spec = get_provider(s.provider)
            if spec.note:
                st.caption(spec.note)

            saved_model = preference(user, "llm_model", spec.models[0])
            if s.provider == "openrouter":
                s.llm_model = openrouter_model_picker(
                    saved_model if saved_provider == s.provider else spec.models[0]
                )
            else:
                options = list(spec.models)
                s.llm_model = st.selectbox(
                    "Model",
                    options,
                    index=options.index(saved_model) if saved_model in options else 0,
                    key=f"model_{s.provider}",
                )

            s.api_key = account_key_controls(directory, user, s.provider)

            s.num_questions = st.slider(
                "Number of questions", 3, 40, int(preference(user, "num_questions", 10))
            )
            s.options_per_question = st.select_slider(
                "Options per question", [3, 4, 5],
                value=int(preference(user, "options_per_question", 4)),
            )
            s.bloom_targets = st.multiselect(
                "Cognitive levels (Bloom)",
                ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"],
                default=preference(
                    user, "bloom_targets", ["Remember", "Understand", "Apply", "Analyze"]
                ),
            )
            s.difficulty_mix = st.select_slider(
                "Difficulty", ["Mostly easy", "Balanced", "Mostly hard"],
                value=preference(user, "difficulty_mix", "Balanced"),
            )
            s.temperature = st.slider("Creativity", 0.0, 1.0, 0.3, 0.1)

        with st.expander("Context & advanced"):
            s.course_context = st.text_area(
                "Course context (optional)",
                value=preference(user, "course_context", ""),
                placeholder="MGT 301 Operations Management, junior-level. Emphasize "
                "capacity planning and the trade-offs between chase and level strategies.",
                height=110,
            )
            s.chunk_seconds = st.slider("Chunk length (minutes)", 3, 20, 10) * 60
            s.compute_type = st.selectbox(
                "Compute type", ["int8", "int8_float16", "float16", "float32"], index=0
            )

        if st.button("💾 Save these settings to my account", use_container_width=True):
            save_settings(directory, user, s)

        security_panel(directory, user)

        st.divider()
        live_meter_panel(s)
        st.divider()

        if st.button("Sign out", use_container_width=True):
            sign_out()
            st.rerun()
        st.caption(f"v{__version__} · audio is transcribed locally and never uploaded.")
    return s


MANAGEMENT_KEY_NAME = "openrouter_management_key"
PROVISIONED_PROVIDER = "openrouter"


def provisioning_client(config: AppConfig) -> ProvisioningClient | None:
    """A client for minting keys, or None when no management key is configured."""
    try:
        management_key = config.get_secret(MANAGEMENT_KEY_NAME)
    except StorageError:
        return None
    if not management_key:
        return None
    try:
        return ProvisioningClient(management_key)
    except ProvisioningError:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def key_usage(_management_key: str, key_hash: str) -> KeyInfo | None:
    """Live limit and spend for one issued key, cached for a minute.

    The leading underscore keeps the management key out of Streamlit's cache
    key — the hash alone identifies the record.
    """
    try:
        return ProvisioningClient(_management_key).get_key(key_hash)
    except ProvisioningError:
        return None


def spend_caption(info: KeyInfo) -> None:
    if info.is_capped:
        st.progress(info.fraction_used, text=f"{info.status_label} used this period")
    else:
        st.caption("This key has no spending cap set.")


def account_key_controls(directory: UserDirectory, user: User, provider: str) -> str:
    """Per-account API key: personal, issued, or from the app's own secrets.

    A key is never rendered back into a widget — only its last four characters
    are shown. A password field pre-filled with a real key puts it in the page,
    where a screenshot or a browser extension can reach it.
    """
    spec = get_provider(provider)
    dek = st.session_state.get("dek")
    key, origin = directory.resolve_key(user, provider, dek)
    record = directory.provisioned_record(user, provider)

    # A key this app minted: show what is left on it rather than asking for one
    # the person never had to supply.
    if origin == "issued" and record.get("hash"):
        _, _, _, config, _, _ = get_backend()
        management_key = ""
        try:
            management_key = config.get_secret(MANAGEMENT_KEY_NAME)
        except StorageError:
            pass

        st.success(f"{spec.label} key issued to you ({mask_key(key)})", icon="\U0001F511")
        info = key_usage(management_key, record["hash"]) if management_key else None
        if info is not None:
            spend_caption(info)
            if info.disabled:
                st.error("This key has been disabled. Ask your administrator.", icon="\U0001F6AB")
        elif record.get("limit"):
            st.caption(
                f"Capped at ${float(record['limit']):,.2f} per "
                f"{record.get('limit_reset', 'month')}."
            )
        return key

    if origin == "personal":
        c1, c2 = st.columns([2, 1])
        c1.success(f"Your {spec.label} key ({mask_key(key)})", icon="\U0001F511")
        if c2.button("Remove", key=f"rm_{provider}", use_container_width=True):
            try:
                directory.save_api_key(user.username, provider, "", dek, actor=user.username)
                st.session_state.user = directory.get(user.username)
                st.rerun()
            except (AuthError, StorageError) as exc:
                st.error(str(exc))
        st.caption("Encrypted so that only your sign-in can read it.")
        return key

    # A saved personal key exists but this session cannot open it — the envelope
    # working as designed after an administrator reset the password.
    if directory.has_api_key(user, provider) and dek is None:
        st.warning(
            "A key is saved on this account but this session cannot decrypt it. "
            "Sign out and back in; if your password was reset, the old key is "
            "gone and you will need to enter a new one.",
            icon="\U0001F512",
        )

    environment_key = get_secret(spec.env_var)
    if environment_key:
        st.info(f"Using {spec.env_var} from the app's secrets.", icon="\U0001F511")
        return environment_key

    typed = st.text_input(
        f"{spec.env_var}", type="password", value="",
        help="Encrypted with a key derived from your password — nobody else, "
        "including an administrator, can read it back.",
    )
    if typed and st.button("Save this key to my account", use_container_width=True):
        try:
            directory.save_api_key(user.username, provider, typed, dek, actor=user.username)
            st.session_state.user = directory.get(user.username)
            st.toast("API key saved.", icon="\U0001F511")
            st.rerun()
        except (AuthError, StorageError) as exc:
            st.error(str(exc))
    if spec.console_url and not typed:
        st.caption(f"[Get a {spec.label} key]({spec.console_url})")
    return typed


def security_panel(directory: UserDirectory, user: User) -> None:
    """Self-service: change a password, or drop every saved key at once."""
    with st.expander("\U0001F512 My security"):
        saved = len(user.api_keys)
        st.caption(
            f"{saved} personal key(s) saved \u00b7 "
            f"{len(user.issued_keys)} issued to you \u00b7 "
            f"session expires after {IDLE_TIMEOUT_MINUTES // 60}h idle"
        )

        with st.form("self_password"):
            st.markdown("**Change my password**")
            current = st.text_input("Current password", type="password")
            fresh = st.text_input("New password", type="password")
            confirm = st.text_input("Confirm new password", type="password")
            if st.form_submit_button("Update password"):
                if fresh != confirm:
                    st.error("The new passwords do not match.")
                else:
                    try:
                        session = directory.authenticate(user.username, current)
                        directory.set_password(
                            user.username, fresh,
                            current_dek=session.dek, actor=user.username,
                        )
                        adopt(directory.authenticate(user.username, fresh))
                        st.toast("Password updated; your saved keys are intact.", icon="\u2705")
                        st.rerun()
                    except (AuthError, StorageError) as exc:
                        st.error(str(exc))

        if saved:
            st.markdown("**Revoke my saved keys**")
            st.caption(
                "Removes every personal key from this account. Keys issued to you "
                "by an administrator are not affected — ask them to revoke those."
            )
            if st.button("Forget all my personal keys", use_container_width=True):
                try:
                    removed = directory.clear_all_api_keys(user.username, actor=user.username)
                    st.session_state.user = directory.get(user.username)
                    st.toast(f"Removed {removed} saved key(s).", icon="\U0001F9F9")
                    st.rerun()
                except (AuthError, StorageError) as exc:
                    st.error(str(exc))


def save_settings(directory: UserDirectory, user: User, s: AppSettings) -> None:
    try:
        directory.save_settings(
            user.username,
            {
                "provider": s.provider,
                "llm_model": s.llm_model,
                "whisper_model": s.whisper_model,
                "num_questions": s.num_questions,
                "options_per_question": s.options_per_question,
                "bloom_targets": s.bloom_targets,
                "difficulty_mix": s.difficulty_mix,
                "course_context": s.course_context,
            },
        )
        st.session_state.user = directory.get(user.username)
        st.toast("Settings saved to your account.", icon="💾")
    except StorageError as exc:
        st.error(str(exc))


def live_meter_panel(s: AppSettings) -> None:
    """Tokens and dollars, updated after every model call rather than at the end."""
    meter: LiveMeter = st.session_state.meter
    st.markdown("**This run**")
    st.session_state["_meter_slot"] = st.empty()
    render_meter(st.session_state["_meter_slot"], meter)

    if st.session_state.session_tokens:
        st.caption(
            f"This session: {st.session_state.session_tokens:,} tokens · "
            + (
                f"${st.session_state.session_cost:,.4f}"
                if st.session_state.session_priced
                else "cost unavailable"
            )
        )
    if s.llm_model not in PRICING:
        st.caption("No published price for this model — tokens are tracked, cost is not.")


def render_meter(slot, meter: LiveMeter) -> None:
    with slot.container():
        c1, c2 = st.columns(2)
        c1.metric("Tokens", f"{meter.total_tokens:,}")
        c2.metric("Cost", meter.cost_label)
        st.caption(f"{meter.calls} model call(s)")


def attach_meter(client: LLMClient, s: AppSettings) -> LiveMeter:
    """Point a fresh meter at this client and stream updates into the sidebar."""
    meter: LiveMeter = st.session_state.meter
    meter.reset()
    meter.provider, meter.model = s.provider, s.llm_model
    slot = st.session_state.get("_meter_slot")

    def on_usage(input_tokens: int, output_tokens: int, rate) -> None:
        meter.record(input_tokens, output_tokens, rate)
        if slot is not None:
            render_meter(slot, meter)

    client.on_usage = on_usage
    return meter


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def order_uploads(files: list, use_upload_order: bool) -> list:
    """Put the parts of a split recording into lecture order."""
    if use_upload_order:
        return list(files)
    return sorted(files, key=lambda f: natural_sort_key(f.name))


def get_library(user: User) -> TranscriptLibrary | None:
    """This account's saved lectures, or None if nothing can be saved."""
    try:
        store, _, _, _, _, _ = get_backend()
        return TranscriptLibrary(store, user.username)
    except (StorageError, LibraryError):
        return None


def autosave_transcript(user: User, origin: str) -> None:
    """Persist the lecture the moment it exists.

    Transcription is the expensive step; losing it to a browser refresh is the
    kind of thing that makes people stop using a tool. Saving happens without
    being asked, and the entry is updated in place as the summary and question
    sets arrive.
    """
    library = get_library(user)
    transcript = st.session_state.transcript
    if library is None or transcript is None:
        return
    try:
        entry = library.save(
            transcript,
            title=st.session_state.get("source_filename", ""),
            origin=origin,
            # Update the entry the per-part checkpoints already created, rather
            # than filing a second copy of the same lecture beside it.
            entry_id=st.session_state.get("library_id"),
        )
        st.session_state.library_id = entry.id
        st.caption(f"Saved to your library as **{entry.title}**.")
    except (LibraryError, StorageError) as exc:
        st.warning(
            f"The transcript could not be saved to your library: {exc}", icon="💾"
        )


def update_saved_lecture(user: User) -> None:
    """Fold the summary and question sets into the saved entry."""
    library = get_library(user)
    transcript = st.session_state.transcript
    if library is None or transcript is None:
        return
    try:
        entry = library.save(
            transcript,
            summary=st.session_state.summary,
            quizzes=list(st.session_state.quiz_versions),
            entry_id=st.session_state.get("library_id"),
        )
        st.session_state.library_id = entry.id
    except (LibraryError, StorageError) as exc:
        st.warning(f"Your library entry was not updated: {exc}", icon="💾")


def reset_downstream() -> None:
    st.session_state.summary = None
    st.session_state.quiz = None
    st.session_state.quiz_versions = []
    st.session_state.active_version = 0
    st.session_state.chunks = []
    st.session_state.generation_notes = []


def checkpoint_saver(user: User, origin: str):
    """A callback that writes each finished part straight to the library.

    This is the fix for the failure that motivated it: a container killed on the
    last part of a split recording used to discard every part before it, because
    the only copy lived in a local list. Now each part is durable the moment it
    exists, and the entry is updated in place rather than duplicated.

    Deliberately silent. It runs mid-run, several times, and a stream of toasts
    would bury the progress bar. It is also best-effort — ``_checkpoint`` in
    :mod:`src.transcribe` swallows what this raises, because storage being
    briefly unavailable is a reason to keep transcribing, not to stop.
    """
    library = get_library(user)
    if library is None:
        return None

    def save(partial: Transcript) -> None:
        entry = library.save(
            partial,
            title=st.session_state.get("source_filename", ""),
            origin=origin,
            entry_id=st.session_state.get("library_id"),
        )
        st.session_state.library_id = entry.id

    return save


def discard_job() -> None:
    """Forget the current job and delete its temporary audio."""
    job = st.session_state.pop("job", None)
    if not job:
        return
    for path in job.get("paths", []):
        try:
            os.unlink(path)
        except OSError:
            pass


def run_transcription(
    uploaded: list,
    settings: AppSettings,
    user: User,
    resume_from: Transcript | None = None,
) -> None:
    """Queue a lecture and hand control back, one part per script run.

    The evidence that led here: a three-part lecture died on part 3 every time,
    and part 3 transcribed perfectly when run on its own. Memory was flat, the
    file was fine, the model fit. What failed was the *length of the run* — an
    unbroken half-hour of work in one Streamlit script execution.

    So the fix is not to make the work faster or lighter. It is to stop asking
    the platform for a long run at all: each script execution transcribes exactly
    one part, saves it, and schedules the next. Every run is then about as long
    as the run we know succeeds. Whatever the real ceiling is — a platform
    timeout, a watchdog, a dropped websocket — this stays under it without
    needing to know its value.

    The cost is honest: continuing depends on the browser triggering the next
    run, so a closed laptop pauses the lecture rather than finishing it. It does
    not *lose* anything, because every finished part is already saved, and the
    resume panel picks it up whenever you come back.
    """
    verdict = check_model_fits(settings.whisper_model, settings.compute_type)
    if not verdict.allowed:
        st.error(verdict.message, icon="🧠")
        return
    if verdict.level == "tight":
        st.warning(verdict.message, icon="🧠")

    discard_job()  # a new lecture supersedes anything half-queued
    paths: list[str] = []
    names: list[str] = []
    for item in uploaded:
        suffix = os.path.splitext(item.name)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(item.getbuffer())
            paths.append(tmp.name)
            names.append(item.name)

    if resume_from is None:
        # Name it before the first checkpoint, so what lands in the library is
        # recognisable rather than "Lecture (25:00)".
        st.session_state.library_id = None
        st.session_state.source_filename = (
            names[0] if len(names) == 1 else f"{os.path.splitext(names[0])[0]}_combined"
        )

    st.session_state.job = {
        "paths": paths,
        "names": names,
        "index": 0,
        "auto": st.session_state.get("auto_continue", True),
        "resuming": resume_from is not None,
    }
    st.rerun()


def transcription_job(settings: AppSettings, user: User) -> bool:
    """Run the next part of a queued lecture. Returns True if a job is active.

    One part per call, deliberately. See :func:`run_transcription` for why.
    """
    job = st.session_state.get("job")
    if not job:
        return False

    paths, names, index = job["paths"], job["names"], job["index"]
    total = len(paths)
    library = get_library(user)

    with st.container(border=True):
        st.markdown(f"### 🎙️ Transcribing — part {index + 1} of {total}")
        st.caption(f"`{names[index]}` · {index} of {total} parts done so far")
        st.progress(index / total)

        if index == 0 and total > 1:
            remaining = sum(probe_duration(p) for p in paths)
            if remaining:
                est = estimate_transcription_minutes(remaining, settings.whisper_model)
                st.caption(
                    f"About {est:.0f} minutes of work in total. Each part is saved "
                    "as it finishes and runs separately, so this survives an "
                    "interruption — but keep this tab open for it to continue."
                )

        # Continue from what is already saved, so the timeline joins correctly.
        resume_from = None
        if index > 0 or job["resuming"]:
            entry_id = st.session_state.get("library_id")
            if entry_id and library is not None:
                try:
                    resume_from = library.load(entry_id).transcript
                except (LibraryError, StorageError) as exc:
                    st.error(
                        f"The parts already transcribed could not be reloaded: {exc}"
                    )
                    discard_job()
                    return True

        bar = st.progress(0.0, text="Loading the Whisper model…")
        try:
            model = get_whisper(settings.whisper_model, settings.compute_type)
            transcript = transcribe_parts(
                model,
                [paths[index]],
                display_names=[names[index]],
                language=settings.language,
                vad_filter=settings.vad_filter,
                beam_size=settings.beam_size,
                progress=lambda f, m: bar.progress(min(1.0, f), text=m),
                on_part_complete=checkpoint_saver(user, origin="audio"),
                resume_from=resume_from,
                remaining_after=names[index + 1 :],
            )
        except TranscriptionError as exc:
            bar.empty()
            st.error(f"Part {index + 1} (`{names[index]}`) failed: {exc}")
            st.caption(
                "Earlier parts are saved. Fix or re-cut this file and use the "
                "resume panel to continue."
            )
            discard_job()
            return True
        bar.empty()

        st.session_state.transcript = transcript
        st.session_state.transcript_note = ""
        job["index"] = index + 1

        if job["index"] < total:
            st.success(f"Part {index + 1} done and saved.")
            if job["auto"]:
                st.rerun()
            st.button(
                f"▶️ Continue with part {job['index'] + 1}",
                type="primary",
                key=f"continue_{job['index']}",
            )
            st.button("Stop here", key=f"stop_{job['index']}", on_click=discard_job)
            return True

        # Finished.
        reset_downstream()
        st.success(
            f"Transcribed {format_timestamp(transcript.duration)} of audio across "
            f"{len(transcript.parts)} part(s) — {transcript.word_count:,} words, "
            f"{len(transcript.segments)} segments."
        )
        if transcript.skipped_parts:
            st.warning(
                "Some parts were skipped and are missing from the transcript:\n\n"
                + "\n".join(f"- {s}" for s in transcript.skipped_parts),
                icon="⚠️",
            )
        autosave_transcript(user, origin="audio")
        discard_job()
    return True


def run_transcript_import(uploaded, user: User) -> None:
    """Load a transcript that already exists, skipping Whisper entirely."""
    try:
        raw = uploaded.getvalue()
        text = raw.decode("utf-8", errors="replace")
        transcript, described = import_transcript(text, uploaded.name)
    except (TranscriptImportError, UnicodeDecodeError) as exc:
        st.error(f"Could not read that transcript: {exc}")
        return

    st.session_state.transcript = transcript
    st.session_state.transcript_note = described
    reset_downstream()
    st.session_state.library_id = None  # a new lecture, not an update to the last
    st.session_state.source_filename = uploaded.name

    st.success(
        f"Imported {described} — {transcript.word_count:,} words, "
        f"{len(transcript.segments)} segments, "
        f"{format_timestamp(transcript.duration)} of lecture."
    )
    if "estimated" in described:
        st.info(
            "This file had no timings, so timestamps are estimated from a normal "
            "speaking rate. Questions will still cite a position in the lecture, "
            "but treat it as approximate rather than a place to scrub to.",
            icon="🕒",
        )
    autosave_transcript(user, origin=described)


def build_client(settings: AppSettings) -> LLMClient | None:
    try:
        return LLMClient(
            provider=settings.provider,
            model=settings.llm_model,
            api_key=settings.resolved_api_key(),
            temperature=settings.temperature,
        )
    except LLMError as exc:
        st.error(str(exc))
        return None


def commit_usage(meter: LiveMeter, user: User, operation: str) -> None:
    """Add this run to the persistent ledger and the session totals."""
    if meter.calls == 0:
        return
    st.session_state.session_tokens += meter.total_tokens
    st.session_state.session_cost += meter.cost
    if not meter.cost_known:
        st.session_state.session_priced = False
    try:
        _, _, usage_log, _, _, _ = get_backend()
        usage_log.append(
            meter.to_record(user.username, operation, st.session_state.source_filename)
        )
    except StorageError as exc:
        st.warning(f"Usage was not recorded: {exc}", icon="📉")


def all_existing_stems() -> list[str]:
    stems: list[str] = []
    for quiz in st.session_state.quiz_versions:
        stems.extend(q.stem for q in quiz.questions)
    return stems


def store_version(quiz: Quiz) -> None:
    quiz.meta.title = quiz.meta.title or "Lecture Quiz"
    st.session_state.quiz_versions.append(quiz)
    st.session_state.active_version = len(st.session_state.quiz_versions) - 1
    st.session_state.quiz = quiz


def build_quiz(questions, summary: Summary, settings: AppSettings, version: int) -> Quiz:
    base = summary.title or "Lecture Quiz"
    return Quiz(
        meta=QuizMeta(
            title=base if version == 1 else f"{base} — Set {version}",
            course=settings.course_context.splitlines()[0][:80]
            if settings.course_context
            else "",
            description=summary.abstract[:400],
            source_filename=st.session_state.source_filename,
            generated_on=dt.date.today().isoformat(),
            model_used=f"{get_provider(settings.provider).label} {settings.llm_model}",
        ),
        questions=questions,
    )


def run_generation(settings: AppSettings, user: User, do_review: bool) -> None:
    transcript = st.session_state.transcript
    client = build_client(settings)
    if client is None or transcript is None:
        return
    meter = attach_meter(client, settings)

    # The report outlives the progress bar. That is the whole point: a run that
    # failed used to look exactly like a lecture with nothing to say.
    report = RunReport()
    st.session_state.run_report = report

    bar = st.progress(0.0, text="Summarizing…")
    try:
        summary, chunks, _ = summarize_transcript(
            client,
            transcript,
            course_context=settings.course_context,
            chunk_seconds=settings.chunk_seconds,
            overlap_seconds=settings.chunk_overlap_seconds,
            progress=lambda f, m: bar.progress(f * 0.35, text=m),
            report=report,
        )
        st.session_state.summary = summary
        st.session_state.chunks = chunks

        questions, notes = generate_question_set(
            client,
            chunks,
            target=settings.num_questions,
            n_options=settings.options_per_question,
            bloom_targets=settings.bloom_targets,
            difficulty_mix=settings.difficulty_mix,
            course_context=settings.course_context,
            do_review=do_review,
            progress=lambda f, m: bar.progress(0.35 + f * 0.65, text=m),
            report=report,
        )
        st.session_state.generation_notes = notes
        store_version(build_quiz(questions, summary, settings, version=1))
        update_saved_lecture(user)
        bar.empty()
        report_outcome(report, questions, notes, settings)
    except LLMError as exc:
        bar.empty()
        st.error(f"**The run stopped early.** {exc}", icon="🛑")
        show_run_report(report, expanded=True)
        if report.advice():
            st.info(report.advice(), icon="💡")
    finally:
        commit_usage(meter, user, "generate")


def report_outcome(report: RunReport, questions, notes, settings: AppSettings) -> None:
    """Say what happened — including the parts that failed silently before."""
    included = len([q for q in questions if q.include])
    failures = report.failures

    if not questions:
        st.error(
            "**No questions were produced.** " + report.headline(), icon="🛑"
        )
    elif failures or included < settings.num_questions:
        st.warning(
            f"**Finished with gaps.** {included} of the "
            f"{settings.num_questions} requested. {report.headline()}",
            icon="⚠️",
        )
    else:
        st.success(f"Generated {included} questions. {report.headline()}")

    for note in notes:
        st.caption(f"· {note}")

    if failures:
        if report.advice():
            st.info(report.advice(), icon="💡")
        show_run_report(report, expanded=True)
    else:
        show_run_report(report, expanded=False)


def show_run_report(report: RunReport, expanded: bool = False) -> None:
    """Every model call, and how it went."""
    if not report.steps:
        return
    failures = len(report.failures)
    title = (
        f"⚠️ Run report — {failures} failed call(s)"
        if failures
        else f"Run report — {len(report.steps)} model calls, all fine"
    )
    with st.expander(title, expanded=expanded):
        st.dataframe(
            pd.DataFrame(report.as_rows()),
            use_container_width=True,
            hide_index=True,
        )
        if failures:
            st.caption(
                "Steps marked ✗ never reached the model, or came back unusable. "
                "The questions you have were built from the steps that did work."
            )


def run_alternative_set(settings: AppSettings, user: User, do_review: bool) -> bool:
    """Generate a fresh set of questions over the same lecture."""
    chunks = st.session_state.chunks
    summary = st.session_state.summary
    client = build_client(settings)
    if client is None or not chunks or summary is None:
        st.error("Run the first generation pass before asking for an alternative set.")
        return False
    meter = attach_meter(client, settings)

    bar = st.progress(0.0, text="Writing an alternative set…")
    try:
        report = RunReport()
        st.session_state.run_report = report
        questions, notes = generate_question_set(
            client,
            chunks,
            target=settings.num_questions,
            n_options=settings.options_per_question,
            bloom_targets=settings.bloom_targets,
            difficulty_mix=settings.difficulty_mix,
            course_context=settings.course_context,
            do_review=do_review,
            progress=lambda f, m: bar.progress(f, text=m),
            report=report,
        )
        bar.empty()
        if not questions:
            st.warning(f"The alternative pass produced nothing usable. {report.headline()}")
            show_run_report(report, expanded=True)
            if report.advice():
                st.info(report.advice(), icon="💡")
            return False

        st.session_state.generation_notes = notes
        store_version(
            build_quiz(
                questions, summary, settings, version=len(st.session_state.quiz_versions) + 1
            )
        )
        update_saved_lecture(user)
        st.toast(
            f"Added Set {len(st.session_state.quiz_versions)} — "
            f"{len([q for q in questions if q.include])} new questions.",
            icon="✨",
        )
        return True
    except LLMError as exc:
        bar.empty()
        st.error(f"**The alternative pass stopped early.** {exc}", icon="🛑")
        return False
    finally:
        commit_usage(meter, user, "alternative")


def run_replacement(
    settings: AppSettings, user: User, index: int, same_section: bool = True
) -> None:
    """Swap one question for a newly written one."""
    quiz: Quiz | None = st.session_state.quiz
    chunks = st.session_state.chunks
    if quiz is None or not chunks:
        return
    client = build_client(settings)
    if client is None:
        return
    meter = attach_meter(client, settings)
    report = RunReport()

    old = quiz.questions[index]
    try:
        with st.spinner("Writing a replacement question…"):
            new = generate_replacement(
                client,
                chunks,
                old,
                existing_stems=all_existing_stems(),
                n_options=settings.options_per_question,
                bloom_targets=settings.bloom_targets,
                difficulty_mix=settings.difficulty_mix,
                course_context=settings.course_context,
                same_section=same_section,
                report=report,
            )
    except LLMError as exc:
        st.error(str(exc))
        return
    finally:
        commit_usage(meter, user, "replace")

    if new is None:
        st.warning(
            f"Could not write a replacement for that question. {report.headline()}"
        )
        show_run_report(report, expanded=bool(report.failures))
        if report.advice():
            st.info(report.advice(), icon="💡")
        return

    quiz.questions[index] = new
    validate_all(quiz.questions)
    st.toast(f"Replaced question {index + 1}.", icon="🔄")
    st.rerun()


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #


def resume_panel(settings: AppSettings, user: User) -> bool:
    """Offer to finish a lecture that was interrupted partway through.

    Shown before anything else, because a half-transcribed lecture sitting in
    the library is the most likely reason someone reopened the app. Re-uploading
    the remaining parts costs only the parts that were never done.

    Returns True when it rendered, so the caller can put a divider under it.
    """
    library = get_library(user)
    if library is None:
        return False
    try:
        unfinished = [e for e in library.entries() if not e.is_complete]
    except (LibraryError, StorageError):
        return False
    if not unfinished:
        return False

    entry = unfinished[0]
    with st.container(border=True):
        st.markdown(f"### ⏸️ Unfinished: {entry.title}")
        st.caption(
            f"{entry.progress_label} · {entry.length_label} saved so far. "
            "The parts already done are safe in your library."
        )
        st.markdown(
            "**Still to transcribe:** "
            + ", ".join(f"`{name}`" for name in entry.pending_parts)
        )

        uploads = st.file_uploader(
            "Upload the remaining part(s) to finish this lecture",
            type=AUDIO_EXTENSIONS,
            accept_multiple_files=True,
            key=f"resume_{entry.id}",
            help="Only the parts listed above. They are appended to the saved "
            "timeline, so timestamps stay correct across the whole lecture.",
        )
        ordered = order_uploads(uploads, use_upload_order=False) if uploads else []

        # Names rarely match exactly — people re-export or rename. Warn, but do
        # not block: the user knows which file is which better than we do.
        if ordered:
            unexpected = [
                f.name for f in ordered if f.name not in entry.pending_parts
            ]
            if unexpected:
                st.warning(
                    "These do not match the expected filenames: "
                    + ", ".join(f"`{n}`" for n in unexpected)
                    + ". They will be appended in the order shown — check that is "
                    "the right order before continuing.",
                    icon="⚠️",
                )
            st.caption("**Order:** " + " → ".join(f.name for f in ordered))

        c1, c2, _ = st.columns([1, 1, 2])
        if c1.button(
            "▶️ Finish transcribing",
            type="primary",
            disabled=not ordered,
            use_container_width=True,
        ):
            try:
                saved = library.load(entry.id)
            except (LibraryError, StorageError) as exc:
                st.error(f"That lecture could not be reopened: {exc}")
                return True
            st.session_state.library_id = entry.id
            st.session_state.source_filename = entry.title
            run_transcription(ordered, settings, user, resume_from=saved.transcript)

        if c2.button(
            "Keep what I have",
            use_container_width=True,
            help="Marks the lecture finished with the parts already transcribed. "
            "You can summarize and generate questions from a partial lecture.",
        ):
            try:
                saved = library.load(entry.id)
                saved.transcript.pending_parts = []
                library.save(
                    saved.transcript,
                    title=entry.title,
                    summary=saved.summary,
                    quizzes=saved.quizzes,
                    origin=entry.origin,
                    entry_id=entry.id,
                )
                st.toast("Marked as finished.", icon="✅")
                st.rerun()
            except (LibraryError, StorageError) as exc:
                st.error(str(exc))

    return True


def input_section(settings: AppSettings, user: User, review: bool) -> None:
    # A job in flight owns the screen: showing an upload form underneath an
    # active transcription invites starting a second one on top of the first.
    if transcription_job(settings, user):
        return

    if resume_panel(settings, user):
        st.divider()

    mode = st.radio(
        "Where is the lecture coming from?",
        ["🎙️ Audio file(s)", "📄 An existing transcript"],
        horizontal=True,
        help="Already have captions from Panopto, Zoom or YouTube? Import them and "
        "skip transcription entirely — it is faster and usually more accurate.",
    )

    if mode.startswith("🎙️"):
        uploaded = st.file_uploader(
            "Lecture audio or video — one file, or several parts of a split recording",
            type=AUDIO_EXTENSIONS,
            accept_multiple_files=True,
            help="Split a long lecture into parts if a single file is too big or slow. "
            "The parts are transcribed in order and stitched into one transcript.",
        )
        ordered: list = []
        if uploaded:
            use_upload_order = False
            if len(uploaded) > 1:
                use_upload_order = st.checkbox(
                    "Use the order I uploaded them in",
                    value=False,
                    help="Off: ordered by filename, with numbers read as numbers "
                    "(so part2 comes before part10).",
                )
            ordered = order_uploads(uploaded, use_upload_order)
            if len(ordered) > 1:
                st.caption(
                    "**Transcription order:** " + " → ".join(f.name for f in ordered)
                )
        action_label, ready, runner = "1 · Transcribe", bool(ordered), (
            lambda: run_transcription(ordered, settings, user)
        )
    else:
        imported = st.file_uploader(
            "Transcript file",
            type=["txt", "srt", "vtt", "md", "text"],
            help="SRT or WebVTT keeps real timings. Timestamped text works too. "
            "Plain text is fine — timings are then estimated.",
        )
        action_label, ready, runner = "1 · Import transcript", imported is not None, (
            lambda: run_transcript_import(imported, user)
        )

    a1, a2, _ = st.columns([1, 1, 2])
    if a1.button(action_label, type="primary", disabled=not ready, use_container_width=True):
        runner()
    if a2.button(
        "2 · Summarize & generate",
        disabled=st.session_state.transcript is None,
        use_container_width=True,
    ):
        run_generation(settings, user, review)


def transcript_tab() -> None:
    transcript: Transcript | None = st.session_state.transcript
    if transcript is None:
        st.info("Upload audio, or import a transcript, to get started.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", format_timestamp(transcript.duration))
    c2.metric("Words", f"{transcript.word_count:,}")
    c3.metric("Segments", len(transcript.segments))
    c4.metric(
        "Parts" if transcript.is_multipart else "Language",
        len(transcript.parts) if transcript.is_multipart else transcript.language.upper(),
    )

    if st.session_state.transcript_note:
        st.caption(f"Source: {st.session_state.transcript_note}")

    if transcript.is_multipart:
        with st.expander("Parts stitched into this transcript"):
            st.caption(
                "Timestamps are on the combined lecture timeline, so a question "
                "tagged 0:52:14 points at the same moment whether the recording "
                "arrived as one file or five."
            )
            for part in transcript.parts:
                st.markdown(f"- {part.label} — {part.segments} segments")

    show_times = st.toggle("Show timestamps", value=True)
    text = transcript.text_with_timestamps() if show_times else transcript.text
    st.text_area("Transcript", text, height=420, label_visibility="collapsed")

    st.markdown("**Download transcript**")
    d1, d2, d3, d4 = st.columns(4)
    stem = os.path.splitext(st.session_state.source_filename or "lecture")[0]
    d1.download_button("Plain text", export_txt(transcript), f"{stem}.txt", "text/plain",
                       use_container_width=True)
    d2.download_button("Timestamped", export_timestamped_txt(transcript),
                       f"{stem}_timestamped.txt", "text/plain", use_container_width=True)
    d3.download_button("SRT captions", export_srt(transcript), f"{stem}.srt", "text/plain",
                       use_container_width=True)
    d4.download_button("WebVTT", export_vtt(transcript), f"{stem}.vtt", "text/vtt",
                       use_container_width=True)


def summary_tab() -> None:
    summary: Summary | None = st.session_state.summary
    if summary is None:
        st.info("Run **Summarize & generate** to see the summary.")
        return

    if summary.title:
        st.subheader(summary.title)
    if summary.abstract:
        st.write(summary.abstract)

    left, right = st.columns(2)
    with left:
        if summary.learning_objectives:
            st.markdown("#### Learning objectives")
            for obj in summary.learning_objectives:
                st.markdown(f"- {obj}")
        if summary.key_points:
            st.markdown("#### Key points")
            for point in summary.key_points:
                st.markdown(f"- {point}")
    with right:
        if summary.outline:
            st.markdown("#### Outline")
            for item in summary.outline:
                st.markdown(
                    f"**{item.get('timestamp', '')} · {item.get('heading', '')}**  \n"
                    f"{item.get('detail', '')}"
                )
        if summary.key_terms:
            st.markdown("#### Key terms")
            for term in summary.key_terms:
                st.markdown(f"- **{term.get('term', '')}** — {term.get('definition', '')}")

    st.download_button(
        "Download summary (Markdown)", summary.as_markdown().encode("utf-8"),
        "lecture_summary.md", "text/markdown",
    )


def version_bar(settings: AppSettings, user: User, review: bool) -> None:
    versions = st.session_state.quiz_versions
    left, right = st.columns([3, 2])

    with left:
        if len(versions) > 1:
            labels = [f"Set {i + 1} ({len(v.included)} questions)" for i, v in enumerate(versions)]
            picked = st.radio(
                "Question set", range(len(versions)),
                index=st.session_state.active_version,
                format_func=lambda i: labels[i], horizontal=True,
            )
            if picked != st.session_state.active_version:
                st.session_state.active_version = picked
                st.session_state.quiz = versions[picked]
                st.rerun()
        else:
            st.caption("One set generated so far.")

    with right:
        if st.button(
            "✨ Generate an alternative set", use_container_width=True,
            help="A whole new set over the same lecture, steered away from every "
            "question you already have. The old set is kept.",
        ):
            if run_alternative_set(settings, user, review):
                st.rerun()


def questions_tab(settings: AppSettings, user: User, review: bool) -> None:
    quiz: Quiz | None = st.session_state.quiz
    if quiz is None or not quiz.questions:
        st.info("Run **Summarize & generate** to build the question bank.")
        return

    version_bar(settings, user, review)
    st.divider()

    cov = coverage_report(quiz.included)
    flagged = sum(1 for q in quiz.included if q.flags)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Included", f"{len(quiz.included)} / {settings.num_questions} requested")
    c2.metric("In the bank", len(quiz.questions))
    c3.metric("Flagged for review", flagged)
    c4.metric(
        "Answer spread",
        " ".join(f"{k}:{v}" for k, v in sorted(cov["answer_position"].items())),
    )

    for note in st.session_state.generation_notes:
        st.caption(f"· {note}")

    report = st.session_state.get("run_report")
    if report is not None and report.failures:
        st.warning(
            f"The last run had problems: {report.headline()} "
            "Some questions may be missing as a result.",
            icon="⚠️",
        )
        if report.advice():
            st.info(report.advice(), icon="💡")
        show_run_report(report, expanded=False)

    if flagged:
        st.warning(
            f"{flagged} included question(s) tripped an item-writing check. "
            "The flags are advisory, not automatic rejections.",
            icon="⚠️",
        )

    with st.expander("Coverage breakdown"):
        b1, b2 = st.columns(2)
        b1.markdown("**Bloom level**")
        b1.bar_chart(cov["bloom"], color=CHART_COLOR)
        b2.markdown("**Difficulty**")
        b2.bar_chart(cov["difficulty"], color=CHART_COLOR)

    st.divider()

    for i, q in enumerate(quiz.questions):
        badge = "⚠️ " if q.flags else ""
        state = "" if q.include else " · not included"
        with st.expander(f"{badge}Q{i + 1}. {q.stem[:90]}{'…' if len(q.stem) > 90 else ''}{state}"):
            top = st.columns(4)
            q.include = top[0].checkbox("Include", value=q.include, key=f"inc_{q.id}")
            top[1].caption(f"**{q.bloom}** · {q.difficulty}")
            top[2].caption(f"⏱ {q.source_timestamp}")
            q.points = top[3].number_input(
                "Points", 0.0, 100.0, float(q.points), 0.5, key=f"pts_{q.id}"
            )

            q.stem = st.text_area("Question", q.stem, key=f"stem_{q.id}", height=80)
            q.options = [
                st.text_input(
                    f"Option {OPTION_LETTERS[j]}"
                    + ("  ✅ correct" if j == q.correct_index else ""),
                    opt, key=f"opt_{q.id}_{j}",
                )
                for j, opt in enumerate(q.options)
            ]
            q.correct_index = st.radio(
                "Correct answer", range(len(q.options)), index=q.correct_index,
                horizontal=True, format_func=lambda j: OPTION_LETTERS[j], key=f"corr_{q.id}",
            )

            if q.rationale:
                st.caption(f"**Why:** {q.rationale}")
            if q.source_quote:
                st.caption(f"**From the lecture:** “{q.source_quote}”")
            for flag in q.flags:
                st.caption(f"⚠️ {flag}")

            st.divider()
            r1, r2 = st.columns([1, 2])
            same_section = r2.toggle(
                "Draw the replacement from the same part of the lecture",
                value=True, key=f"same_{q.id}",
            )
            if r1.button(
                "🔄 Replace this question", key=f"repl_{q.id}",
                use_container_width=True, disabled=not st.session_state.chunks,
            ):
                run_replacement(settings, user, i, same_section=same_section)


def export_tab() -> None:
    quiz: Quiz | None = st.session_state.quiz
    if quiz is None or not quiz.included:
        st.info("Generate and select at least one question to enable exports.")
        return

    version = st.session_state.active_version
    if len(st.session_state.quiz_versions) > 1:
        st.caption(
            f"Exporting **Set {version + 1}** of {len(st.session_state.quiz_versions)}. "
            "Switch sets on the Questions tab."
        )
    quiz.meta.title = st.text_input("Quiz title", quiz.meta.title, key=f"title_{version}")
    quiz.meta.course = st.text_input(
        "Course label (optional)", quiz.meta.course, key=f"course_{version}"
    )
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in quiz.meta.title)[:60] or "quiz"
    summary = st.session_state.summary

    st.markdown("#### Learning management system")
    l1, l2 = st.columns(2)
    l1.download_button(
        "⬇️ QTI 1.2 · Canvas (.zip)", export_qti12_canvas(quiz), f"{stem}_qti12.zip",
        "application/zip", use_container_width=True,
        help="Canvas → Settings → Import Course Content → QTI .zip file.",
    )
    l2.download_button(
        "⬇️ QTI 2.1 · IMS standard (.zip)", export_qti21(quiz), f"{stem}_qti21.zip",
        "application/zip", use_container_width=True,
    )

    st.markdown("#### Spreadsheet")
    s1, s2 = st.columns(2)
    s1.download_button("⬇️ Excel workbook (.xlsx)", export_xlsx(quiz), f"{stem}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
    s2.download_button("⬇️ CSV", export_csv(quiz), f"{stem}.csv", "text/csv",
                       use_container_width=True)

    st.markdown("#### Printable")
    include_key = st.checkbox("Include instructor answer key", value=True)
    include_summary = st.checkbox("Include lecture summary in the document", value=False)
    doc_summary = summary if include_summary else None

    p1, p2, p3 = st.columns(3)
    p1.download_button(
        "⬇️ Word (.docx)", export_docx(quiz, doc_summary, include_key), f"{stem}.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        use_container_width=True,
    )
    p2.download_button("⬇️ PDF", export_pdf(quiz, doc_summary, include_key), f"{stem}.pdf",
                       "application/pdf", use_container_width=True)
    p3.download_button("⬇️ Markdown", export_markdown(quiz, doc_summary), f"{stem}.md",
                       "text/markdown", use_container_width=True)


def library_tab(user: User) -> None:
    """Saved lectures — the transcript, its summary, and its question sets."""
    library = get_library(user)
    if library is None:
        st.error("Your library is unavailable because storage is not configured.")
        return

    store, _, _, _, warning, _ = get_backend()
    if warning:
        st.warning(
            "Nothing here will survive a restart until storage is configured. "
            + warning,
            icon="⚠️",
        )

    try:
        entries = library.entries()
    except StorageError as exc:
        st.error(str(exc))
        return

    st.caption(
        "Lectures are saved automatically as soon as they are transcribed or "
        "imported, and updated when you generate questions. Stored encrypted in "
        f"{store.describe()}."
    )

    if not entries:
        st.info(
            "Nothing saved yet. Transcribe or import a lecture and it will appear here."
        )
        return

    current = st.session_state.get("library_id")
    for entry in entries:
        marker = " · open" if entry.id == current else ""
        with st.expander(f"**{entry.title}**{marker}", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Length", entry.length_label)
            c2.metric("Words", f"{entry.word_count:,}")
            c3.metric("Parts", entry.parts)
            c4.metric("Questions", entry.questions or "—")
            st.caption(
                f"{entry.summary_label} · saved {entry.saved_label} · "
                f"{entry.origin or 'audio'}"
                + (f" · {entry.source_filename}" if entry.source_filename else "")
            )

            a1, a2 = st.columns(2)
            if a1.button(
                "📂 Open this lecture", key=f"open_{entry.id}", use_container_width=True,
                type="primary" if entry.id != current else "secondary",
            ):
                load_saved_lecture(library, entry.id)

            if a2.button("🗑 Delete", key=f"delx_{entry.id}", use_container_width=True):
                st.session_state[f"confirm_del_{entry.id}"] = True

            if st.session_state.get(f"confirm_del_{entry.id}"):
                st.warning(
                    f"Delete **{entry.title}** and everything generated from it? "
                    "This cannot be undone.",
                    icon="⚠️",
                )
                d1, d2 = st.columns(2)
                if d1.button("Delete permanently", key=f"dely_{entry.id}", type="primary"):
                    try:
                        library.delete(entry.id)
                        if st.session_state.get("library_id") == entry.id:
                            st.session_state.library_id = None
                        st.session_state.pop(f"confirm_del_{entry.id}", None)
                        st.toast(f"Deleted {entry.title}.", icon="🗑")
                        st.rerun()
                    except (LibraryError, StorageError) as exc:
                        st.error(str(exc))
                if d2.button("Cancel", key=f"deln_{entry.id}"):
                    st.session_state.pop(f"confirm_del_{entry.id}", None)
                    st.rerun()

            with st.form(f"rename_{entry.id}"):
                new_title = st.text_input("Title", entry.title)
                if st.form_submit_button("Rename") and new_title != entry.title:
                    try:
                        library.rename(entry.id, new_title)
                        st.rerun()
                    except (LibraryError, StorageError) as exc:
                        st.error(str(exc))


def load_saved_lecture(library: TranscriptLibrary, entry_id: str) -> None:
    """Restore a whole working state — not just the text."""
    try:
        saved = library.load(entry_id)
    except (LibraryError, StorageError) as exc:
        st.error(str(exc))
        return

    st.session_state.transcript = saved.transcript
    st.session_state.transcript_note = saved.entry.origin
    st.session_state.source_filename = saved.entry.source_filename or saved.entry.title
    st.session_state.summary = saved.summary
    st.session_state.quiz_versions = list(saved.quizzes)
    st.session_state.quiz = saved.quizzes[-1] if saved.quizzes else None
    st.session_state.active_version = max(0, len(saved.quizzes) - 1)
    st.session_state.generation_notes = []
    st.session_state.run_report = None
    st.session_state.library_id = entry_id

    # Chunks are cheap to rebuild and are what the regenerate and replace
    # buttons need; without them a reopened lecture would be read-only.
    st.session_state.chunks = chunk_transcript(saved.transcript, 600, 30)

    st.toast(f"Opened {saved.entry.title}.", icon="📂")
    st.rerun()


def usage_tab(user: User) -> None:
    """What this account has spent — live for the current run, cumulative below."""
    _, _, usage_log, _, _, _ = get_backend()

    scope_mine = True
    if user.is_admin:
        scope_mine = st.radio(
            "Scope", ["My usage", "Everyone"], horizontal=True, index=0
        ) == "My usage"
    username = user.username if scope_mine else None

    try:
        totals = usage_log.totals(username)
        records = usage_log.records(username)
    except StorageError as exc:
        st.error(str(exc))
        return

    meter: LiveMeter = st.session_state.meter
    st.markdown("#### Right now")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tokens this run", f"{meter.total_tokens:,}")
    c2.metric("Cost this run", meter.cost_label)
    c3.metric("Model calls", meter.calls)

    st.markdown("#### All time")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tokens", f"{totals.total_tokens:,}")
    m2.metric("Estimated cost", totals.cost_label)
    m3.metric("Model calls", f"{totals.calls:,}")
    m4.metric("Runs", f"{totals.runs:,}")

    if totals.unpriced_runs:
        st.caption(
            f"{totals.unpriced_runs} run(s) used a model with no published price; "
            "their tokens are counted but their cost is not."
        )
    if not records:
        st.info("Nothing recorded yet. Generate a quiz and it will appear here.")
        return

    st.caption(
        "Costs are estimates from published rates. Your provider's dashboard is "
        "the authority on what you were actually billed."
    )

    daily = usage_log.by_day(username, days=30)
    if daily:
        st.markdown("**Estimated cost per day (last 30 days)**")
        st.bar_chart(
            pd.DataFrame({"cost": daily.values()}, index=list(daily.keys())),
            y="cost", color=CHART_COLOR, height=220,
        )

    left, right = st.columns(2)
    by_model = usage_log.by_model(username)
    if by_model:
        left.markdown("**Estimated cost by model**")
        left.bar_chart(
            pd.DataFrame(
                {"cost": [t.cost for t in by_model.values()]}, index=list(by_model.keys())
            ),
            y="cost", color=CHART_COLOR, horizontal=True, height=240,
        )

    if user.is_admin and not scope_mine:
        by_user = usage_log.by_user()
        if by_user:
            right.markdown("**Estimated cost by user**")
            right.bar_chart(
                pd.DataFrame(
                    {"cost": [t.cost for t in by_user.values()]}, index=list(by_user.keys())
                ),
                y="cost", color=CHART_COLOR, horizontal=True, height=240,
            )
    else:
        by_operation = usage_log.by_operation(username)
        if by_operation:
            right.markdown("**Estimated cost by step**")
            right.bar_chart(
                pd.DataFrame(
                    {"cost": [t.cost for t in by_operation.values()]},
                    index=list(by_operation.keys()),
                ),
                y="cost", color=CHART_COLOR, horizontal=True, height=240,
            )

    with st.expander("Every recorded run"):
        frame = pd.DataFrame(
            [
                {
                    "When": r.timestamp.replace("T", " ")[:16],
                    "User": r.username,
                    "Model": r.model,
                    "Step": r.operation,
                    "In": r.input_tokens,
                    "Out": r.output_tokens,
                    "Calls": r.calls,
                    "Cost": round(r.cost, 5) if r.cost_known else None,
                    "Source": r.source,
                }
                for r in reversed(records)
            ]
        )
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Usage as CSV", frame.to_csv(index=False).encode("utf-8"),
            "usage.csv", "text/csv",
        )


def admin_tab(directory: UserDirectory, user: User) -> None:
    store, _, _, config, warning, audit = get_backend()

    st.markdown("#### Storage")
    st.caption(f"Backend: **{store.describe()}**")
    if warning:
        st.warning(warning, icon="⚠️")

    management_key_panel(config)

    st.markdown("#### Add a user")
    with st.form("new_user", clear_on_submit=True):
        c1, c2 = st.columns(2)
        username = c1.text_input("Username")
        display_name = c2.text_input("Display name (optional)")
        c3, c4 = st.columns(2)
        role = c3.selectbox("Role", ROLES, index=1)
        password = c4.text_input("Temporary password", value=suggest_password())
        st.caption("They will be asked to choose their own password at first sign-in.")
        if st.form_submit_button("Create account", type="primary"):
            try:
                directory.create_user(
                    username, password, role, display_name, actor=user.username
                )
                st.success(
                    f"Created **{username}**. Give them this temporary password: "
                    f"`{password}`"
                )
            except (AuthError, StorageError) as exc:
                st.error(str(exc))

    st.markdown("#### Accounts")
    try:
        users = directory.all_users()
    except StorageError as exc:
        st.error(str(exc))
        return

    for account in users:
        marker = "" if account.active else " · disabled"
        if account.is_locked:
            marker += " · locked"
        with st.expander(
            f"{account.label} ({account.username}) — {account.role}{marker}"
        ):
            st.caption(
                f"Created {account.created_at[:10]} · "
                f"last sign-in {account.last_login[:16].replace('T', ' ') or 'never'} · "
                f"{len(account.api_keys)} saved API key(s)"
            )
            issued_key_panel(directory, config, account)

            c1, c2, c3 = st.columns(3)

            new_role = c1.selectbox(
                "Role", ROLES, index=ROLES.index(account.role), key=f"role_{account.username}"
            )
            if new_role != account.role and c1.button("Apply role", key=f"ar_{account.username}"):
                _admin_action(directory.set_role, account.username, new_role, user.username)

            label = "Disable" if account.active else "Enable"
            if c2.button(label, key=f"act_{account.username}", use_container_width=True):
                _admin_action(
                    directory.set_active, account.username, not account.active, user.username
                )

            if account.is_locked:
                st.warning(
                    f"Locked after repeated failed sign-ins — "
                    f"{account.lock_minutes_left} minute(s) left.",
                    icon="🔒",
                )
                if st.button("Unlock now", key=f"unl_{account.username}"):
                    _admin_action(directory.unlock, account.username, user.username)

            if c3.button("Reset password", key=f"pw_{account.username}", use_container_width=True):
                st.session_state[f"confirm_reset_{account.username}"] = True

            if st.session_state.get(f"confirm_reset_{account.username}"):
                st.warning(
                    "Resetting this password will **permanently destroy** any "
                    "personal API key this user saved. Their keys are encrypted "
                    "with their password, which is exactly why you cannot read "
                    "them — and why a reset cannot recover them. Keys you issued "
                    "to them are unaffected.",
                    icon="⚠️",
                )
                y1, y2 = st.columns(2)
                if y1.button("Reset anyway", key=f"pwy_{account.username}", type="primary"):
                    fresh = suggest_password()
                    try:
                        directory.set_password(
                            account.username, fresh, clear_flag=False, actor=user.username
                        )
                        st.session_state.pop(f"confirm_reset_{account.username}", None)
                        st.success(
                            f"New temporary password for {account.username}: `{fresh}` — "
                            "their saved personal keys were cleared."
                        )
                    except (AuthError, StorageError) as exc:
                        st.error(str(exc))
                if y2.button("Cancel", key=f"pwn_{account.username}"):
                    st.session_state.pop(f"confirm_reset_{account.username}", None)
                    st.rerun()

            if account.username != user.username:
                if st.button(
                    f"Delete {account.username}", key=f"del_{account.username}"
                ):
                    # Revoke first: a deleted account whose key still works at
                    # OpenRouter is exactly the loose end this feature exists to
                    # prevent.
                    revoke_issued_key(directory, config, account, quiet=True)
                    _admin_action(directory.delete_user, account.username, user.username)

    st.divider()
    audit_panel(audit)


# --------------------------------------------------------------------------- #
# Issued keys
# --------------------------------------------------------------------------- #


def management_key_panel(config: AppConfig) -> None:
    """Configure the one key that mints all the others."""
    st.markdown("#### Issued API keys (OpenRouter)")

    if config.cipher is None:
        # Nothing here can work without a key to encrypt with, so say that
        # plainly rather than letting someone paste a credential into a field
        # that will refuse it.
        st.error(
            "**APP_SECRET is not set**, so this deployment has no encryption key "
            "and nothing can be saved — not the management key, not accounts, not "
            "settings. Everything you have entered so far lives in memory only "
            "and disappears when the app restarts.",
            icon="\U0001F511",
        )
        st.markdown(
            "**To fix it:**\n\n"
            "1. Generate a secret (`python3` on macOS — plain `python` is not "
            "a command there):\n"
            "   ```\n"
            "   python3 -c \"import secrets; print(secrets.token_urlsafe(48))\"\n"
            "   # or, with no Python: openssl rand -base64 48\n"
            "   ```\n"
            "2. Add it where this app reads configuration:\n"
            "   - **Streamlit Community Cloud** — Manage app -> Settings -> Secrets:\n"
            "     ```toml\n"
            '     APP_SECRET = "paste-it-here"\n'
            "     ```\n"
            "   - **Running locally** — put the same line in "
            "`.streamlit/secrets.toml`, or `APP_SECRET=...` in `.env`.\n"
            "3. Restart the app and create the administrator account again.\n\n"
            "Keep that value backed up: it is the key to everything stored. "
            "See **SECURITY.md** for rotation."
        )
        return

    try:
        configured = config.has_secret(MANAGEMENT_KEY_NAME)
    except StorageError as exc:
        st.error(str(exc))
        return

    if not configured:
        st.caption(
            "Issue each person a capped, revocable OpenRouter key instead of asking "
            "them for a personal one. A leaked key then costs at most its cap, and "
            "revoking it is one click. Needs a **management key** from "
            "[openrouter.ai/settings/management-keys]"
            "(https://openrouter.ai/settings/management-keys) — an ordinary "
            "inference key will not work."
        )
        typed = st.text_input(
            "OpenRouter management key", type="password", key="mgmt_key_input"
        )
        if typed and st.button("Verify and save", type="primary"):
            try:
                ProvisioningClient(typed).verify()
                config.set_secret(MANAGEMENT_KEY_NAME, typed)
                _, _, _, _, _, audit_log = get_backend()
                audit_log.record(
                    "management_key.set", st.session_state.user.username
                )
                st.success("Management key verified and saved.")
                st.rerun()
            except (ProvisioningError, StorageError, RuntimeError) as exc:
                st.error(str(exc))
        return

    c1, c2 = st.columns([3, 1])
    c1.success("Management key configured — you can issue keys below.", icon="🔑")
    if c2.button("Remove", key="rm_mgmt", use_container_width=True):
        try:
            config.set_secret(MANAGEMENT_KEY_NAME, "")
            _, _, _, _, _, audit_log = get_backend()
            audit_log.record("management_key.cleared", st.session_state.user.username)
            st.toast("Management key removed. Issued keys still work until revoked.")
            st.rerun()
        except (StorageError, RuntimeError) as exc:
            st.error(str(exc))

    d1, d2 = st.columns(2)
    current_limit = float(config.get("default_key_limit", DEFAULT_LIMIT))
    presets = list(LIMIT_PRESETS)
    if current_limit not in presets:
        presets = sorted({*presets, current_limit})
    new_limit = d1.selectbox(
        "Default cap per user (USD)",
        presets,
        index=presets.index(current_limit),
        format_func=lambda v: f"${v:,.2f}",
    )
    current_reset = config.get("default_key_reset", "monthly")
    new_reset = d2.selectbox(
        "Cap resets", RESET_PERIODS, index=RESET_PERIODS.index(current_reset)
    )
    if (new_limit, new_reset) != (current_limit, current_reset):
        try:
            config.set("default_key_limit", float(new_limit))
            config.set("default_key_reset", new_reset)
        except StorageError as exc:
            st.error(str(exc))


def issued_key_panel(directory: UserDirectory, config: AppConfig, account: User) -> None:
    """Issue, monitor, adjust or revoke one account's key."""
    client = provisioning_client(config)
    record = account.provisioned_keys.get(PROVISIONED_PROVIDER, {})

    if client is None:
        if record:
            st.caption(
                "This account has an issued key, but no management key is "
                "configured — restore it above to inspect or revoke."
            )
        return

    if not record.get("hash"):
        c1, c2 = st.columns([1, 1])
        default_limit = float(config.get("default_key_limit", DEFAULT_LIMIT))
        presets = list(LIMIT_PRESETS)
        if default_limit not in presets:
            presets = sorted({*presets, default_limit})
        limit = c1.selectbox(
            "Cap (USD)",
            presets,
            index=presets.index(default_limit),
            format_func=lambda v: f"${v:,.2f}",
            key=f"lim_{account.username}",
        )
        if c2.button(
            "🔑 Issue an OpenRouter key",
            key=f"mint_{account.username}",
            use_container_width=True,
        ):
            issue_key(directory, config, account, float(limit), client)
        return

    info = key_usage(config.get_secret(MANAGEMENT_KEY_NAME), record["hash"])
    if info is None:
        st.warning(
            "Could not read this key's usage from OpenRouter just now.", icon="📶"
        )
    else:
        spend_caption(info)
        st.caption(
            f"`{info.label or record.get('name', '')}` · "
            f"resets {info.limit_reset or record.get('limit_reset', 'monthly')}"
            + (" · **disabled**" if info.disabled else "")
        )

    c1, c2, c3 = st.columns(3)
    presets = list(LIMIT_PRESETS)
    existing = float(record.get("limit") or DEFAULT_LIMIT)
    if existing not in presets:
        presets = sorted({*presets, existing})
    new_cap = c1.selectbox(
        "Cap (USD)",
        presets,
        index=presets.index(existing),
        format_func=lambda v: f"${v:,.2f}",
        key=f"cap_{account.username}",
    )
    if new_cap != existing and c1.button("Apply cap", key=f"apply_{account.username}"):
        try:
            client.update_key(record["hash"], limit=float(new_cap))
            record["limit"] = float(new_cap)
            directory.record_provisioned_key(
                account.username,
                PROVISIONED_PROVIDER,
                directory.get_issued_key(account, PROVISIONED_PROVIDER),
                record,
                actor=st.session_state.user.username,
            )
            key_usage.clear()
            st.rerun()
        except (ProvisioningError, AuthError, StorageError) as exc:
            st.error(str(exc))

    if info is not None:
        toggle_label = "Enable key" if info.disabled else "Pause key"
        if c2.button(toggle_label, key=f"tog_{account.username}", use_container_width=True):
            try:
                client.update_key(record["hash"], disabled=not info.disabled)
                key_usage.clear()
                st.rerun()
            except ProvisioningError as exc:
                st.error(str(exc))

    if c3.button("Revoke key", key=f"rev_{account.username}", use_container_width=True):
        revoke_issued_key(directory, config, account)
        st.rerun()


def issue_key(
    directory: UserDirectory,
    config: AppConfig,
    account: User,
    limit: float,
    client: ProvisioningClient,
) -> None:
    reset = config.get("default_key_reset", "monthly")
    try:
        minted = client.create_key(key_name_for(account.username), limit=limit, limit_reset=reset)
    except ProvisioningError as exc:
        st.error(str(exc))
        return

    try:
        directory.record_provisioned_key(
            account.username,
            PROVISIONED_PROVIDER,
            minted.secret,
            {
                "hash": minted.info.hash,
                "name": minted.info.name or key_name_for(account.username),
                "limit": limit,
                "limit_reset": reset,
                "created_at": minted.info.created_at,
            },
            actor=st.session_state.user.username,
        )
    except (AuthError, StorageError) as exc:
        # The key exists upstream but we could not store it — say so, with the
        # hash, so it can be cleaned up rather than left billing quietly.
        st.error(
            f"The key was created at OpenRouter but could not be saved here ({exc}). "
            f"Delete it manually — its hash is `{minted.info.hash}`."
        )
        return

    st.success(
        f"Issued a key to **{account.username}**, capped at ${limit:,.2f} per {reset}. "
        "They do not need to enter anything — it is already on their account."
    )
    key_usage.clear()


def revoke_issued_key(
    directory: UserDirectory, config: AppConfig, account: User, quiet: bool = False
) -> None:
    record = account.provisioned_keys.get(PROVISIONED_PROVIDER, {})
    if not record.get("hash"):
        return
    client = provisioning_client(config)

    revoked = False
    if client is not None:
        try:
            client.delete_key(record["hash"])
            revoked = True
        except ProvisioningError as exc:
            if not quiet:
                st.error(f"OpenRouter would not revoke the key: {exc}")

    try:
        directory.clear_provisioned_key(
            account.username, PROVISIONED_PROVIDER, actor=st.session_state.user.username
        )
    except (AuthError, StorageError) as exc:
        if not quiet:
            st.error(str(exc))
        return

    key_usage.clear()
    if not quiet:
        if revoked:
            st.toast(f"Revoked {account.username}'s key.", icon="🚫")
        else:
            st.warning(
                "Removed the key from this app, but it may still be live at "
                f"OpenRouter — delete hash `{record['hash']}` there.",
                icon="⚠️",
            )


def audit_panel(audit: AuditLog) -> None:
    """The record of who did what to whom. Never what the credential was."""
    st.markdown("#### Security log")
    st.caption(
        "Sign-ins, lockouts, account changes and key events. Credentials "
        "themselves are never written here."
    )
    try:
        events = audit.events(limit=250)
    except StorageError as exc:
        st.error(str(exc))
        return
    if not events:
        st.caption("Nothing recorded yet.")
        return

    frame = pd.DataFrame(
        [
            {
                "When": e.timestamp.replace("T", " ")[:16],
                "Event": e.event,
                "By": e.actor,
                "Subject": e.subject,
                "Detail": e.detail,
            }
            for e in events
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True, height=300)
    st.download_button(
        "⬇️ Security log as CSV",
        frame.to_csv(index=False).encode("utf-8"),
        "security_log.csv",
        "text/csv",
    )


def _admin_action(function, *args) -> None:
    try:
        function(*args)
        st.rerun()
    except (AuthError, StorageError) as exc:
        st.error(str(exc))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    init_state()
    try:
        _, directory, _, config, backend_warning, _ = get_backend()
    except StorageError as exc:
        st.error(f"Storage could not start: {exc}")
        return

    if not gate(directory):
        return

    user = current_user()
    settings = sidebar(directory, user)

    st.title("🎓 Lecture Quiz Builder")
    st.caption(
        "Transcribe a lecture, summarize it, and generate a reviewable "
        "multiple-choice bank you can import straight into your LMS."
    )
    if backend_warning:
        st.warning(backend_warning, icon="⚠️")
        if "APP_SECRET" in backend_warning:
            with st.expander("How to set APP_SECRET", expanded=False):
                st.markdown(
                    "Generate one (`python3` on macOS):\n"
                    "```\n"
                    "python3 -c \"import secrets; print(secrets.token_urlsafe(48))\"\n"
                    "# or: openssl rand -base64 48\n"
                    "```\n"
                    "Then add `APP_SECRET = \"...\"` to your Streamlit secrets "
                    "(**Manage app → Settings → Secrets** on Community Cloud) or to "
                    "`.streamlit/secrets.toml` locally, and restart. "
                    "Until then nothing is saved between restarts."
                )

    review = st.sidebar.checkbox(
        "Run a second-pass quality review",
        value=True,
        help="A reviewer prompt re-reads the drafted items and repairs or drops weak "
        "ones; anything it drops is replaced, so you still get the count you asked for.",
    )

    diagnostics_panel(settings, user)
    input_section(settings, user, review)
    st.divider()

    names = [
        "📝 Transcript", "📋 Summary", "❓ Questions",
        "⬇️ Export", "📚 Library", "📊 Usage",
    ]
    if user.is_admin:
        names.append("🛠️ Admin")
    tabs = st.tabs(names)

    with tabs[0]:
        transcript_tab()
    with tabs[1]:
        summary_tab()
    with tabs[2]:
        questions_tab(settings, user, review)
    with tabs[3]:
        export_tab()
    with tabs[4]:
        library_tab(user)
    with tabs[5]:
        usage_tab(user)
    if user.is_admin:
        with tabs[6]:
            admin_tab(directory, user)


if __name__ == "__main__":
    main()
