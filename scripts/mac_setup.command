#!/bin/bash
#
# Double-click this file in Finder to set up and start Lecture Quiz Builder on a
# Mac. It is a .command file so that double-clicking opens it in Terminal — no
# typing required, which is the whole point.
#
# What it does, all of it inside this folder and none of it elsewhere:
#   1. checks Python is present
#   2. creates a private virtual environment in .venv/
#   3. installs the dependencies
#   4. generates APP_SECRET into .env, once, if there isn't one
#   5. checks whether Ollama is running and says so
#   6. starts the app in your browser
#
# Safe to run again. After the first time it skips straight to starting up.

set -u

cd "$(dirname "$0")/.." || exit 1
HERE="$(pwd)"

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
warn() { printf "\033[33m%s\033[0m\n" "$1"; }
fail() { printf "\033[31m%s\033[0m\n" "$1"; }
ok()   { printf "\033[32m%s\033[0m\n" "$1"; }

echo
bold "Lecture Quiz Builder — setup"
echo "Folder: $HERE"
echo

# --------------------------------------------------------------------------- #
# 1. Python
# --------------------------------------------------------------------------- #

if ! command -v python3 >/dev/null 2>&1; then
  fail "Python 3 is not installed."
  echo
  echo "macOS asks to install the developer tools the first time you need them."
  echo "Run this, accept the prompt, wait for it to finish, then double-click"
  echo "this file again:"
  echo
  echo "    xcode-select --install"
  echo
  echo "If that does not work, install Python from https://www.python.org/downloads/"
  echo
  read -r -p "Press Return to close. "
  exit 1
fi

PY_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
ok "Python $PY_VERSION found."

# --------------------------------------------------------------------------- #
# 2 & 3. Virtual environment and dependencies
# --------------------------------------------------------------------------- #
#
# A virtual environment keeps these packages inside this folder rather than
# changing the Python your Mac uses for anything else. Deleting the folder
# removes every trace.

if [ ! -d ".venv" ]; then
  echo "Creating a private Python environment (.venv)…"
  if ! python3 -m venv .venv; then
    fail "Could not create the environment."
    read -r -p "Press Return to close. "
    exit 1
  fi
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Installing takes a few minutes the first time — faster-whisper and its model
# runtime are large. Afterwards pip finds everything already present.
NEEDS_INSTALL=1
if python -c "import streamlit, faster_whisper" >/dev/null 2>&1; then
  NEEDS_INSTALL=0
fi

if [ "$NEEDS_INSTALL" -eq 1 ]; then
  echo "Installing dependencies. The first run takes a few minutes…"
  python -m pip install --quiet --upgrade pip
  if ! python -m pip install --quiet -r requirements.txt; then
    fail "Installation failed. The output above says why."
    read -r -p "Press Return to close. "
    exit 1
  fi
  ok "Dependencies installed."
else
  ok "Dependencies already installed."
fi

# --------------------------------------------------------------------------- #
# 4. APP_SECRET
# --------------------------------------------------------------------------- #
#
# Without this the app runs but saves nothing. Generated once and never
# regenerated: changing it makes everything already saved unreadable, so an
# existing .env is left strictly alone.

if [ ! -f ".env" ]; then
  echo "Creating .env with a new APP_SECRET…"
  SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
  {
    echo "# Created by scripts/mac_setup.command"
    echo "# APP_SECRET encrypts everything this app saves."
    echo "# Back it up where you keep passwords. If you lose it, your saved"
    echo "# lectures cannot be recovered — by anyone, ever."
    echo "APP_SECRET=$SECRET"
  } > .env
  ok "Created .env"
  echo
  warn "Back up the APP_SECRET line in .env where you keep passwords."
  echo
elif grep -q "^APP_SECRET=.\+" .env 2>/dev/null; then
  ok "APP_SECRET already set."
else
  warn "There is a .env file but no APP_SECRET in it."
  echo "  The app will run, but nothing will be saved between sessions."
  echo "  Add a line like APP_SECRET=... to $HERE/.env"
  echo
fi

# --------------------------------------------------------------------------- #
# 5. Is there a model to talk to?
# --------------------------------------------------------------------------- #
#
# Only a courtesy check. The app works fine with a hosted model, so a missing
# Ollama is a note, never a reason to stop.

if curl -s --max-time 2 http://localhost:11434/v1/models >/dev/null 2>&1; then
  MODEL_COUNT="$(curl -s --max-time 2 http://localhost:11434/v1/models \
    | python -c 'import json,sys; print(len(json.load(sys.stdin).get("data", [])))' 2>/dev/null || echo 0)"
  if [ "$MODEL_COUNT" -gt 0 ]; then
    ok "Ollama is running with $MODEL_COUNT model(s) ready."
  else
    warn "Ollama is running but has no models downloaded."
    echo "  In another Terminal window:  ollama pull qwen2.5:14b"
  fi
elif curl -s --max-time 2 http://localhost:1234/v1/models >/dev/null 2>&1; then
  ok "LM Studio is running."
else
  warn "No local model server detected — that is fine."
  echo "  The app starts anyway and works with a hosted model."
  echo "  For a local one: install Ollama from https://ollama.com/download,"
  echo "  open it, then run:  ollama pull qwen2.5:14b"
fi

# --------------------------------------------------------------------------- #
# 6. Go
# --------------------------------------------------------------------------- #

echo
bold "Starting the app…"
echo "Your browser opens at http://localhost:8501"
echo
echo "Leave this Terminal window open while you use the app."
echo "To stop it, press Control-C here, or just close the window."
echo

streamlit run app.py

echo
echo "The app has stopped."
read -r -p "Press Return to close. "
