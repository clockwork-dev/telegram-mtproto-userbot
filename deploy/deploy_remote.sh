#!/usr/bin/env bash
set -euo pipefail

# Multi-Profile Remote Deployment & Verification Script for telegram-group-bot
# Usage:
#   ./deploy/deploy_remote.sh [profile] [user@host] [remote_dir]
# Supported profiles:
#   bot_api (default) -> deploys to /opt/telegram-group-bot, service telegram-userbot.service
#   userbot           -> deploys to /opt/telegram-mtproto-userbot, service telegram-mtproto-userbot.service

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Parse arguments flexibly
PROFILE="bot_api"
REMOTE_HOST="root@192.168.1.164"
CUSTOM_REMOTE_DIR=""

if [ $# -ge 1 ]; then
  case "$1" in
    bot_api|userbot)
      PROFILE="$1"
      shift
      ;;
    *)
      if [[ "$1" == *"@"* ]] || [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        REMOTE_HOST="$1"
        shift
        if [ $# -ge 1 ]; then
          if [ "$1" = "bot_api" ] || [ "$1" = "userbot" ]; then
            PROFILE="$1"
            shift
          elif [[ "$1" != "/"* ]] && [[ "$1" != "."* ]]; then
            echo "[ERROR] Unknown deployment profile '$1'. Use 'bot_api' or 'userbot'." >&2
            exit 1
          fi
        fi
      else
        echo "[ERROR] Unknown deployment profile '$1'. Use 'bot_api' or 'userbot'." >&2
        exit 1
      fi
      ;;
  esac
fi

if [ $# -ge 1 ]; then
  if [[ "$1" == *"@"* ]] || [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    REMOTE_HOST="$1"
    shift
  elif [[ "$1" == "/"* ]] || [[ "$1" == "."* ]]; then
    CUSTOM_REMOTE_DIR="$1"
    shift
  else
    REMOTE_HOST="$1"
    shift
  fi
fi

if [ $# -ge 1 ] && [ -z "${CUSTOM_REMOTE_DIR}" ]; then
  CUSTOM_REMOTE_DIR="$1"
  shift
fi

# Profile-specific configurations and strict isolation guards
case "${PROFILE}" in
  bot_api)
    REMOTE_DIR="${CUSTOM_REMOTE_DIR:-/opt/telegram-group-bot}"
    SERVICE_NAME="telegram-userbot.service"
    SERVICE_SRC="systemd/telegram-userbot.service"
    CONFIG_FILE="config/bot.yaml"
    ;;
  userbot)
    REMOTE_DIR="${CUSTOM_REMOTE_DIR:-/opt/telegram-mtproto-userbot}"
    SERVICE_NAME="telegram-mtproto-userbot.service"
    SERVICE_SRC="systemd/telegram-mtproto-userbot.service"
    CONFIG_FILE="config/userbot.yaml"
    ;;
  *)
    echo "[ERROR] Unknown deployment profile '${PROFILE}'. Use 'bot_api' or 'userbot'." >&2
    exit 1
    ;;
esac

# Normalize REMOTE_DIR: canonicalize path, collapse repeated slashes, and strip trailing slashes
if command -v realpath >/dev/null 2>&1; then
  REMOTE_DIR="$(realpath -m "${REMOTE_DIR}")"
else
  REMOTE_DIR="$(echo "${REMOTE_DIR}" | sed -E 's#/+#/#g')"
  if [ "${REMOTE_DIR}" != "/" ]; then
    REMOTE_DIR="${REMOTE_DIR%/}"
  fi
fi

# Safety check: prevent cross-profile contamination
if [ "${PROFILE}" = "userbot" ] && [ "${REMOTE_DIR}" = "/opt/telegram-group-bot" ]; then
  echo "[FATAL] Profile 'userbot' must NEVER target '/opt/telegram-group-bot'!" >&2
  exit 2
fi

if [ "${PROFILE}" = "bot_api" ] && [ "${REMOTE_DIR}" = "/opt/telegram-mtproto-userbot" ]; then
  echo "[FATAL] Profile 'bot_api' must NEVER target '/opt/telegram-mtproto-userbot'!" >&2
  exit 2
fi

# SSH & Rsync credentials
SSH_PASS="${SSH_PASSWORD:-${SSHPASS:-13251325}}"
if command -v sshpass >/dev/null 2>&1 && [ -n "${SSH_PASS}" ]; then
  SSH_CMD=(sshpass -p "${SSH_PASS}" ssh -o StrictHostKeyChecking=no)
  RSYNC_RSH="sshpass -p '${SSH_PASS}' ssh -o StrictHostKeyChecking=no"
else
  SSH_CMD=(ssh -o StrictHostKeyChecking=no)
  RSYNC_RSH="ssh -o StrictHostKeyChecking=no"
fi

echo "=========================================================="
echo " Starting Deployment:"
echo "   Profile:       ${PROFILE}"
echo "   Remote Target: ${REMOTE_HOST}:${REMOTE_DIR}"
echo "   Service Unit:  ${SERVICE_NAME}"
echo "   Config:        ${CONFIG_FILE}"
echo "=========================================================="

# Pre-deployment environment preparation
if [ "${PROFILE}" = "userbot" ]; then
  echo "==> Preparing isolated directories and shared assets on ${REMOTE_HOST} ..."
  "${SSH_CMD[@]}" "${REMOTE_HOST}" "
    set -euo pipefail
    mkdir -p '${REMOTE_DIR}'
    chmod 700 '${REMOTE_DIR}'
    mkdir -p '${REMOTE_DIR}/data/models' '${REMOTE_DIR}/data/session' '${REMOTE_DIR}/data/tmp' '${REMOTE_DIR}/data/antigravity'
    chmod 700 '${REMOTE_DIR}/data' '${REMOTE_DIR}/data/session' '${REMOTE_DIR}/data/tmp' '${REMOTE_DIR}/data/antigravity'

    # Symlink HuggingFace Whisper model cache (0 MB disk duplication)
    if [ -d '/opt/telegram-group-bot/data/models/whisper' ] && [ ! -e '${REMOTE_DIR}/data/models/whisper' ]; then
      echo '==> Creating Whisper model cache symlink -> /opt/telegram-group-bot/data/models/whisper ...'
      ln -sfn /opt/telegram-group-bot/data/models/whisper '${REMOTE_DIR}/data/models/whisper'
    fi

    # Seed isolated Antigravity OAuth tokens
    if [ -d '/opt/telegram-group-bot/data/antigravity/.gemini' ] && [ ! -d '${REMOTE_DIR}/data/antigravity/.gemini' ]; then
      echo '==> Seeding isolated Antigravity OAuth tokens ...'
      cp -a /opt/telegram-group-bot/data/antigravity/.gemini '${REMOTE_DIR}/data/antigravity/'
      chmod 700 '${REMOTE_DIR}/data/antigravity'
      find '${REMOTE_DIR}/data/antigravity' -type d -exec chmod 700 {} +
      find '${REMOTE_DIR}/data/antigravity' -type f -exec chmod 600 {} +
    fi

    # Create .env with empty TELEGRAM_API_ID / TELEGRAM_API_HASH if not exists
    if [ ! -f '${REMOTE_DIR}/.env' ]; then
      echo '==> Initializing default template .env (mode 0600) ...'
      cat << 'ENVEOF' > '${REMOTE_DIR}/.env'
# Telegram MTProto Userbot Credentials (from https://my.telegram.org)
TELEGRAM_API_ID=
TELEGRAM_API_HASH=
TELEGRAM_PHONE=
TELEGRAM_SESSION_PATH=data/session/userbot.session

# Antigravity OAuth & Binary Isolation
ANTIGRAVITY_DATA_DIR=data/antigravity
ANTIGRAVITY_BINARY=/usr/local/bin/agy
ENVEOF
      chmod 600 '${REMOTE_DIR}/.env'
    fi
  "
else
  echo "==> Preparing target directory on ${REMOTE_HOST} ..."
  "${SSH_CMD[@]}" "${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}' && chmod 700 '${REMOTE_DIR}'"
fi

# Code sync via rsync
echo "==> Syncing codebase to ${REMOTE_HOST}:${REMOTE_DIR} ..."
rsync -az --delete \
  -e "${RSYNC_RSH}" \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude 'data/' \
  --exclude '.env' \
  --exclude '*.session*' \
  "${REPO_ROOT}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

# Server environment verification & systemd service setup
echo "==> Setting up environment and executing verification on ${REMOTE_HOST} ..."
"${SSH_CMD[@]}" "${REMOTE_HOST}" "
  set -euo pipefail
  cd '${REMOTE_DIR}'

  if [ ! -d .venv ]; then
    echo '==> Creating Python virtualenv in ${REMOTE_DIR}/.venv ...'
    python3 -m venv .venv || (apt-get update -qq && apt-get install -y -qq python3-venv && python3 -m venv .venv)
    if [ -d '/opt/telegram-group-bot/.venv/lib/python3.13/site-packages' ]; then
      echo '==> Reusing site-packages from existing service ...'
      cp -a /opt/telegram-group-bot/.venv/lib/python3.13/site-packages/* .venv/lib/python3.13/site-packages/ 2>/dev/null || true
    fi
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -e '.[dev]'
    .venv/bin/pip install --force-reinstall --no-deps ruff pytest
  else
    echo '==> Updating package install in existing .venv ...'
    .venv/bin/pip install -e . --no-deps
    if [ ! -f .venv/bin/ruff ] || [ ! -f .venv/bin/pytest ]; then
      .venv/bin/pip install --force-reinstall --no-deps ruff pytest
    fi
  fi

  echo '==> Running ruff verification ...'
  .venv/bin/ruff check .

  echo '==> Running pytest regression suite ...'
  .venv/bin/pytest -v tests/test_telegram_auth.py tests/test_telegram_heuristics.py tests/test_telegram_client.py tests/test_telegram_voice.py tests/e2e/

  echo '==> Running dry-run smoke test for ${CONFIG_FILE} ...'
  .venv/bin/telegram-group-bot run --config '${CONFIG_FILE}' --dry-run

  echo '==> Installing service unit /etc/systemd/system/${SERVICE_NAME} ...'
  cp '${SERVICE_SRC}' '/etc/systemd/system/${SERVICE_NAME}'
  systemctl daemon-reload
  systemd-analyze verify '/etc/systemd/system/${SERVICE_NAME}'

  if [ '${PROFILE}' = 'userbot' ]; then
    echo '==> Checking ConditionPathExists guard for ${SERVICE_NAME} ...'
    # Unit should be inactive/dead without crash-looping because userbot.session does not exist yet
    systemctl status '${SERVICE_NAME}' --no-pager || true
  fi

  echo '==> Verifying original telegram-userbot.service is active and healthy ...'
  systemctl is-active telegram-userbot.service

  echo '==> Memory status on remote server:'
  free -m
"

echo "=========================================================="
echo " Deployment of profile '${PROFILE}' succeeded on ${REMOTE_HOST}!"
echo "=========================================================="
