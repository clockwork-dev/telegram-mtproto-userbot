#!/usr/bin/env bash
set -euo pipefail

# Interactive Secret & Session Re-authentication Wizard for MTProto Userbot
# Usage:
#   ./deploy/reauth_userbot.sh
#   ./deploy/reauth_userbot.sh --help

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
CONFIG_FILE="${REPO_ROOT}/config/userbot.yaml"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
BOT_BIN="${REPO_ROOT}/.venv/bin/telegram-group-bot"
SERVICE_NAME="telegram-mtproto-userbot.service"
DATA_AGY_DIR="${REPO_ROOT}/data/antigravity"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat << 'EOF'
Использование: ./deploy/reauth_userbot.sh

Интерактивный мастер смены секретов и аккаунтов для telegram-mtproto-userbot:
  1. Аккуратно останавливает systemd-сервис telegram-mtproto-userbot.service (если запущен).
  2. Позволяет обновить TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_PHONE и GEMINI_API_KEY
     в .env (права 0600) и синхронизирует api_id / api_hash в config/userbot.yaml.
  3. Сохраняет резервную копию старой Telegram-сессии (.session) и запускает интерактивный
     вход по коду/2FA для нового аккаунта (с возможностью отката при ошибке).
  4. Опционально переавторизует изолированный аккаунт Antigravity (auth-agy).
  5. Опционально архивирует старую базу памяти (data/memory.sqlite) для чистого старта.
  6. Запускает сервис обратно и проверяет его статус.
EOF
  exit 0
fi

if [[ ! -x "${PYTHON_BIN}" || ! -x "${BOT_BIN}" ]]; then
  echo "[ERROR] Виртуальное окружение не найдено в ${REPO_ROOT}/.venv" >&2
  exit 1
fi

umask 077

echo "=================================================================="
echo "  Мастер смены аккаунта и секретов (telegram-mtproto-userbot)"
echo "  Папка проекта: ${REPO_ROOT}"
echo "=================================================================="

# 1. Stop running systemd service to unlock SQLite session & DB files
SERVICE_WAS_ACTIVE=0
if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet "${SERVICE_NAME}"; then
  echo "==> Останавливаю активный сервис ${SERVICE_NAME}..."
  systemctl stop "${SERVICE_NAME}"
  SERVICE_WAS_ACTIVE=1
  echo "    Сервис остановлен."
fi

# Ensure .env exists with strict 0600 permissions
if [[ ! -f "${ENV_FILE}" ]]; then
  if [[ -f "${REPO_ROOT}/.env.userbot.example" ]]; then
    cp "${REPO_ROOT}/.env.userbot.example" "${ENV_FILE}"
  else
    cat << 'EOF' > "${ENV_FILE}"
# Telegram MTProto Userbot Credentials (from https://my.telegram.org)
TELEGRAM_API_ID=
TELEGRAM_API_HASH=
TELEGRAM_PHONE=
TELEGRAM_SESSION_PATH=data/session/userbot.session

# Antigravity OAuth & Binary Isolation
ANTIGRAVITY_DATA_DIR=data/antigravity
ANTIGRAVITY_BINARY=/usr/local/bin/agy
GEMINI_API_KEY=
EOF
  fi
fi
chmod 600 "${ENV_FILE}"

# Helper: read a key from .env safely via Python
get_env_val() {
  local key="$1"
  "${PYTHON_BIN}" - "${ENV_FILE}" "${key}" << 'PYEOF'
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
target_key = sys.argv[2]
if env_path.is_file():
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == target_key:
            print(v.strip().strip("'\""))
            sys.exit(0)
PYEOF
}

mask_secret() {
  local s="$1"
  local len=${#s}
  if [[ ${len} -eq 0 ]]; then
    echo "не задано"
  elif [[ ${len} -le 6 ]]; then
    echo "****"
  else
    echo "${s:0:3}****${s: -3}"
  fi
}

CUR_API_ID="$(get_env_val "TELEGRAM_API_ID")"
CUR_API_HASH="$(get_env_val "TELEGRAM_API_HASH")"
CUR_PHONE="$(get_env_val "TELEGRAM_PHONE")"
CUR_SESSION_REL="$(get_env_val "TELEGRAM_SESSION_PATH")"
CUR_GEMINI_KEY="$(get_env_val "GEMINI_API_KEY")"

if [[ -z "${CUR_SESSION_REL}" ]]; then
  CUR_SESSION_REL="data/session/userbot.session"
fi
if [[ "${CUR_SESSION_REL}" = /* ]]; then
  SESSION_PATH="${CUR_SESSION_REL}"
else
  SESSION_PATH="${REPO_ROOT}/${CUR_SESSION_REL}"
fi

echo ""
echo "--- Шаг 1. Секреты Telegram (my.telegram.org) и окружения ---"
echo "(Нажми Enter, чтобы оставить текущее значение без изменений)"
echo ""

# Prompt TELEGRAM_API_ID
while true; do
  read -r -p "TELEGRAM_API_ID [текущий: $(mask_secret "${CUR_API_ID}")]: " INPUT_API_ID
  INPUT_API_ID="${INPUT_API_ID//[[:space:]]/}"
  NEW_API_ID="${INPUT_API_ID:-${CUR_API_ID}}"
  if [[ -z "${NEW_API_ID}" || ! "${NEW_API_ID}" =~ ^[0-9]+$ || "${NEW_API_ID}" == "12345" || "${NEW_API_ID}" == "12345678" ]]; then
    echo "  [!] Укажи реальный числовой API ID с https://my.telegram.org"
    continue
  fi
  break
done

# Prompt TELEGRAM_API_HASH
while true; do
  read -r -p "TELEGRAM_API_HASH [текущий: $(mask_secret "${CUR_API_HASH}")]: " INPUT_API_HASH
  INPUT_API_HASH="${INPUT_API_HASH//[[:space:]]/}"
  NEW_API_HASH="${INPUT_API_HASH:-${CUR_API_HASH}}"
  if [[ -z "${NEW_API_HASH}" || "${NEW_API_HASH,,}" == *"0123456789abcdef"* ]]; then
    echo "  [!] Укажи реальный API Hash с https://my.telegram.org"
    continue
  fi
  break
done

# Prompt TELEGRAM_PHONE (optional if using QR login)
read -r -p "TELEGRAM_PHONE в формате +... (необязательно для QR, Enter — оставить $(mask_secret "${CUR_PHONE}")): " INPUT_PHONE
INPUT_PHONE="${INPUT_PHONE//[[:space:]]/}"
NEW_PHONE="${INPUT_PHONE:-${CUR_PHONE}}"
if [[ -n "${NEW_PHONE}" && "${NEW_PHONE}" != +* ]]; then
  NEW_PHONE="+${NEW_PHONE}"
fi

# Optional GEMINI_API_KEY update
NEW_GEMINI_KEY="${CUR_GEMINI_KEY}"
read -r -p "Поменять GEMINI_API_KEY (TTS / эмбеддинги)? [текущий: $(mask_secret "${CUR_GEMINI_KEY}")] [y/N]: " CHANGE_GEMINI
if [[ "${CHANGE_GEMINI,,}" == "y" || "${CHANGE_GEMINI,,}" == "yes" || "${CHANGE_GEMINI,,}" == "д" ]]; then
  read -r -s -p "Введи новый GEMINI_API_KEY (ввод скрыт): " INPUT_GEMINI
  echo ""
  INPUT_GEMINI="${INPUT_GEMINI//[[:space:]]/}"
  if [[ -n "${INPUT_GEMINI}" ]]; then
    NEW_GEMINI_KEY="${INPUT_GEMINI}"
  else
    echo "  [i] Пустой ввод — оставлен прежний GEMINI_API_KEY."
  fi
fi

# Atomically update .env and sync api_id / api_hash in config/userbot.yaml
NEW_API_ID="${NEW_API_ID}" \
NEW_API_HASH="${NEW_API_HASH}" \
NEW_PHONE="${NEW_PHONE}" \
NEW_GEMINI_KEY="${NEW_GEMINI_KEY}" \
"${PYTHON_BIN}" - "${ENV_FILE}" "${CONFIG_FILE}" << 'PYEOF'
import os
import re
import sys
import tempfile
from pathlib import Path

env_path = Path(sys.argv[1])
cfg_path = Path(sys.argv[2])

updates = {
    "TELEGRAM_API_ID": os.environ["NEW_API_ID"],
    "TELEGRAM_API_HASH": os.environ["NEW_API_HASH"],
    "TELEGRAM_PHONE": os.environ["NEW_PHONE"],
}
gemini_val = os.environ.get("NEW_GEMINI_KEY", "")
if gemini_val:
    updates["GEMINI_API_KEY"] = gemini_val

lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.is_file() else []
seen = set()
out_lines = []
for raw in lines:
    stripped = raw.strip()
    if stripped and not stripped.startswith("#") and "=" in stripped:
        k, _ = stripped.split("=", 1)
        k_clean = k.strip()
        if k_clean in updates:
            out_lines.append(f"{k_clean}={updates[k_clean]}")
            seen.add(k_clean)
            continue
    out_lines.append(raw)

for k, v in updates.items():
    if k not in seen:
        out_lines.append(f"{k}={v}")

fd, tmp_env = tempfile.mkstemp(dir=str(env_path.parent), prefix=".env.tmp.")
try:
    os.chmod(tmp_env, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    os.replace(tmp_env, env_path)
    os.chmod(env_path, 0o600)
except Exception:
    if os.path.exists(tmp_env):
        os.unlink(tmp_env)
    raise

if cfg_path.is_file():
    cfg_text = cfg_path.read_text(encoding="utf-8")
    cfg_text = re.sub(
        r"(?m)^(\s*api_id:\s*).*$",
        lambda m: f"{m.group(1)}{updates['TELEGRAM_API_ID']}",
        cfg_text,
        count=1,
    )
    cfg_text = re.sub(
        r"(?m)^(\s*api_hash:\s*).*$",
        lambda m: f'{m.group(1)}"{updates["TELEGRAM_API_HASH"]}"',
        cfg_text,
        count=1,
    )
    fd_c, tmp_cfg = tempfile.mkstemp(dir=str(cfg_path.parent), prefix=".userbot.yaml.tmp.")
    try:
        st_mode = cfg_path.stat().st_mode & 0o777
        os.chmod(tmp_cfg, st_mode)
        with os.fdopen(fd_c, "w", encoding="utf-8") as f:
            f.write(cfg_text)
        os.replace(tmp_cfg, cfg_path)
    except Exception:
        if os.path.exists(tmp_cfg):
            os.unlink(tmp_cfg)
        raise
PYEOF

echo "==> Файлы .env (0600) и config/userbot.yaml обновлены."

# 2. Interactive Telegram MTProto session re-authentication
echo ""
echo "--- Шаг 2. Авторизация Telegram-сессии (MTProto) ---"
echo "  [Напоминание] Перед входом убедись, что на новом аккаунте отключена"
echo "  синхронизация контактов и скрыт номер телефона (Конфиденциальность -> Никто)."
echo ""
echo "  Способ входа:"
echo "    [1] По QR-коду в терминале (рекомендуется — без ожидания кода/SMS)"
echo "    [2] По номеру телефона и коду подтверждения"
echo "    [0] Пропустить пересоздание Telegram-сессии"
read -r -p "Выбери способ [1]: " TG_AUTH_MODE
TG_AUTH_MODE="${TG_AUTH_MODE:-1}"

if [[ "${TG_AUTH_MODE}" == "1" || "${TG_AUTH_MODE}" == "2" ]]; then
  SESSION_DIR="$(dirname "${SESSION_PATH}")"
  mkdir -p "${SESSION_DIR}"
  chmod 700 "${SESSION_DIR}"

  BACKUP_SESSION=""
  TS="$(date +%Y%m%d_%H%M%S)"
  if [[ -f "${SESSION_PATH}" ]]; then
    BACKUP_DIR="${SESSION_DIR}/backups"
    mkdir -p "${BACKUP_DIR}"
    chmod 700 "${BACKUP_DIR}"
    BACKUP_SESSION="${BACKUP_DIR}/$(basename "${SESSION_PATH}").${TS}.bak"
    cp -p "${SESSION_PATH}" "${BACKUP_SESSION}"
    chmod 600 "${BACKUP_SESSION}"
    rm -f "${SESSION_PATH}" "${SESSION_PATH}-journal" "${SESSION_PATH}-wal" "${SESSION_PATH}-shm"
    echo "==> Старая сессия сохранена в бэкап: ${BACKUP_SESSION}"
  fi

  AUTH_CMD=("${BOT_BIN}" auth-telegram --config "${CONFIG_FILE}" --env-file "${ENV_FILE}")
  if [[ "${TG_AUTH_MODE}" == "1" ]]; then
    AUTH_CMD+=(--qr)
    echo "==> Запускаю вход по QR-коду..."
  else
    if [[ -n "${NEW_PHONE}" ]]; then
      AUTH_CMD+=(--phone "${NEW_PHONE}")
    fi
    echo "==> Запускаю вход по коду подтверждения..."
  fi

  if "${AUTH_CMD[@]}"; then
    # Verify that the newly created session is actually authorized
    IS_AUTHED="$("${PYTHON_BIN}" - "${SESSION_PATH}" "${NEW_API_ID}" "${NEW_API_HASH}" << 'PYEOF'
import asyncio
import sys
from pathlib import Path
from telethon import TelegramClient

session_path = Path(sys.argv[1]).resolve()
api_id = int(sys.argv[2])
api_hash = sys.argv[3]
session_name = str(session_path.with_suffix("")) if session_path.suffix == ".session" else str(session_path)

async def check() -> bool:
    if not session_path.is_file():
        return False
    client = TelegramClient(session_name, api_id=api_id, api_hash=api_hash)
    await client.connect()
    try:
        return bool(await client.is_user_authorized())
    finally:
        await client.disconnect()

print("yes" if asyncio.run(check()) else "no")
PYEOF
)"
    if [[ "${IS_AUTHED}" != "yes" ]]; then
      echo "[ERROR] Сессия не была авторизована." >&2
      rm -f "${SESSION_PATH}" "${SESSION_PATH}-journal" "${SESSION_PATH}-wal" "${SESSION_PATH}-shm"
      if [[ -n "${BACKUP_SESSION}" && -f "${BACKUP_SESSION}" ]]; then
        read -r -p "Восстановить предыдущую сессию из бэкапа? [Y/n]: " RESTORE_SES
        RESTORE_SES="${RESTORE_SES:-Y}"
        if [[ "${RESTORE_SES,,}" == "y" || "${RESTORE_SES,,}" == "yes" || "${RESTORE_SES,,}" == "д" ]]; then
          cp -p "${BACKUP_SESSION}" "${SESSION_PATH}"
          chmod 600 "${SESSION_PATH}"
          echo "==> Старая сессия восстановлена."
        fi
      fi
      exit 1
    fi
  else
    echo "[ERROR] Команда auth-telegram завершилась с ошибкой." >&2
    if [[ -n "${BACKUP_SESSION}" && -f "${BACKUP_SESSION}" ]]; then
      read -r -p "Восстановить предыдущую сессию из бэкапа? [Y/n]: " RESTORE_SES
      RESTORE_SES="${RESTORE_SES:-Y}"
      if [[ "${RESTORE_SES,,}" == "y" || "${RESTORE_SES,,}" == "yes" || "${RESTORE_SES,,}" == "д" ]]; then
        cp -p "${BACKUP_SESSION}" "${SESSION_PATH}"
        chmod 600 "${SESSION_PATH}"
        echo "==> Старая сессия восстановлена."
      fi
    fi
    exit 1
  fi
fi

# 3. Optional Antigravity OAuth re-authentication
echo ""
echo "--- Шаг 3. Аккаунт подписки Antigravity (agy OAuth) ---"
read -r -p "Переавторизовать аккаунт Antigravity в ${DATA_AGY_DIR}? [y/N]: " DO_AGY_AUTH
if [[ "${DO_AGY_AUTH,,}" == "y" || "${DO_AGY_AUTH,,}" == "yes" || "${DO_AGY_AUTH,,}" == "д" ]]; then
  TS_AGY="$(date +%Y%m%d_%H%M%S)"
  for tok in \
    "${DATA_AGY_DIR}/.gemini/jetski-standalone-oauth-token" \
    "${DATA_AGY_DIR}/.gemini/antigravity-cli/jetski-standalone-oauth-token" \
    "${DATA_AGY_DIR}/.gemini/antigravity-cli/antigravity-oauth-token"; do
    if [[ -f "${tok}" ]]; then
      mv "${tok}" "${tok}.${TS_AGY}.bak"
    fi
  done
  "${BOT_BIN}" auth-agy --data-dir "${DATA_AGY_DIR}"
fi

# 4. Optional memory database archive/reset
MEM_DB="${REPO_ROOT}/data/memory.sqlite"
if [[ -f "${MEM_DB}" ]]; then
  echo ""
  echo "--- Шаг 4. База памяти бота (data/memory.sqlite) ---"
  read -r -p "Перенести старую память в бэкап и начать с чистой базы? [y/N]: " RESET_MEM
  if [[ "${RESET_MEM,,}" == "y" || "${RESET_MEM,,}" == "yes" || "${RESET_MEM,,}" == "д" ]]; then
    mkdir -p "${REPO_ROOT}/data/backups"
    chmod 700 "${REPO_ROOT}/data/backups"
    TS_MEM="$(date +%Y%m%d_%H%M%S)"
    BACKUP_MEM="${REPO_ROOT}/data/backups/memory.pre-reauth.${TS_MEM}.sqlite"
    "${BOT_BIN}" backup-memory --config "${CONFIG_FILE}" --output-dir "${REPO_ROOT}/data/backups" --no-compress >/dev/null 2>&1 || cp -p "${MEM_DB}" "${BACKUP_MEM}"
    mv "${MEM_DB}" "${BACKUP_MEM}"
    rm -f "${MEM_DB}-wal" "${MEM_DB}-shm" "${MEM_DB}-journal"
    chmod 600 "${BACKUP_MEM}"
    echo "==> Старая база перенесена в ${BACKUP_MEM}"
  fi
fi

# Ensure botops owns all updated session, DB, config, and OAuth files when run as root
if [[ "$(id -u)" -eq 0 ]] && id botops >/dev/null 2>&1; then
  chown -R botops:botops "${REPO_ROOT}"
fi

# 5. Start systemd service
echo ""
echo "--- Шаг 5. Запуск сервиса ---"
DEFAULT_START="Y"
if [[ ${SERVICE_WAS_ACTIVE} -eq 0 ]]; then
  DEFAULT_START="Y"
fi
read -r -p "Запустить сервис ${SERVICE_NAME} сейчас? [${DEFAULT_START}/n]: " START_SVC
START_SVC="${START_SVC:-${DEFAULT_START}}"

if [[ "${START_SVC,,}" == "y" || "${START_SVC,,}" == "yes" || "${START_SVC,,}" == "д" ]]; then
  systemctl start "${SERVICE_NAME}"
  sleep 2
  systemctl status "${SERVICE_NAME}" --no-pager -l || true
  echo ""
  echo "Готово! Логи в реальном времени:"
  echo "  journalctl -u ${SERVICE_NAME} -n 50 -f"
else
  echo "Сервис не запущен. Когда будешь готов, выполни:"
  echo "  systemctl start ${SERVICE_NAME}"
fi
