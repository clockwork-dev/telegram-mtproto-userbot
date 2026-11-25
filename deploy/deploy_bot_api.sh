#!/usr/bin/env bash
set -euo pipefail

# Deploy wrapper for 'bot_api' profile (deploys to /opt/telegram-group-bot)
# Usage: ./deploy/deploy_bot_api.sh [user@host] [remote_dir]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/deploy_remote.sh" "bot_api" "$@"
