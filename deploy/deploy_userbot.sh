#!/usr/bin/env bash
set -euo pipefail

# Deploy wrapper for 'userbot' profile (deploys to /opt/telegram-mtproto-userbot)
# Usage: ./deploy/deploy_userbot.sh [user@host] [remote_dir]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/deploy_remote.sh" "userbot" "$@"
