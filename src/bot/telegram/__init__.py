"""Telegram MTProto Userbot & Bot API transport package."""

from bot.telegram.auth import authenticate_session, enforce_session_permissions, interactive_auth
from bot.telegram.bot_api_client import (
    BotAPIMessageEvent,
    TelegramBotAPIClient,
    normalize_reaction_emoji,
    resolve_bot_token_from_env_or_config,
)
from bot.telegram.client import (
    MAX_DOCUMENT_SIZE_BYTES,
    TelegramUserbot,
    calculate_typing_delay,
    validate_document_path,
)
from bot.telegram.handlers import (
    IncomingMessage,
    MessageDispatcher,
    enrich_reply_metadata,
    parse_incoming_message,
    parse_incoming_message_sync,
)
from bot.telegram.heuristics import ReplyAction, ReplyHeuristics

__all__ = [
    "TelegramUserbot",
    "TelegramBotAPIClient",
    "BotAPIMessageEvent",
    "MAX_DOCUMENT_SIZE_BYTES",
    "validate_document_path",
    "normalize_reaction_emoji",
    "resolve_bot_token_from_env_or_config",
    "calculate_typing_delay",
    "IncomingMessage",
    "MessageDispatcher",
    "enrich_reply_metadata",
    "parse_incoming_message",
    "parse_incoming_message_sync",
    "ReplyAction",
    "ReplyHeuristics",
    "interactive_auth",
    "authenticate_session",
    "enforce_session_permissions",
]
