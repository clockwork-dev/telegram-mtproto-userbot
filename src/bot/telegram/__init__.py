"""Telegram MTProto Userbot package."""

from bot.telegram.auth import authenticate_session, enforce_session_permissions, interactive_auth
from bot.telegram.client import TelegramUserbot, calculate_typing_delay
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
