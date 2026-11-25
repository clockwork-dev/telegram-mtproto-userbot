"""Direct Python SDK client for google-antigravity."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bot.antigravity.auth import (
    build_clean_child_env,
    resolve_antigravity_home,
    setup_isolated_home,
    sync_updated_isolated_tokens,
)
from bot.antigravity.errors import (
    AntigravityError,
    AntigravityRateLimitError,
    classify_error,
)
from bot.antigravity.stream import ToolProfile, resolve_allowed_tools
from bot.core.models import MediaAttachment, ProviderResponse, UsageInfo

logger = logging.getLogger(__name__)

try:
    from google.antigravity import (
        Agent,
        CapabilitiesConfig,
        LocalAgentConfig,
        from_bytes,
        from_file,
    )

    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False


def is_sdk_available() -> bool:
    """Return True if google-antigravity SDK is installed and usable."""
    return _SDK_AVAILABLE


def has_gemini_api_key() -> bool:
    """Return True if GEMINI_API_KEY is configured in the environment."""
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


_SDK_CHAT_TOOLS: list[str] = ["search_web", "read_url_content"]
_SDK_OPS_TOOLS: list[str] = [
    "search_web",
    "read_url_content",
    "list_directory",
    "search_directory",
    "find_file",
    "view_file",
    "create_file",
    "edit_file",
    "run_command",
]


class AntigravitySDKClient:
    """Direct Python integration with google-antigravity SDK."""

    def __init__(
        self,
        data_dir: Path | str | None = None,
        default_model: str = "gemini-3.8-flash",
    ) -> None:
        if not _SDK_AVAILABLE:
            raise ImportError(
                "google-antigravity is not installed. Install via pip install google-antigravity."
            )
        self.data_dir = resolve_antigravity_home(data_dir)
        self.default_model = default_model

    async def generate_response(
        self,
        prompt: str,
        system_instructions: str | None = None,
        media_attachments: Sequence[MediaAttachment] | None = None,
        conversation_id: str | None = None,
        model: str | None = None,
        tool_profile: ToolProfile = "chat",
        mode: str | None = None,
        cwd: Path | str | None = None,
    ) -> ProviderResponse:
        """Execute chat turn using direct Agent SDK with tool interception and isolation."""
        resolve_allowed_tools(tool_profile)
        if not has_gemini_api_key() and getattr(Agent, "__module__", "").startswith("google."):
            raise AntigravityError(
                "Direct SDK requires GEMINI_API_KEY; using OAuth CLI transport instead."
            )

        resolved_model = model or self.default_model

        isolated_home, gemini_dir = setup_isolated_home(self.data_dir)
        clean_env = build_clean_child_env(isolated_home, force_file_auth=True)

        # Build prompt payload (including multimodal attachments)
        prompt_items: list[Any] = [prompt]
        if media_attachments:
            for att in media_attachments:
                try:
                    if att.file_path and att.file_path.is_file():
                        prompt_items.append(from_file(att.file_path, description=att.description))
                    elif att.data:
                        prompt_items.append(
                            from_bytes(att.data, mime_type=att.mime_type, description=att.description)
                        )
                except Exception as exc:
                    logger.warning("Failed preparing media attachment %s for SDK: %s", att, exc)

        chat_input = prompt_items if len(prompt_items) > 1 else prompt

        enabled_sdk_tools = (
            list(_SDK_OPS_TOOLS)
            if tool_profile in ("ops", "self_dev")
            else list(_SDK_CHAT_TOOLS)
        )
        caps = CapabilitiesConfig(
            enable_subagents=(tool_profile in ("ops", "self_dev")),
            enabled_tools=enabled_sdk_tools,
        )

        agent_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "system_instructions": system_instructions,
            "capabilities": caps,
            "conversation_id": conversation_id,
            "env": clean_env,
        }
        if cwd is not None:
            agent_kwargs["workspaces"] = [str(Path(cwd).resolve())]
        agent_config = LocalAgentConfig(**agent_kwargs)

        # Temporary env patching so any internal Go/keyring checks use isolated home
        old_env = dict(os.environ)
        os.environ.update(clean_env)

        try:
            async with Agent(agent_config) as agent:
                response = await agent.chat(chat_input)

                # Collect reasoning / thinking tokens
                thinking_parts: list[str] = []
                try:
                    if hasattr(response, "thoughts"):
                        async for thought in response.thoughts:
                            if isinstance(thought, str):
                                thinking_parts.append(thought)
                except Exception as exc:
                    logger.debug("Thoughts extraction ended: %s", exc)

                text = await response.text()
                extracted_conv = getattr(agent, "conversation_id", conversation_id)
                stop_reason = getattr(response, "stop_reason", "stop") or "stop"

                # Extract token usage
                usage_info: UsageInfo | None = None
                meta = getattr(response, "usage_metadata", None)
                if meta is not None:
                    usage_info = UsageInfo(
                        prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
                        completion_tokens=getattr(meta, "candidates_token_count", 0) or 0,
                        total_tokens=getattr(meta, "total_token_count", 0) or 0,
                        reasoning_tokens=getattr(meta, "thoughts_token_count", 0) or 0,
                        cached_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
                        input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
                        output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
                    )

                return ProviderResponse(
                    text=text,
                    thinking="".join(thinking_parts) or None,
                    conversation_id=extracted_conv,
                    finish_reason=stop_reason,
                    usage=usage_info,
                    model=resolved_model,
                    raw_response=response,
                )

        except Exception as exc:
            verdict = classify_error(exc)
            if verdict.category == "rate_limit":
                raise AntigravityRateLimitError(str(exc), verdict=verdict) from exc
            raise AntigravityError(str(exc), verdict=verdict) from exc
        finally:
            os.environ.clear()
            os.environ.update(old_env)
            sync_updated_isolated_tokens(gemini_dir, data_dir=self.data_dir)
