"""Unit tests for QuotaBudgetManager: usage_history, cooldowns, economy mode, routing, and /quota."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from bot.config.models import ModelRoleConfig, ModelRouterConfig, QuotaConfig
from bot.core.models import UsageInfo
from bot.quota.manager import QuotaBudgetManager


class _FakeUsageStore:
    """In-memory mock matching MemoryStore's usage_history interface."""

    def __init__(self, initial_rows: list[dict[str, Any]] | None = None) -> None:
        self.rows: list[dict[str, Any]] = list(initial_rows or [])

    def _sync_get_usage_rows(self, since_iso: str = "") -> list[dict[str, Any]]:
        return [r for r in self.rows if not since_iso or str(r.get("timestamp") or "") >= since_iso]

    async def get_usage_rows(self, since_iso: str = "") -> list[dict[str, Any]]:
        return self._sync_get_usage_rows(since_iso=since_iso)

    async def record_usage(self, **kwargs: Any) -> None:
        self.rows.append(dict(kwargs))


@pytest.mark.asyncio
async def test_quota_usage_recording_sync_and_economy_threshold(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    two_hours_ago = (now - timedelta(hours=2)).isoformat()
    store = _FakeUsageStore(
        initial_rows=[
            {
                "timestamp": two_hours_ago,
                "model": "gemini-3.8-flash",
                "source": "group",
                "total_tokens": 50_000,
            }
        ]
    )
    q_cfg = QuotaConfig(
        enabled=True,
        hourly_token_budget=100_000,
        daily_token_budget=1_000_000,
        economy_threshold_ratio=0.75,
    )
    qm = QuotaBudgetManager(memory_store=store, quota_config=q_cfg, data_dir=tmp_path)  # type: ignore[arg-type]

    h0, d0 = qm.get_usage_totals()
    assert h0 == 0
    assert d0 == 50_000
    assert qm.is_economy_mode() is False
    assert qm.format_context_hint() == ""

    await qm.record_usage(
        UsageInfo(input_tokens=60_000, output_tokens=20_000, total_tokens=80_000),
        model="gemini-4-hydrogen",
        source="group",
    )
    h1, d1 = qm.get_usage_totals()
    assert h1 == 80_000
    assert d1 == 130_000
    assert len(store.rows) == 2

    # 80,000 >= 100,000 * 0.75 -> enters economy mode
    assert qm.is_economy_mode() is True
    assert qm.consume_economy_notification_flag() is True
    assert qm.consume_economy_notification_flag() is False
    assert qm.should_allow_proactive() is False
    assert qm.should_allow_spontaneous() is False
    assert qm.should_allow_self_edit(priority="auto")[0] is False
    assert qm.should_allow_self_edit(priority="admin")[0] is True

    # Cache-friendly hint contains NO rolling token counters
    hint = qm.format_context_hint()
    assert "mode=economy" in hint
    assert "tokens_1h=" not in hint
    assert "tokens_24h=" not in hint


def test_quota_cooldowns_buckets_and_model_routing(tmp_path: Path) -> None:
    router = ModelRouterConfig(
        group=ModelRoleConfig(model="gemini-4-hydrogen", effort="low"),
        dm_admin=ModelRoleConfig(model="gemini-4-hydrogen", effort="medium"),
        self_edit_judge=ModelRoleConfig(model="gemini-4-hydrogen", effort="low"),
        self_edit=ModelRoleConfig(model="claude-sonnet-5-5", effort="medium"),
        fallback_models=[
            ModelRoleConfig(model="gemini-4-argon", effort="medium"),  # auto-excluded!
            ModelRoleConfig(model="gemini-3.8-flash", effort="low"),
            ModelRoleConfig(model="claude-sonnet-5-5", effort="medium"),
            ModelRoleConfig(model="gpt-oss-120b-medium", effort=None),
        ],
    )
    qm = QuotaBudgetManager(model_router=router, data_dir=tmp_path)

    # Normal routing
    assert qm.select_model("group") == ("gemini-4-hydrogen", "low")
    assert qm.select_model("dm_admin") == ("gemini-4-hydrogen", "medium")
    # self_edit / self_edit_judge preserve high effort
    assert qm.select_model("self_edit_judge") == ("gemini-4-hydrogen", "high")
    assert qm.select_model("self_edit") == ("claude-sonnet-5-5", "high")

    # Cooldown on gemini-4-hydrogen-medium strips effort suffix and rotates group -> gemini-3.8-flash (skipping argon!)
    qm.record_rate_limit("gemini-4-hydrogen-medium", retry_after=120.0, bucket_wide=False)
    assert qm.is_model_cooled_down("gemini-4-hydrogen") is True
    assert qm.is_model_cooled_down("gemini-3.8-flash") is False
    assert qm.select_model("group") == ("gemini-3.8-flash", "low")

    # Bucket-wide cooldown on gemini_models rotates -> claude-sonnet-5-5
    qm.record_rate_limit("gemini-3.8-flash", retry_after=120.0, bucket_wide=True)
    assert qm.get_bucket_for_model("gemini-3.8-flash") == "gemini_models"
    assert qm.is_model_cooled_down("gemini-3.8-flash") is True
    assert qm.select_model("group") == ("claude-sonnet-5-5", "medium")

    # Cooldown on claude-sonnet-5-5 rotates -> gpt-oss-120b-medium (preserving its literal model ID)
    qm.record_rate_limit("claude-sonnet-5-5", retry_after=120.0, bucket_wide=False)
    assert qm.select_model("group") == ("gpt-oss-120b-medium", "medium")


def test_quota_warnings_ttl_and_live_quota_tsv_report(tmp_path: Path) -> None:
    qm = QuotaBudgetManager(data_dir=tmp_path)

    future_reset = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    qm.record_quota_warning("gemini-4-hydrogen", remaining_pct=12.5, resets_at=future_reset)
    assert qm.is_economy_mode() is True
    assert qm.select_model("group") == ("gemini-3.8-flash", "low")

    tsv = (
        "Gemini Models\t5h rolling\t84%\t2026-11-24T18:00:00Z\n"
        "Claude & GPT\t5h rolling\t100%\t2026-11-24T19:00:00Z\n"
    )
    with (
        patch("bot.quota.manager.resolve_antigravity_binary", return_value="/mock/bin/agy"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stdout=tsv, stderr="")
        parsed = qm.sync_live_quota()
    assert len(parsed) == 2
    assert parsed[0]["bucket"] == "Gemini Models"
    assert parsed[0]["remaining_pct"] == 84.0

    report = qm.format_quota_report()
    assert "Quota & Budget Status" in report
    assert "ECONOMY" in report
    assert "Gemini Models" in report
    assert "gemini-4-hydrogen" in report

    # Warning expires after resets_at passes
    qm._prune_quota_warnings(now_dt=datetime.now(timezone.utc) + timedelta(minutes=15))
    assert qm.is_economy_mode() is False
    assert "gemini-4-hydrogen" not in qm._quota_warnings
