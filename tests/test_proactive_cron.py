"""Unit tests for 5-field CronExpression parser/evaluator and CronDigestJobConfig."""

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml
from pydantic import ValidationError

from bot.config.manager import ConfigManager
from bot.config.models import BotConfig, CronDigestJobConfig, ProactiveConfig
from bot.proactive.cron import CronExpression

KYIV_TZ = ZoneInfo("Europe/Kyiv")


def test_cron_parse_basic_and_aliases():
    """Verify standard 5-field parsing with wildcards, steps, ranges, lists, and aliases."""
    expr = CronExpression.parse("*/15 9-17 1,15 jan-mar,dec mon-fri")
    assert expr.minutes == frozenset({0, 15, 30, 45})
    assert expr.hours == frozenset(range(9, 18))
    assert expr.days_of_month == frozenset({1, 15})
    assert expr.months == frozenset({1, 2, 3, 12})
    assert expr.days_of_week == frozenset({1, 2, 3, 4, 5})
    assert expr.dom_wildcard is False
    assert expr.dow_wildcard is False


def test_cron_parse_stepped_ranges_and_sunday_zero_seven():
    """Verify stepped ranges and weekday 7 normalized to 0 (Sunday)."""
    expr = CronExpression.parse("5-20/5 0-23/6 * * 5-7")
    assert expr.minutes == frozenset({5, 10, 15, 20})
    assert expr.hours == frozenset({0, 6, 12, 18})
    assert expr.days_of_week == frozenset({5, 6, 0})
    assert expr.dom_wildcard is True
    assert expr.dow_wildcard is False

    sun_seven = CronExpression.parse("0 12 * * 7")
    sun_zero = CronExpression.parse("0 12 * * 0")
    sun_alias = CronExpression.parse("0 12 * * SUN")
    assert sun_seven.days_of_week == frozenset({0})
    assert sun_zero.days_of_week == frozenset({0})
    assert sun_alias.days_of_week == frozenset({0})

    # Sunday 2026-11-15
    dt_sun = datetime(2026, 11, 15, 12, 0, tzinfo=KYIV_TZ)
    assert sun_seven.matches(dt_sun)
    assert sun_zero.matches(dt_sun)
    assert sun_alias.matches(dt_sun)


def test_cron_posix_dom_dow_or_semantics():
    """Verify POSIX OR-semantics when both DOM and DOW are restricted, and AND when one is '*'."""
    # 2026-11-15 is Sunday (dow=0), 2026-11-16 is Monday (dow=1), 2026-11-23 is Monday (dow=1)
    both_restricted = CronExpression.parse("0 9 15 * 1", timezone="Europe/Kyiv")
    assert both_restricted.matches(datetime(2026, 11, 15, 9, 0, tzinfo=KYIV_TZ))  # 15th (Sun)
    assert both_restricted.matches(datetime(2026, 11, 16, 9, 0, tzinfo=KYIV_TZ))  # Mon (16th)
    assert not both_restricted.matches(datetime(2026, 11, 17, 9, 0, tzinfo=KYIV_TZ))  # Tue (17th)

    # DOM is '*' -> only DOW applies
    only_dow = CronExpression.parse("0 9 * * 1", timezone="Europe/Kyiv")
    assert not only_dow.matches(datetime(2026, 11, 15, 9, 0, tzinfo=KYIV_TZ))  # Sun 15th
    assert only_dow.matches(datetime(2026, 11, 16, 9, 0, tzinfo=KYIV_TZ))  # Mon 16th

    # DOW is '*' -> only DOM applies
    only_dom = CronExpression.parse("0 9 15 * *", timezone="Europe/Kyiv")
    assert only_dom.matches(datetime(2026, 11, 15, 9, 0, tzinfo=KYIV_TZ))
    assert not only_dom.matches(datetime(2026, 11, 16, 9, 0, tzinfo=KYIV_TZ))


@pytest.mark.parametrize(
    "bad_expr",
    [
        "",
        "   ",
        "* * * *",
        "* * * * * *",
        "60 * * * *",
        "0 24 * * *",
        "0 0 0 * *",
        "0 0 32 * *",
        "0 0 * 0 *",
        "0 0 * 13 *",
        "0 0 * * 8",
        "*/0 * * * *",
        "*/-1 * * * *",
        "*/abc * * * *",
        "5/10 * * * *",
        "10-5 * * * *",
        "1-2-3 * * * *",
        "1/2/3 * * * *",
        "1,,2 * * * *",
        ",1 * * * *",
        "1, * * * *",
        "0 0 * january *",
        "0 0 * * monday",
    ],
)
def test_cron_parse_invalid_expressions(bad_expr: str):
    """Invalid cron expressions must raise ValueError."""
    with pytest.raises(ValueError):
        CronExpression.parse(bad_expr)


def test_cron_invalid_timezone():
    """Invalid timezone names must raise ValueError."""
    with pytest.raises(ValueError):
        CronExpression.parse("0 9 * * *", timezone="Mars/Olympus_Mons")
    with pytest.raises(ValueError):
        CronExpression.parse("0 9 * * *", timezone="   ")


def test_kyiv_timezone_and_dst_transitions():
    """Verify Europe/Kyiv UTC offset shifts and spring-forward / fall-back DST behavior."""
    morning = CronExpression.parse("0 9 * * *", timezone="Europe/Kyiv")

    # Winter (EET = UTC+2): 2026-01-15 09:00 Kyiv == 07:00 UTC
    assert morning.matches(datetime(2026, 1, 15, 7, 0, tzinfo=timezone.utc))
    assert not morning.matches(datetime(2026, 1, 15, 6, 0, tzinfo=timezone.utc))

    # Summer (EEST = UTC+3): 2026-07-15 09:00 Kyiv == 06:00 UTC
    assert morning.matches(datetime(2026, 7, 15, 6, 0, tzinfo=timezone.utc))
    assert not morning.matches(datetime(2026, 7, 15, 7, 0, tzinfo=timezone.utc))

    # Spring-forward in Europe/Kyiv: 2026-03-29 03:00 jumps to 04:00
    three_am = CronExpression.parse("0 3 * * *", timezone="Europe/Kyiv")
    before_jump_utc = datetime(2026, 3, 29, 0, 30, tzinfo=timezone.utc)  # 02:30+02:00
    # 03:00 does not exist on 2026-03-29, so next_run skips to 2026-03-30 03:00+03:00
    nxt = three_am.next_run(before_jump_utc)
    assert nxt == datetime(2026, 3, 30, 3, 0, tzinfo=KYIV_TZ)
    assert nxt.utcoffset().total_seconds() == 3 * 3600

    # 09:00 on spring-forward day is 06:00 UTC
    due_spring = morning.due_slots(
        since_dt=datetime(2026, 3, 29, 5, 59, 30, tzinfo=timezone.utc),
        now_dt=datetime(2026, 3, 29, 6, 0, 15, tzinfo=timezone.utc),
    )
    assert due_spring == [datetime(2026, 3, 29, 9, 0, tzinfo=KYIV_TZ)]

    # Fall-back in Europe/Kyiv: 2026-10-25 04:00+03:00 falls back to 03:00+02:00
    # 00:00 UTC = 03:00+03:00 (fold=0), 01:00 UTC = 03:00+02:00 (fold=1)
    due_fall = three_am.due_slots(
        since_dt=datetime(2026, 10, 24, 23, 50, tzinfo=timezone.utc),
        now_dt=datetime(2026, 10, 25, 1, 10, tzinfo=timezone.utc),
        max_lookback_minutes=120,
    )
    assert len(due_fall) == 2
    assert due_fall[0].astimezone(timezone.utc) == datetime(2026, 10, 25, 0, 0, tzinfo=timezone.utc)
    assert due_fall[1].astimezone(timezone.utc) == datetime(2026, 10, 25, 1, 0, tzinfo=timezone.utc)

    # 09:00 on fall-back day is 07:00 UTC (UTC+2)
    assert morning.matches(datetime(2026, 10, 25, 7, 0, tzinfo=timezone.utc))


def test_due_slots_poll_jitter_and_lookback_cap():
    """Verify due_slots handles polling jitter, boundary exclusivity, and lookback cap."""
    morning = CronExpression.parse("0 9 * * *", timezone="Europe/Kyiv")

    # Jittered poll: previous poll at 08:59:55, current poll at 09:01:02
    slots = morning.due_slots(
        since_dt=datetime(2026, 11, 13, 8, 59, 55, tzinfo=KYIV_TZ),
        now_dt=datetime(2026, 11, 13, 9, 1, 2, tzinfo=KYIV_TZ),
    )
    assert slots == [datetime(2026, 11, 13, 9, 0, tzinfo=KYIV_TZ)]
    assert slots[0].tzinfo == KYIV_TZ

    # Exact minute boundary: (08:59:00, 09:00:00] includes 09:00:00
    exact_hit = morning.due_slots(
        since_dt=datetime(2026, 11, 13, 8, 59, 0, tzinfo=KYIV_TZ),
        now_dt=datetime(2026, 11, 13, 9, 0, 0, tzinfo=KYIV_TZ),
    )
    assert exact_hit == [datetime(2026, 11, 13, 9, 0, tzinfo=KYIV_TZ)]

    # Subsequent poll after 09:00:00 does NOT re-fire 09:00:00
    no_dup = morning.due_slots(
        since_dt=datetime(2026, 11, 13, 9, 0, 0, tzinfo=KYIV_TZ),
        now_dt=datetime(2026, 11, 13, 9, 0, 45, tzinfo=KYIV_TZ),
    )
    assert no_dup == []

    # Empty when now_dt <= since_dt
    assert (
        morning.due_slots(
            since_dt=datetime(2026, 11, 13, 9, 5, tzinfo=KYIV_TZ),
            now_dt=datetime(2026, 11, 13, 9, 5, tzinfo=KYIV_TZ),
        )
        == []
    )

    # Lookback cap of 60 minutes prevents firing a slot from 3 hours ago
    stale = morning.due_slots(
        since_dt=datetime(2026, 11, 13, 6, 0, tzinfo=KYIV_TZ),
        now_dt=datetime(2026, 11, 13, 12, 0, tzinfo=KYIV_TZ),
        max_lookback_minutes=60,
    )
    assert stale == []

    # Multiple matches within lookback window
    every_15 = CronExpression.parse("*/15 * * * *", timezone="Europe/Kyiv")
    multi = every_15.due_slots(
        since_dt=datetime(2026, 11, 13, 8, 55, tzinfo=KYIV_TZ),
        now_dt=datetime(2026, 11, 13, 9, 32, tzinfo=KYIV_TZ),
    )
    assert multi == [
        datetime(2026, 11, 13, 9, 0, tzinfo=KYIV_TZ),
        datetime(2026, 11, 13, 9, 15, tzinfo=KYIV_TZ),
        datetime(2026, 11, 13, 9, 30, tzinfo=KYIV_TZ),
    ]


def test_next_run_and_leap_year():
    """Verify next_run strictly advances past after_dt, handles leap years, and rejects impossible dates."""
    morning = CronExpression.parse("0 9 * * *", timezone="Europe/Kyiv")

    assert morning.next_run(datetime(2026, 11, 13, 8, 59, 59, tzinfo=KYIV_TZ)) == datetime(
        2026, 11, 13, 9, 0, tzinfo=KYIV_TZ
    )
    assert morning.next_run(datetime(2026, 11, 13, 9, 0, 0, tzinfo=KYIV_TZ)) == datetime(
        2026, 11, 14, 9, 0, tzinfo=KYIV_TZ
    )

    leap = CronExpression.parse("0 0 29 2 *", timezone="Europe/Kyiv")
    assert leap.next_run(datetime(2024, 2, 29, 0, 0, tzinfo=KYIV_TZ)) == datetime(
        2028, 2, 29, 0, 0, tzinfo=KYIV_TZ
    )

    impossible = CronExpression.parse("0 0 31 2 *", timezone="Europe/Kyiv")
    with pytest.raises(ValueError):
        impossible.next_run(datetime(2026, 1, 1, 0, 0, tzinfo=KYIV_TZ))


def test_cron_digest_pydantic_validation_and_backward_compat():
    """Verify CronDigestJobConfig validation and backward-compatible ProactiveConfig default."""
    default_proactive = ProactiveConfig()
    assert default_proactive.cron_digests == []

    job = CronDigestJobConfig(
        id="test_digest",
        cron="0 9 * * *",
        timezone="Europe/Kyiv",
        slot_label="morning",
        topics=["tech", "world"],
    )
    assert job.enabled is True
    assert job.respect_quiet_hours is True
    assert job.chat_ids == []

    with pytest.raises(ValidationError):
        CronDigestJobConfig(id="bad_cron", cron="99 * * * *")

    with pytest.raises(ValidationError):
        CronDigestJobConfig(id="bad_tz", cron="0 9 * * *", timezone="Invalid/Zone")

    with pytest.raises(ValidationError):
        CronDigestJobConfig(id="   ", cron="0 9 * * *")


def test_repo_bot_yaml_loads_default_cron_digests():
    """Verify bot.example.yaml (or bot.yaml) loads morning_digest and evening_digest via ConfigManager."""
    repo_config_dir = Path(__file__).resolve().parent.parent / "config"
    mgr = ConfigManager(config_path=repo_config_dir / "bot.example.yaml")
    assert mgr.bot is not None

    digests = {job.id: job for job in mgr.bot.proactive.cron_digests}
    assert "morning_digest" in digests
    assert "evening_digest" in digests

    morning = digests["morning_digest"]
    assert morning.enabled is True
    assert morning.cron == "0 9 * * *"
    assert morning.timezone == "Europe/Kyiv"
    assert morning.respect_quiet_hours is True
    assert morning.prompt_hint != ""

    evening = digests["evening_digest"]
    assert evening.enabled is True
    assert evening.cron == "0 20 * * *"
    assert evening.timezone == "Europe/Kyiv"
    assert evening.respect_quiet_hours is True
    assert evening.prompt_hint != ""


def test_bot_config_yaml_roundtrip_without_cron_digests(tmp_path: Path):
    """Verify BotConfig/ConfigManager still loads legacy YAML without cron_digests."""
    bot_yaml = tmp_path / "bot.yaml"
    bot_yaml.write_text(
        yaml.dump(
            {
                "telegram": {"api_id": 12345, "api_hash": "abcdef"},
                "proactive": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )
    mgr = ConfigManager(config_dir=tmp_path)
    assert isinstance(mgr.bot, BotConfig)
    assert mgr.bot.proactive.cron_digests == []
