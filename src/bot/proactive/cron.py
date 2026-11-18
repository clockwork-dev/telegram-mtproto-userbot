"""Zero-dependency standard 5-field cron expression parser and timezone-aware evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Dict, FrozenSet, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TimezoneLike = Union[str, tzinfo]
_UTC = timezone.utc

_MONTH_ALIASES: Dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_DOW_ALIASES: Dict[str, int] = {
    "sun": 0,
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5,
    "sat": 6,
}

_MAX_DAYS_IN_MONTH: Dict[int, int] = {
    1: 31,
    2: 29,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}

_MAX_NEXT_RUN_DAYS = 366 * 9  # Covers 8-year leap-year gaps (e.g. 2096 -> 2104)


def parse_timezone(tz: TimezoneLike) -> tzinfo:
    """Validate and resolve a timezone string or tzinfo object."""
    if isinstance(tz, tzinfo):
        return tz
    if isinstance(tz, str):
        cleaned = tz.strip()
        if not cleaned:
            raise ValueError("Timezone must not be empty")
        try:
            return ZoneInfo(cleaned)
        except (ZoneInfoNotFoundError, KeyError, ValueError) as exc:
            raise ValueError(f"Invalid timezone: {tz!r}") from exc
    raise ValueError(f"Invalid timezone type: {type(tz).__name__}")


def _parse_int_or_alias(
    token: str,
    field_name: str,
    aliases: Optional[Dict[str, int]] = None,
) -> int:
    if not token:
        raise ValueError(f"Empty token in {field_name} field")
    lower = token.lower()
    if aliases is not None and lower in aliases:
        return aliases[lower]
    if token.isascii() and token.isdigit():
        return int(token)
    raise ValueError(f"Invalid token {token!r} in {field_name} field")


def _parse_cron_field(
    raw: str,
    field_name: str,
    min_val: int,
    max_val: int,
    aliases: Optional[Dict[str, int]] = None,
    allow_seven_as_zero: bool = False,
) -> Tuple[FrozenSet[int], bool]:
    if not raw:
        raise ValueError(f"Empty {field_name} field")

    is_wildcard = raw == "*"
    max_allowed = 7 if allow_seven_as_zero else max_val
    values: set[int] = set()

    for part in raw.split(","):
        if not part:
            raise ValueError(f"Empty list element in {field_name} field: {raw!r}")

        slash_parts = part.split("/")
        if len(slash_parts) > 2:
            raise ValueError(f"Invalid step syntax in {field_name} field: {part!r}")

        base = slash_parts[0]
        step = 1
        has_step = len(slash_parts) == 2
        if has_step:
            step_str = slash_parts[1]
            if not (step_str.isascii() and step_str.isdigit()):
                raise ValueError(f"Invalid step value {step_str!r} in {field_name} field")
            step = int(step_str)
            if step <= 0:
                raise ValueError(f"Step must be >= 1 in {field_name} field: {part!r}")

        if base == "*":
            start, end = min_val, max_val
        elif "-" in base:
            range_parts = base.split("-")
            if len(range_parts) != 2:
                raise ValueError(f"Invalid range syntax in {field_name} field: {part!r}")
            start = _parse_int_or_alias(range_parts[0], field_name, aliases)
            end = _parse_int_or_alias(range_parts[1], field_name, aliases)
            if start < min_val or start > max_allowed:
                raise ValueError(
                    f"Value {start} out of range [{min_val}, {max_allowed}] in {field_name}"
                )
            if end < min_val or end > max_allowed:
                raise ValueError(
                    f"Value {end} out of range [{min_val}, {max_allowed}] in {field_name}"
                )
            if start > end:
                raise ValueError(
                    f"Range start {start} > end {end} in {field_name} field: {part!r}"
                )
        else:
            if has_step:
                raise ValueError(
                    f"Step requires '*' or range in {field_name} field: {part!r}"
                )
            val = _parse_int_or_alias(base, field_name, aliases)
            if val < min_val or val > max_allowed:
                raise ValueError(
                    f"Value {val} out of range [{min_val}, {max_allowed}] in {field_name}"
                )
            start = end = val

        for v in range(start, end + 1, step):
            values.add(0 if (allow_seven_as_zero and v == 7) else v)

    return frozenset(values), is_wildcard


def _cron_dow(dt: datetime) -> int:
    """Convert Python datetime.weekday() (Mon=0..Sun=6) to cron DOW (Sun=0..Sat=6)."""
    return (dt.weekday() + 1) % 7


def _normalize_dt(dt: datetime, tz: tzinfo) -> datetime:
    """Convert dt to tz, attaching tz if naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


@dataclass(frozen=True)
class CronExpression:
    """Immutable 5-field standard cron expression (`minute hour dom month dow`)."""

    expression: str
    minutes: FrozenSet[int]
    hours: FrozenSet[int]
    days_of_month: FrozenSet[int]
    months: FrozenSet[int]
    days_of_week: FrozenSet[int]
    dom_wildcard: bool
    dow_wildcard: bool
    timezone: Optional[tzinfo] = None

    @classmethod
    def parse(
        cls,
        expr: str,
        timezone: Optional[TimezoneLike] = None,
    ) -> "CronExpression":
        """Parse a 5-field cron expression and optional default job timezone."""
        if not isinstance(expr, str):
            raise ValueError(f"Cron expression must be a string, got {type(expr).__name__}")

        cleaned = expr.strip()
        parts = cleaned.split()
        if len(parts) != 5:
            raise ValueError(
                f"Expected 5 cron fields (minute hour dom month dow), got {len(parts)}: {expr!r}"
            )

        min_raw, hour_raw, dom_raw, month_raw, dow_raw = parts

        minutes, _ = _parse_cron_field(min_raw, "minute", 0, 59)
        hours, _ = _parse_cron_field(hour_raw, "hour", 0, 23)
        days_of_month, dom_wildcard = _parse_cron_field(dom_raw, "day_of_month", 1, 31)
        months, _ = _parse_cron_field(month_raw, "month", 1, 12, aliases=_MONTH_ALIASES)
        days_of_week, dow_wildcard = _parse_cron_field(
            dow_raw,
            "day_of_week",
            0,
            6,
            aliases=_DOW_ALIASES,
            allow_seven_as_zero=True,
        )

        resolved_tz = parse_timezone(timezone) if timezone is not None else None

        return cls(
            expression=cleaned,
            minutes=minutes,
            hours=hours,
            days_of_month=days_of_month,
            months=months,
            days_of_week=days_of_week,
            dom_wildcard=dom_wildcard,
            dow_wildcard=dow_wildcard,
            timezone=resolved_tz,
        )

    def _resolve_tz(
        self,
        tz_override: Optional[TimezoneLike],
        *fallback_dts: datetime,
    ) -> tzinfo:
        if tz_override is not None:
            return parse_timezone(tz_override)
        if self.timezone is not None:
            return self.timezone
        for dt in fallback_dts:
            if dt.tzinfo is not None:
                return dt.tzinfo
        return timezone.utc

    def _matches_date(self, local_dt: datetime) -> bool:
        if local_dt.month not in self.months:
            return False

        dom_ok = local_dt.day in self.days_of_month
        dow_ok = _cron_dow(local_dt) in self.days_of_week

        if self.dom_wildcard and self.dow_wildcard:
            return True
        if self.dom_wildcard:
            return dow_ok
        if self.dow_wildcard:
            return dom_ok
        # Standard POSIX OR-semantics when both DOM and DOW are restricted.
        return dom_ok or dow_ok

    def _matches_local(self, local_dt: datetime) -> bool:
        return (
            local_dt.minute in self.minutes
            and local_dt.hour in self.hours
            and self._matches_date(local_dt)
        )

    def matches(
        self,
        dt: datetime,
        timezone: Optional[TimezoneLike] = None,
    ) -> bool:
        """Return True if `dt` matches the cron expression at minute granularity."""
        tz = self._resolve_tz(timezone, dt)
        local_dt = _normalize_dt(dt, tz)
        return self._matches_local(local_dt)

    def due_slots(
        self,
        since_dt: datetime,
        now_dt: datetime,
        timezone: Optional[TimezoneLike] = None,
        max_lookback_minutes: Optional[int] = 60,
    ) -> List[datetime]:
        """Return tz-aware minute-aligned matching slots in `(since_dt, now_dt]`.

        Iterates in UTC and projects each minute into the target timezone so
        spring-forward skipped wall times and fall-back repeated wall times are
        handled cleanly, and jittered polling loops never miss an exact minute.
        """
        if max_lookback_minutes is not None and max_lookback_minutes < 0:
            raise ValueError("max_lookback_minutes must be non-negative")

        tz = self._resolve_tz(timezone, now_dt, since_dt)
        since_utc = _normalize_dt(since_dt, tz).astimezone(_UTC)
        now_utc = _normalize_dt(now_dt, tz).astimezone(_UTC)

        if now_utc <= since_utc:
            return []

        first_after_since = since_utc.replace(second=0, microsecond=0) + timedelta(minutes=1)
        if max_lookback_minutes is not None:
            lookback_floor = now_utc - timedelta(minutes=max_lookback_minutes)
            if lookback_floor.second == 0 and lookback_floor.microsecond == 0:
                first_after_floor = lookback_floor
            else:
                first_after_floor = lookback_floor.replace(second=0, microsecond=0) + timedelta(
                    minutes=1
                )
            cur_utc = max(first_after_since, first_after_floor)
        else:
            cur_utc = first_after_since

        end_utc = now_utc.replace(second=0, microsecond=0)
        slots: List[datetime] = []
        while cur_utc <= end_utc:
            local_dt = cur_utc.astimezone(tz)
            if self._matches_local(local_dt):
                slots.append(local_dt)
            cur_utc += timedelta(minutes=1)

        return slots

    def next_run(
        self,
        after_dt: datetime,
        timezone: Optional[TimezoneLike] = None,
    ) -> datetime:
        """Compute the earliest matching tz-aware minute strictly after `after_dt`."""
        if self.dow_wildcard and not any(
            d <= _MAX_DAYS_IN_MONTH[m] for m in self.months for d in self.days_of_month
        ):
            raise ValueError(
                f"Cron expression {self.expression!r} never matches any calendar date"
            )

        tz = self._resolve_tz(timezone, after_dt)
        after_utc = _normalize_dt(after_dt, tz).astimezone(_UTC)
        cur_utc = after_utc.replace(second=0, microsecond=0) + timedelta(minutes=1)
        limit_utc = cur_utc + timedelta(days=_MAX_NEXT_RUN_DAYS)

        while cur_utc < limit_utc:
            local_dt = cur_utc.astimezone(tz)

            if not self._matches_date(local_dt):
                if local_dt.hour < 21:
                    cur_utc += timedelta(hours=21 - local_dt.hour)
                elif local_dt.minute < 40:
                    cur_utc += timedelta(minutes=40 - local_dt.minute)
                else:
                    cur_utc += timedelta(minutes=1)
                continue

            if local_dt.hour not in self.hours:
                if local_dt.minute < 40:
                    cur_utc += timedelta(minutes=40 - local_dt.minute)
                else:
                    cur_utc += timedelta(minutes=1)
                continue

            if local_dt.minute not in self.minutes:
                cur_utc += timedelta(minutes=1)
                continue

            return local_dt

        raise ValueError(
            f"No matching run time found within {_MAX_NEXT_RUN_DAYS} days for {self.expression!r}"
        )
