from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional, Union

import pandas as pd
from zoneinfo import ZoneInfo


# Centralised timezone definitions for user-facing timestamps.
DISPLAY_TIMEZONE = ZoneInfo("America/New_York")
UTC_TIMEZONE = ZoneInfo("UTC")


def _convert_epoch_numeric(value: Union[int, float]) -> Optional[pd.Timestamp]:
    """Return a timestamp parsed from a numeric epoch value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    absolute = abs(float(value))
    if absolute < 1e11:
        unit = "s"
    elif absolute < 1e14:
        unit = "ms"
    elif absolute < 1e17:
        unit = "us"
    else:
        unit = "ns"

    return pd.to_datetime(value, unit=unit, errors="coerce", utc=True)


def coerce_to_display_timezone(value: Any) -> Optional[pd.Timestamp]:
    """Convert any timestamp-like value to the configured display timezone."""
    if value is None:
        return None

    timestamp: Optional[pd.Timestamp]

    if isinstance(value, pd.Timestamp):
        timestamp = value
    elif isinstance(value, datetime):
        timestamp = pd.Timestamp(value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        timestamp = _convert_epoch_numeric(value)
    else:
        if isinstance(value, str) and not value.strip():
            return None
        try:
            timestamp = pd.to_datetime(value, errors="coerce")
        except Exception:
            return None

    if timestamp is None or pd.isna(timestamp):
        return None

    try:
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(UTC_TIMEZONE)
        return timestamp.tz_convert(DISPLAY_TIMEZONE)
    except Exception:
        return None


def now_in_display_timezone() -> datetime:
    """Return the current time in the configured display timezone."""
    return datetime.now(tz=DISPLAY_TIMEZONE)
