"""US Federal Holidays & Calendar Feature Engineering.

Zero external dependencies: self-contained calculation of US Federal holidays,
observed dates, and adjacent indicator features (day-before, day-after, bridge days).
"""

from __future__ import annotations

import functools
import numpy as np
import pandas as pd


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """Return the nth occurrence of a weekday in a month (0=Monday, 6=Sunday)."""
    first_day = pd.Timestamp(year, month, 1)
    offset = (weekday - first_day.weekday()) % 7
    return first_day + pd.Timedelta(days=offset + (n - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    """Return the last occurrence of a weekday in a month."""
    next_month = pd.Timestamp(year + 1, 1, 1) if month == 12 else pd.Timestamp(year, month + 1, 1)
    last_day = next_month - pd.Timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - pd.Timedelta(days=offset)


def _adjust_observed(ts: pd.Timestamp) -> pd.Timestamp:
    """Apply federal rule: Saturday holiday observed Friday; Sunday observed Monday."""
    if ts.weekday() == 5:  # Saturday
        return ts - pd.Timedelta(days=1)
    if ts.weekday() == 6:  # Sunday
        return ts + pd.Timedelta(days=1)
    return ts


@functools.lru_cache(maxsize=32)
def get_us_federal_holidays(year: int) -> set[pd.Timestamp]:
    """Return all observed US federal holidays for a given year as timestamps."""
    holidays = set()

    # 1. New Year's Day (Jan 1)
    holidays.add(_adjust_observed(pd.Timestamp(year, 1, 1)))

    # 2. Martin Luther King Jr. Day (3rd Monday in January)
    holidays.add(_nth_weekday(year, 1, 0, 3))

    # 3. Washington's Birthday / Presidents' Day (3rd Monday in February)
    holidays.add(_nth_weekday(year, 2, 0, 3))

    # 4. Memorial Day (Last Monday in May)
    holidays.add(_last_weekday(year, 5, 0))

    # 5. Juneteenth National Independence Day (June 19)
    holidays.add(_adjust_observed(pd.Timestamp(year, 6, 19)))

    # 6. Independence Day (July 4)
    holidays.add(_adjust_observed(pd.Timestamp(year, 7, 4)))

    # 7. Labor Day (1st Monday in September)
    holidays.add(_nth_weekday(year, 9, 0, 1))

    # 8. Columbus Day (2nd Monday in October)
    holidays.add(_nth_weekday(year, 10, 0, 2))

    # 9. Veterans Day (November 11)
    holidays.add(_adjust_observed(pd.Timestamp(year, 11, 11)))

    # 10. Thanksgiving Day (4th Thursday in November)
    holidays.add(_nth_weekday(year, 11, 3, 4))

    # 11. Christmas Day (December 25)
    holidays.add(_adjust_observed(pd.Timestamp(year, 12, 25)))

    return holidays


def get_holiday_set(min_year: int, max_year: int) -> set[pd.Timestamp]:
    """Return observed federal holidays across a range of years (inclusive)."""
    # Include year-1 and year+1 to capture boundary shifts (e.g. Dec 31 observed New Year)
    all_holidays = set()
    for y in range(min_year - 1, max_year + 2):
        all_holidays.update(get_us_federal_holidays(y))
    return all_holidays


def build_holiday_features(dates: pd.Series) -> pd.DataFrame:
    """Generate 4 holiday indicator features for a Series of target dates.

    Features generated:
      - is_holiday: 1 if date is an observed US federal holiday, 0 otherwise.
      - is_day_before_holiday: 1 if tomorrow is a holiday.
      - is_day_after_holiday: 1 if yesterday was a holiday.
      - is_bridge_day: 1 if date is adjacent to a holiday weekend (e.g. Friday
        before Monday holiday, Monday after Friday holiday, Black Friday).
    """
    dt_series = pd.to_datetime(dates).dt.normalize()
    if dt_series.empty:
        return pd.DataFrame(
            columns=["is_holiday", "is_day_before_holiday", "is_day_after_holiday", "is_bridge_day"]
        )

    min_yr = dt_series.dt.year.min()
    max_yr = dt_series.dt.year.max()
    holiday_set = get_holiday_set(min_yr, max_yr)

    is_holiday = dt_series.isin(holiday_set).astype(int)
    is_day_before = (dt_series + pd.Timedelta(days=1)).isin(holiday_set).astype(int)
    is_day_after = (dt_series - pd.Timedelta(days=1)).isin(holiday_set).astype(int)

    dayofweek = dt_series.dt.dayofweek
    # Bridge days:
    # 1) Friday (dayofweek=4) where Monday (d+3) is a holiday
    # 2) Monday (dayofweek=0) where Friday (d-3) was a holiday
    # 3) Friday (dayofweek=4) where Thursday (d-1) was a holiday (e.g. day after Thanksgiving)
    # 4) Monday (dayofweek=0) where Tuesday (d+1) is a holiday
    bridge_friday_mon_hol = (dayofweek == 4) & (dt_series + pd.Timedelta(days=3)).isin(holiday_set)
    bridge_mon_fri_hol = (dayofweek == 0) & (dt_series - pd.Timedelta(days=3)).isin(holiday_set)
    bridge_day_after_thu = (dayofweek == 4) & (dt_series - pd.Timedelta(days=1)).isin(holiday_set)
    bridge_day_before_tue = (dayofweek == 0) & (dt_series + pd.Timedelta(days=1)).isin(holiday_set)

    is_bridge = (
        bridge_friday_mon_hol | bridge_mon_fri_hol | bridge_day_after_thu | bridge_day_before_tue
    ).astype(int)

    return pd.DataFrame({
        "is_holiday": is_holiday.to_numpy(),
        "is_day_before_holiday": is_day_before.to_numpy(),
        "is_day_after_holiday": is_day_after.to_numpy(),
        "is_bridge_day": is_bridge.to_numpy(),
    }, index=dates.index)
