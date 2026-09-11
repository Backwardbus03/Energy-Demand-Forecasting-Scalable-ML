"""Unit tests for US Federal Holidays feature engineering module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.holidays import (
    _adjust_observed,
    build_holiday_features,
    get_us_federal_holidays,
)


class TestHolidayGeneration:
    def test_known_holidays_2024(self):
        holidays_2024 = get_us_federal_holidays(2024)
        expected = {
            pd.Timestamp("2024-01-01"),  # New Year's
            pd.Timestamp("2024-01-15"),  # MLK
            pd.Timestamp("2024-02-19"),  # Presidents
            pd.Timestamp("2024-05-27"),  # Memorial Day
            pd.Timestamp("2024-06-19"),  # Juneteenth
            pd.Timestamp("2024-07-04"),  # Independence Day
            pd.Timestamp("2024-09-02"),  # Labor Day
            pd.Timestamp("2024-10-14"),  # Columbus Day
            pd.Timestamp("2024-11-11"),  # Veterans Day
            pd.Timestamp("2024-11-28"),  # Thanksgiving
            pd.Timestamp("2024-12-25"),  # Christmas
        }
        assert expected == holidays_2024

    def test_saturday_observed_shift_friday(self):
        # Christmas 2021 was Saturday Dec 25 -> observed Friday Dec 24
        holidays_2021 = get_us_federal_holidays(2021)
        assert pd.Timestamp("2021-12-24") in holidays_2021
        assert pd.Timestamp("2021-12-25") not in holidays_2021

    def test_sunday_observed_shift_monday(self):
        # July 4, 2021 was Sunday -> observed Monday July 5
        holidays_2021 = get_us_federal_holidays(2021)
        assert pd.Timestamp("2021-07-05") in holidays_2021
        assert pd.Timestamp("2021-07-04") not in holidays_2021


class TestHolidayFeatures:
    def test_thanksgiving_indicators_2024(self):
        dates = pd.date_range("2024-11-26", "2024-11-30", freq="D")
        feat = build_holiday_features(pd.Series(dates))

        # Nov 27 (Wed) is day before Thanksgiving
        wed = feat.iloc[1]
        assert wed["is_holiday"] == 0
        assert wed["is_day_before_holiday"] == 1
        assert wed["is_day_after_holiday"] == 0

        # Nov 28 (Thu) is Thanksgiving
        thu = feat.iloc[2]
        assert thu["is_holiday"] == 1
        assert thu["is_day_before_holiday"] == 0
        assert thu["is_day_after_holiday"] == 0

        # Nov 29 (Fri - Black Friday) is day after and bridge day
        fri = feat.iloc[3]
        assert fri["is_holiday"] == 0
        assert fri["is_day_after_holiday"] == 1
        assert fri["is_bridge_day"] == 1

    def test_memorial_day_bridge_friday(self):
        # Memorial Day 2024 is Monday May 27
        # Friday May 24 should be a bridge day
        dates = pd.Series([pd.Timestamp("2024-05-24"), pd.Timestamp("2024-05-27")])
        feat = build_holiday_features(dates)

        assert feat.iloc[0]["is_bridge_day"] == 1
        assert feat.iloc[1]["is_holiday"] == 1

    def test_empty_dates(self):
        feat = build_holiday_features(pd.Series([], dtype="datetime64[ns]"))
        assert feat.empty
        assert list(feat.columns) == [
            "is_holiday",
            "is_day_before_holiday",
            "is_day_after_holiday",
            "is_bridge_day",
        ]
