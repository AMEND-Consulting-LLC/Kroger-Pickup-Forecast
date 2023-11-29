# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:23:47 2023

@author: BrandonLester
"""

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime, timedelta, date
import numpy as np
import holidays


def create_yearend_holiday_df(days_before, start_date, end_date):

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date).to_pydatetime()

    df_holidays = pd.DataFrame({"date": holidays})

    df_holidays["year"] = df_holidays["date"].apply(lambda x: x.year)
    df_holidays["month"] = df_holidays["date"].apply(lambda x: x.month)
    df_holidays["day"] = df_holidays["date"].apply(lambda x: x.day)

    df_hss = df_holidays[df_holidays["month"].isin([11,12])]
    df_hss = df_hss[(df_hss["day"]>=20) & (df_hss["day"]<31)]


    df_hss["pre_holiday_start"] = df_hss["date"].apply(lambda x: x - timedelta(days=days_before))

    list_dates = list(df_hss["date"])
    list_starts = list(df_hss["pre_holiday_start"])

    df_predates = pd.DataFrame()

    for i in range(len(list_dates)):
        list_range = pd.date_range(start=list_starts[i], end=list_dates[i]).to_list()
        list_origdate = np.full(len(list_range), list_dates[i])
        df_date = pd.DataFrame({"date": list_origdate, "predate": list_range})
        df_predates = pd.concat([df_predates, df_date])
        
    df_predates["month"] = df_predates["date"].apply(lambda x: x.month)

    df_predates["prethanksgiving"] = np.where((df_predates["month"] == 11) & (df_predates["date"] != df_predates["predate"]), 1, 0)
    df_predates["thanksgiving"] = np.where((df_predates["month"] == 11) & (df_predates["date"] == df_predates["predate"]), 1, 0)

    df_predates["prechristmas"] = np.where((df_predates["month"] == 12) & (df_predates["date"] != df_predates["predate"]), 1, 0)
    df_predates["christmas"] = np.where((df_predates["month"] == 12) & (df_predates["date"] == df_predates["predate"]), 1, 0)

    return df_predates


df_yearend_holidays = create_yearend_holiday_df(3, '2020-01-01', '2022-12-31')
df_yearend_holidays = df_yearend_holidays.drop(["thanksgiving", "christmas", "month", "date"], axis = 1).reset_index(drop=True)


us_holidays = holidays.country_holidays("US")

dates = pd.date_range(start="2020-10-01", end="2022-12-31")
holiday_name = [us_holidays.get(x) for x in dates]

df_holidays = pd.DataFrame({"date": dates, "holiday": holiday_name})
df_holidays["flag"] = np.where(df_holidays["holiday"].isnull(), 0, 1)
df_holidays = df_holidays[df_holidays["flag"] == 1].reset_index(drop=True)

df_holidays_wide = df_holidays.pivot(index="date", columns="holiday", values="flag").fillna(0).reset_index()
