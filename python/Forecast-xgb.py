
export_name = "forecast_startdates_xgb_tune.csv"

# %%import packages
import pandas as pd
import numpy as np
#from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import os
#import inspect
import datetime
from statistics import mean
#import xgboost as xgb
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# %%
# # Find script location
# script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
# script_dir = script_dir.replace("\\", "/")

# # Locate parent directory
# parent_dir =  os.path.abspath(os.path.join(script_dir, os.pardir))
# parent_dir = parent_dir.replace("\\", "/")

# # Define file path for input data
# file_path = parent_dir + "/data/"
# output_path = file_path + '/output/'


os.chdir("C:/Users/BrandonLester/Desktop/Kroger-Pickup-Forecast")
cur_dir = Path.cwd()
data_path = cur_dir / "data"

data_filename = "Cincinnati Division Forecast Pilot Data.xlsx"
data_sheetname = "Forecast Data"

def mape_objective(y_true, y_pred):
    gradient = np.sign(y_true - y_pred) / (np.abs(y_true) + 1e-10)
    hessian = 1.0 / (np.abs(y_true) + 1e-10)
    return gradient, hessian


# %% set dates

# %% read data
#file_name = file_path + 'Cincinnati Division Forecast Pilot Data.xlsx'

data = pd.read_excel(data_path / data_filename, sheet_name=data_sheetname)

df = pd.DataFrame(data)
df = df.rename(columns= {'PICKUP_DT':'ds', 'Orders Packed':'y'})

# %% drop closed stores
df = df.drop(df[df['STORE_NO'].isin({305, 922, 400, 789, 430, 800, 506, 504})].index)


# %% add holiday flags
# List of holidays
# prethanksgiving = pd.DataFrame({
#   'holiday': 'prethanksgiving',
#   'ds': pd.to_datetime(['2020-11-21', '2021-11-23', '2022-11-22']),
#   'lower_window': -1,
#   'upper_window': 0,
# })
# thankseve = pd.DataFrame({
#   'holiday': 'thankseve',
#   'ds': pd.to_datetime(['2020-11-22', '2021-11-24', '2022-11-23']),
#   'lower_window': 0,
#   'upper_window': 0,
# })
# prechristmas = pd.DataFrame({
#   'holiday': 'prechristmas',
#   'ds': pd.to_datetime(['2020-12-23', '2021-12-23', '2022-12-23']),
#   'lower_window': -1,
#   'upper_window': 0,
# })
# christeve = pd.DataFrame({
#   'holiday': 'christeve',
#   'ds': pd.to_datetime(['2020-12-24', '2021-12-24', '2022-12-24']),
#   'lower_window': 0,
#   'upper_window': 0,
# })
# holidays = pd.concat((prethanksgiving, prechristmas, thankseve, christeve))

from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime, timedelta, date
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


df = df.merge(df_yearend_holidays, left_on = "ds", right_on = "predate", how = "left")
df = df.merge(df_holidays_wide, left_on = "ds", right_on = "date", how = "left")
df = df.fillna(0)
df = df.drop(["predate", "date"], axis = 1)

df["DayOfWeek"] = [x.weekday() for x in df["ds"]]
df["MonthOfYear"] = [x.month for x in df["ds"]]

# %%create forecast for each store
startdate_final = pd.read_csv(data_path /'prophet_startdate.csv')
Store_Startdate = startdate_final.groupby(['StartDate', 'Store'], as_index=False)['MAE'].mean()
Best_Store_Startdate = Store_Startdate.set_index('StartDate').groupby('Store', as_index=False).idxmin()

# Initialize an empty dictionary to store MAE for each store
full_forecast = pd.DataFrame(columns = ['ds', 'store', 'yhat'])

global_forecast_start = '2022-11-06'

# Perform forecasting for each store separately
for store_id in df[df['ds'] == '2022-10-08']['STORE_NO'].unique():
    
    print(store_id)
    
    full_store_data = df[df['STORE_NO'] == store_id].copy()
    
    startdate = Best_Store_Startdate[Best_Store_Startdate['Store'] == store_id]['MAE'].values[0]
    
    store_data = full_store_data[full_store_data['ds'] >= startdate].reset_index(drop=True)
    

    ### SETUP MODEL
    store_train_data = store_data[store_data["ds"] < global_forecast_start]
    store_test_data = store_data[store_data["ds"] >= global_forecast_start]
    
    X_train = store_train_data.iloc[:,6:]
    y_train = store_train_data.iloc[:,5]
    
    X_test = store_test_data.iloc[:,6:]
    
    
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    
    # Initialize the XGBoost Regressor
    xgb_model = XGBRegressor(objective='reg:absoluteerror', eval_metric = "mape", random_state=42)
    #xgb_model = XGBRegressor(objective=mape_objective, random_state=42)
            
    
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_absolute_percentage_error',
        #cv=3,  # Number of cross-validation folds
        #n_jobs=-1  # Use all available CPU cores
    )   
    
    
    ### TRAIN THE MODEL
    #xgb_model.fit(X_train, y_train)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    
    ### FORECAST   
    #y_pred = xgb_model.predict(X_test)
    y_pred = best_model.predict(X_test)

    ### SAVE RESULTS    
    store_forecast = store_test_data[["ds"]]
    store_forecast["yhat"] = y_pred
    store_forecast["store"] = store_id
    

    full_forecast = pd.concat([full_forecast, store_forecast])
        


# %%

full_forecast.to_csv(data_path / export_name)

