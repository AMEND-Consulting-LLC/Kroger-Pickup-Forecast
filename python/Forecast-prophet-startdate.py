# %%import packages
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import os
import inspect
import datetime
from statistics import mean

# %%
# Find script location
script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
script_dir = script_dir.replace("\\", "/")

# Locate parent directory
parent_dir =  os.path.abspath(os.path.join(script_dir, os.pardir))
parent_dir = parent_dir.replace("\\", "/")

# Define file path for input data
file_path = parent_dir + "/data/"
output_path = file_path + '/output/'

# %% set dates

# %% read data
file_name = file_path + 'Cincinnati Division Forecast Pilot Data.xlsx'
data = pd.read_excel(file_name, sheet_name='Forecast Data')
df = pd.DataFrame(data)
df = df.rename(columns= {'PICKUP_DT':'ds', 'Orders Packed':'y'})

# %% drop closed stores
df = df.drop(df[df['STORE_NO'].isin({305, 922, 400, 789, 430, 800, 506, 504})].index)
# %% add holiday flags
# List of holidays
prethanksgiving = pd.DataFrame({
  'holiday': 'prethanksgiving',
  'ds': pd.to_datetime(['2020-11-22', '2021-11-24', '2022-11-23']),
  'lower_window': -2,
  'upper_window': 0,
})
prechristmas = pd.DataFrame({
  'holiday': 'prechristmas',
  'ds': pd.to_datetime(['2020-12-24', '2021-12-24', '2022-12-24']),
  'lower_window': -2,
  'upper_window': 0,
})
holidays = pd.concat((prethanksgiving, prechristmas))

# %%create forecast for each store

# Initialize an empty dictionary to store MAE for each store
df_mae = pd.DataFrame(columns = ['Store', 'StartDate', 'Week', 'MAE'])


# Perform forecasting for each store separately
for store_id in df[df['ds'] == '2022-10-08']['STORE_NO'].unique():
    print(store_id)
    mae_per_week = {}
    for startmonth in range(23):
        startdate = pd.to_datetime('2020-02-02') + pd.DateOffset(months=startmonth)
        print(startdate)
        full_store_data = df[df['STORE_NO'] == store_id].copy()
        for week in range(8):
            forecast_start = pd.to_datetime('2022-11-06') +pd.DateOffset(weeks=week)
            forecast_end = pd.to_datetime(forecast_start) + pd.DateOffset(days=6)
            forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='D')
            if pd.to_datetime('2022-12-25') in forecast_dates:
                forecast_dates = forecast_dates.delete(forecast_dates.get_loc(pd.to_datetime('2022-12-25')))
            dates = pd.date_range(start=startdate, end=pd.to_datetime(forecast_start) + pd.DateOffset(days=-29), freq='D')

            store_data = full_store_data[full_store_data['ds'].isin(dates)]

            # Initialize Prophet model
            model = Prophet()
            model = Prophet(holidays=holidays)
            model.add_country_holidays(country_name='US')
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.fit(store_data)

            # Make future dataframe for forecasting (next 28 days)
            future = model.make_future_dataframe(periods=35, freq='D')

            # Make predictions
            forecast = model.predict(future)
            # fig = model.plot_components(forecast)
            # print(forecast)
            # # Calculate MAE
            used_actuals = full_store_data[full_store_data['ds'].isin(forecast_dates)]
            actual_values = used_actuals['y']
            used_forecast = forecast[forecast['ds'].isin(forecast_dates)]
            predicted_values = used_forecast['yhat']
            mae = mean_absolute_error(actual_values, predicted_values)
            # mae_per_week[week] = mae
            df_mae.loc[len(df_mae.index)] = [store_id, startdate, week, mae] 
    # store_mae = sum(mae_per_week.values())/len(mae_per_week)
    print(df_mae)
    # print(store_mae)


# print("Mean Absolute Error (MAE) for Each Store:")
# for store_id, mae in mae_per_store.items():
#     print(f"Store {store_id}: MAE = {mae}")

# %%
df_mae.to_csv(output_path + "prophet_startdate.csv")
# %%
