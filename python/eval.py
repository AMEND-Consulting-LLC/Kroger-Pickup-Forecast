# -*- coding: utf-8 -*-
"""

Forecast model evaluation

"""

#%% SETUP
import os
from pathlib import Path
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("C:/Users/BrandonLester/Desktop/Kroger-Pickup-Forecast")
cur_dir = Path.cwd()
data_path = cur_dir / "data"

data_filename = "Cincinnati Division Forecast Pilot Data.xlsx"
data_sheetname = "Forecast Data"

results_filename = "Doug File.xlsx"
results_sheetname = "Export"

forecast_filename = "prophet_full_forecast_4.2.20start.csv"

#%% FUNCTIONS
def add_fake_forecasts(df_input, error_size, colname):
    df = df_input.copy()
    error_pct_name = colname + "_input_forecast_error_pct"
    order_name = colname + "_forecast_orders"
    error_name = colname + "_forecast_error"
    
    
    df[error_pct_name] = np.random.randint(-error_size, error_size, size=len(df.index))
    df[error_pct_name] = 1+df[error_pct_name]/100
    
    df[order_name] = df["actual_orders"] * df[error_pct_name]
    df[order_name] = df[order_name].apply(lambda x: int(np.ceil(x)))
    
    df[error_name] = df[order_name] - df["actual_orders"]
    df[error_name+"_pct"] = df[error_name] / df["actual_orders"]
    
    return(df)

def calc_metrics(ytrue, ypred):
    r2 = metrics.r2_score(ytrue, ypred)
    mae = metrics.mean_absolute_error(ytrue, ypred)
    mape = metrics.mean_absolute_percentage_error(ytrue, ypred)
    mse = metrics.mean_squared_error(ytrue, ypred)
    rmse = np.sqrt(mse)
    
    dict_metrics = {"r2": r2,
                    "mae": mae,
                    "mape": mape,
                    "mse": mse,
                    "rmse": rmse}
    
    return dict_metrics

def print_metrics(dict_metrics, title_string):
    print(title_string)
    for met in dict_metrics.keys():
        print("{} = {}".format(met.upper(), np.round(dict_metrics[met],2)))


def create_value_bins(input_value, bin_size):

    # Calculate the bin number
    bin_number = (input_value // bin_size) + 1

    # Calculate the bin range
    bin_start = (bin_number - 1) * bin_size
    bin_end = bin_start + bin_size

    # Return the bin information
    return {
        "input_value": input_value,
        "bin_number": bin_number,
        "bin_range": (bin_start, bin_end)
    }


def plot_error_distribution(df, colname):
    sns.histplot(data = df, x = colname + "_forecast_error_pct", binwidth=0.1)
    plt.title(colname.title() + " Forecast Error Distribution")
    plt.xlabel("Error Percentage (%)")
    plt.ylabel("Store / Day Occurences")
    plt.xticks(plt.xticks()[0], [int(x * 100) for x in plt.xticks()[0]])

#%% DATA PREP


### TRAINING DATA WITH FORECAST PERIOD
data_import = pd.read_excel(data_path / data_filename, sheet_name = data_sheetname)

#df_data = data_import[data_import["PICKUP_DT"] >= "10-1-2022"].reset_index(drop=True)
#df_data = df_data.rename(columns = {"All_Orders" : "ACTUAL_ORDERS"})
#df_data = df_data[["PICKUP_DT", "DIVISION_NO", "STORE_NO", "ACTUAL_ORDERS"]]
df_data = data_import.set_axis([x.lower() for x in data_import.columns], axis = 1)
df_data = df_data.rename(columns = {"orders packed" : "orders_packed"})
df_data["division_no"] = df_data["division_no"].astype(str)
df_data["store_no"] = df_data["store_no"].astype(str)

# df_data = add_fake_forecasts(df_data, 30, "manual")
# df_data = add_fake_forecasts(df_data, 15, "auto")


### FORECAST PERIOD ACTUAL AND KROGER FORECAST
results_import = pd.read_excel(data_path / results_filename, sheet_name = results_sheetname)

df_results = results_import.rename(columns = {"DATE": "pickup_dt", 
                                              "Kroger Published Forecast": "manual_forecast_orders",
                                              "Actual Orders": "actual_orders"})

df_results = df_results[~df_results["Key"].isnull()]
df_results = df_results[~df_results["actual_orders"].isnull()]

df_results["division_no"] = [x.split("_")[0][1:] for x in df_results["Key"]]
df_results["store_no"] = [x.split("_")[1][2:] for x in df_results["Key"]]

df_results = df_results[["pickup_dt", "division_no", "store_no", "actual_orders", "manual_forecast_orders"]]
df_results["actual_orders"] = df_results["actual_orders"].astype(int)
df_results["manual_forecast_orders"] = df_results["manual_forecast_orders"].astype(int)
df_results["pickup_dt"] = pd.to_datetime(df_results["pickup_dt"])


### AMEND FORECASTED VALUES
forecast_import = pd.read_csv(data_path / forecast_filename)
df_forecast = forecast_import.rename(columns = {"ds": "pickup_dt", "store": "store_no", "yhat": "auto_forecast_orders"})
df_forecast = df_forecast.iloc[:,1:].astype({"store_no": "str"})
df_forecast["pickup_dt"] = pd.to_datetime(df_forecast["pickup_dt"])



### JOIN ALL 3
df_results = df_results.merge(df_forecast, on = ["pickup_dt", "store_no"])
df_results["auto_forecast_orders"] = np.where(df_results["auto_forecast_orders"] < 0, 0, round(df_results["auto_forecast_orders"]))
df_results["auto_forecast_orders"] = df_results["auto_forecast_orders"].astype(int)
df_results = df_results.merge(df_data.drop("division_no", axis=1), on = ["pickup_dt", "store_no"], how = "left")



df_results["manual_forecast_error_pct"] = (df_results["manual_forecast_orders"] - df_results["actual_orders"]) / df_results["actual_orders"]
df_results["auto_forecast_error_pct"] = (df_results["auto_forecast_orders"] - df_results["actual_orders"]) / df_results["actual_orders"]

#%% EVALUATION

# ACCURACY METRICS
manual_metrics = calc_metrics(ytrue = df_results["actual_orders"], 
                              ypred = df_results["manual_forecast_orders"])

print_metrics(manual_metrics, "\nMANUAL METRICS")

auto_metrics = calc_metrics(ytrue = df_results["actual_orders"], 
                              ypred = df_results["auto_forecast_orders"])

print_metrics(auto_metrics, "\nAUTO METRICS")


# RESIDUAL DISTRIBUTION
df_results["manual_error_bin"] = df_results["manual_forecast_error"].apply(lambda x: create_value_bins(x, 10)["bin_number"])
sns.histplot(data = df_results, x = "manual_error_bin", binwidth=1)

df_results["auto_error_bin"] = df_results["auto_forecast_error"].apply(lambda x: create_value_bins(x, 10)["bin_number"])
sns.histplot(data = df_results, x = "auto_error_bin", binwidth=1)

#TODO: bucket anything larger than x%
plot_error_distribution(df_results, "manual")
plot_error_distribution(df_results, "auto")



# STORE SEGMENTATION

def accuracy_by_store(df, colname):
    
    pred_name = colname+"_forecast_orders"
    mae_name = colname+"_mae"
    
    return_df = df \
        .groupby("store_no") \
        .agg({"actual_orders": lambda x: metrics.mean_absolute_error(y_true = x, y_pred = df_results.loc[x.index, pred_name])}) \
        .reset_index() \
        .rename(columns = {"actual_orders" : mae_name})
        
    return return_df

df_manual_acc = accuracy_by_store(df_results, "manual")
df_auto_acc = accuracy_by_store(df_results, "auto")


df_orders_agg = df_results.groupby("store_no").agg({"actual_orders": ["sum", "mean"],
                                                    "manual_forecast_orders": ["sum", "mean"],
                                                    "auto_forecast_orders": ["sum", "mean"]}).reset_index()
    

df_orders_agg.columns = ["store_no"] + [f'{col}_{agg}' for col, agg in df_orders_agg.columns][1:]


df_store_agg = df_orders_agg.merge(df_manual_acc, on = "store_no").merge(df_auto_acc, on = "store_no")

sns.scatterplot(df_store_agg, x = "actual_orders_sum", y = "manual_mae")
sns.scatterplot(df_store_agg, x = "actual_orders_sum", y = "auto_mae")


# IMPORTANT FACTORS
# FORECAST HIGHLIGHTS & LOWLIGHTS COMPARED TO MANUAL



