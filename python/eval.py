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

forecast_filename = "official_forecast.xlsx"
forecast_sheetname = "forecast_by_day"

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




def plot_error_distribution(df, colname, barcolor):
    x_list = list(range(-60,61,10))
    x_ticks = [x/100 for x in x_list]
    x_labels = [str(x)+"%" for x in x_list]
    x_labels[0] = "<-50%"
    x_labels[-1] = ">50%"
    
    sns.histplot(data = df, x = colname + "_forecast_error_pct_plot", binwidth=0.1, color = barcolor)
    plt.title(colname.title() + " Forecast Error Distribution")
    plt.xlabel("Error Percentage (%)")
    plt.ylabel("Store / Day Occurences")
    #plt.xticks(plt.xticks()[0], [int(x * 100) for x in plt.xticks()[0]])
    plt.xticks(x_ticks, x_labels)
    
    
def accuracy_by_store(df, colname):
    
    pred_name = colname+"_forecast_orders"
    mae_name = colname+"_mae"
    
    return_df = df \
        .groupby("store_no") \
        .agg({"actual_orders": lambda x: metrics.mean_absolute_error(y_true = x, y_pred = df_results.loc[x.index, pred_name])}) \
        .reset_index() \
        .rename(columns = {"actual_orders" : mae_name})
        
    return return_df

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
forecast_import = pd.read_excel(data_path / forecast_filename, sheet_name = forecast_sheetname)
df_forecast = forecast_import.rename(columns = {"ds": "pickup_dt", "store": "store_no", "yhat": "auto_forecast_orders"})
df_forecast = df_forecast.iloc[:,1:].astype({"store_no": "str"})
df_forecast["pickup_dt"] = pd.to_datetime(df_forecast["pickup_dt"])



### JOIN ALL 3
df_results = df_results.merge(df_forecast, on = ["pickup_dt", "store_no"])
df_results["auto_forecast_orders"] = np.where(df_results["auto_forecast_orders"] < 0, 0, round(df_results["auto_forecast_orders"]))
df_results["auto_forecast_orders"] = df_results["auto_forecast_orders"].astype(int)
df_results = df_results.merge(df_data.drop("division_no", axis=1), on = ["pickup_dt", "store_no"], how = "left")



#%% EVALUATION

### % Accuracy Metrics

# add % error and absolute % error
df_results["manual_forecast_error_pct"] = (df_results["manual_forecast_orders"] - df_results["actual_orders"]) / df_results["actual_orders"]
df_results["auto_forecast_error_pct"] = (df_results["auto_forecast_orders"] - df_results["actual_orders"]) / df_results["actual_orders"]

df_results["manual_absolute_error_pct"] = abs(df_results["manual_forecast_error_pct"])
df_results["auto_absolute_error_pct"] = abs(df_results["auto_forecast_error_pct"])

# filter 0 order days (all cancelled and unable to calculate % metrics for these days)
df_results = df_results[df_results["actual_orders"] > 0] # removes store 925 on 12/23/22 and store 344 on 12/24/22 - all orders cancelled


df_results["manual_absolute_error_pct"].mean()
df_results["auto_absolute_error_pct"].mean()


# ACCURACY METRICS
manual_metrics = calc_metrics(ytrue = df_results["actual_orders"], 
                              ypred = df_results["manual_forecast_orders"])

print_metrics(manual_metrics, "\nMANUAL METRICS")

auto_metrics = calc_metrics(ytrue = df_results["actual_orders"], 
                              ypred = df_results["auto_forecast_orders"])

print_metrics(auto_metrics, "\nAUTO METRICS")



# # RESIDUAL DISTRIBUTION
# df_results["manual_error_bin"] = df_results["manual_forecast_error"].apply(lambda x: create_value_bins(x, 10)["bin_number"])
# sns.histplot(data = df_results, x = "manual_error_bin", binwidth=1)

# df_results["auto_error_bin"] = df_results["auto_forecast_error"].apply(lambda x: create_value_bins(x, 10)["bin_number"])
# sns.histplot(data = df_results, x = "auto_error_bin", binwidth=1)


df_results["manual_forecast_error_pct_plot"] = np.where(df_results["manual_forecast_error_pct"] > 0.5, 0.6, df_results["manual_forecast_error_pct"])
df_results["manual_forecast_error_pct_plot"] = np.where(df_results["manual_forecast_error_pct_plot"] < -0.5, -0.6, df_results["manual_forecast_error_pct_plot"])

df_results["auto_forecast_error_pct_plot"] = np.where(df_results["auto_forecast_error_pct"] > 0.5, 0.6, df_results["auto_forecast_error_pct"])
df_results["auto_forecast_error_pct_plot"] = np.where(df_results["auto_forecast_error_pct_plot"] < -0.5, -0.6, df_results["auto_forecast_error_pct_plot"])


#TODO: bucket anything larger than x%
plot_error_distribution(df_results, "manual", barcolor = "blue")
plot_error_distribution(df_results, "auto", barcolor = "orange")
plt.title("Auto (Orange) vs Manual (Blue) Error Distribution")
plt.savefig('exports/error_distribution.png', dpi=300)


# STORE SEGMENTATION
df_manual_acc = accuracy_by_store(df_results, "manual")
df_auto_acc = accuracy_by_store(df_results, "auto")


df_orders_agg = df_results.groupby("store_no").agg({"actual_orders": ["sum", "mean"],
                                                    "manual_forecast_orders": ["sum", "mean"],
                                                    "auto_forecast_orders": ["sum", "mean"]}).reset_index()
    

df_orders_agg.columns = ["store_no"] + [f'{col}_{agg}' for col, agg in df_orders_agg.columns][1:]


df_store_agg = df_orders_agg.merge(df_manual_acc, on = "store_no").merge(df_auto_acc, on = "store_no")

sns.scatterplot(df_store_agg, x = "actual_orders_sum", y = "manual_mae")
sns.scatterplot(df_store_agg, x = "actual_orders_sum", y = "auto_mae")


df_store_acc_agg = df_results.groupby("store_no").agg({"manual_absolute_error_pct": "mean", 
                                                       "auto_absolute_error_pct": "mean",
                                                       "actual_orders": "mean"}).reset_index()

sns.regplot(df_store_acc_agg, x = "actual_orders", y = "manual_absolute_error_pct")
sns.regplot(df_store_acc_agg, x = "actual_orders", y = "auto_absolute_error_pct")
plt.title("Auto (Orange) vs Manual (Blue) By Store Average Error")
plt.xlabel("Daily Order Volume")
plt.ylabel("Absolute Error Percentage")
plt.savefig('exports/store_errors.png', dpi=300)



# by week chart
df_results["week"] = df_results['pickup_dt'].dt.strftime('%U')

df_date_results = df_results.groupby("week").agg({"manual_forecast_error_pct": "mean",
                                                       "auto_forecast_error_pct":"mean"})

sns.lineplot(df_date_results, x="week", y="manual_forecast_error_pct", label="Kroger Error Percent")
sns.lineplot(df_date_results, x="week", y="auto_forecast_error_pct", label="AMEND Error Percent").set(title='Weekly Error Percent')
plt.xlabel('Week')
plt.ylabel('Error Percent')
plt.axhline(y=0, color = 'red')
plt.legend()


# IMPORTANT FACTORS
# FORECAST HIGHLIGHTS & LOWLIGHTS COMPARED TO MANUAL



