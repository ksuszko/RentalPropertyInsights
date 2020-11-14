#import libraries

#for data manipulation
import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from pandas import read_csv
from pandas import datetime
import collections

#for visualization
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates
import seaborn as sns
from prettytable import PrettyTable
from IPython.display import HTML, display

#statistics
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from pmdarima.model_selection import train_test_split

#other
import requests
from operator import itemgetter
import datetime
import warnings
import datetime


def df_ts(zipcodes,melted):
    #Create individualized time series for each zipcode.
    df_ts = []
    df_ts_2 = []
    for zc in zipcodes:
    #Create separate dataframes for each zipcode with a monthly frequency.
        df = melted[melted['RegionName']==zc].asfreq('MS')
        df['RegionName']= df['RegionName'].astype('Int64')
        df_ts.append(df)
        
    return(df_ts)
        
def timeseries(df_ts):
    #Create individualized time series for each zipcode.
    for i in range(len(df_ts)):
        df_ts[i].MeanValue.plot(label=df_ts[i].RegionName[0],figsize=(15,6))
        plt.legend()
    
def monthly_return(df_ts):

    #Calculate monthly returns in new column 'ret' for each zipcode.
    for zc in range(len(df_ts)):
        df_ts[zc]['ret']=np.nan*len(df_ts[zc])
        for i in range(len(df_ts[zc])-1):
            df_ts[zc]['ret'][i+1]= (df_ts[zc].MeanValue.iloc[i+1] / df_ts[zc].MeanValue.iloc[i]) - 1
        
    #Plot the monthly returns of each zipcode
    for i in range(len(df_ts)):
        df_ts[i].ret.plot(figsize=(11,5), color = 'b')
        plt.title(f'Zipcode: {df_ts[i].RegionName[0]}')
        plt.xlabel('Date')
        plt.ylabel('Returns (%)')
        plt.legend(loc='best')
        plt.show()
        
def rollmean_std(df_ts):

    for i in range(len(df_ts)):
        rolmean = df_ts[i].ret.rolling(window = 12, center = False).mean()
        rolstd = df_ts[i].ret.rolling(window = 12, center = False).std()
        fig = plt.figure(figsize=(11,5))
        orig = plt.plot(df_ts[i].ret, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title(f'Rolling Mean & Standard Deviation for Zipcode: {df_ts[i].RegionName[0]}')
        plt.show()
        
def adfuller_fisrt_diff(df_ts):
    for i in range(len(df_ts)):
        results = adfuller(df_ts[i].ret.diff().dropna())
        print(f'ADFuller test p-value for zipcode: {df_ts[i].RegionName[0]}')
        print('p-value:',results[1])
        if results[1]>0.05:
            print('Fail to reject the null hypothesis. Data is not stationary.\n')
        else:
            print('Reject the null hypothesis. Data is stationary.\n')
        
        
def evaluate_arima_model(X, arima_order):
    """
    To be used with "evaluate_models" function
    Defines test/train as 2/3 of dataset input
    Returns error of train vs. test
    """
    warnings.filterwarnings("ignore")
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    """
    Input a time series dataframe
    Calculates AIC for range of pdq values specified
    Returns pdq combination with lowest error
    """
    warnings.filterwarnings("ignore")
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    #print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    
def ARIMA_model(df_ts, orders):
    warnings.filterwarnings("ignore")
    forecasts_11_20=[]
    forecasts_11_21=[]
    forecasts_11_22=[]
    forecasts_11_23=[]
    forecasts_11_24=[]
    forecasts_11_25=[]
    forecasts_11_26=[]
    forecasts_11_27=[]
    forecasts_11_28=[]
    forecasts_11_29=[]
    forecasts_11_30=[]
    for i in range(len(df_ts)):
        try:
            model=ARIMA(df_ts[i].dropna().MeanValue,order=orders[i])
            model_fit = model.fit(disp=1)
        except:
            pass
        forecast = model_fit.forecast(161)
        actual_foreacst = forecast[0]
        forecast_conf_int = forecast[2]
        # make dataframe with forecast and 95% confidence interval 
        df_forecast = pd.DataFrame({'time': pd.date_range(start = '2017-07-01', end = '2030-11-01', freq = 'MS')})
        df_forecast['forecast'] = actual_foreacst
        df_forecast['lower_bound'] = forecast_conf_int[:, 0]
        df_forecast['upper_bound'] = forecast_conf_int[:, 1]
        df_forecast.set_index('time', inplace = True)
        # combine raw data dataframe and forecast dataframe
        df_new = pd.concat([df_ts[i].MeanValue, df_forecast])
        #figure
        fig = plt.figure(figsize = (12, 8))

        plt.plot(df_new[0], label = 'raw data')
        plt.plot(df_new['forecast'], label = 'forecast')
        plt.fill_between(df_new.index, df_new['lower_bound'], df_new['upper_bound'], color="k", alpha=.15,
                label = 'confidence interval')
        plt.legend(loc = 'upper left')
        title=('Forecast for {}').format(df_ts[i].RegionName[0])
        plt.title(title)
        # forecasted price today
        forcast_today = df_new.loc['2020-11-01', 'forecast']
        forcast_lower = df_new.loc['2020-11-01', 'lower_bound']
        forcast_upper = df_new.loc['2020-11-01', 'upper_bound']
        print('The forecasted price today for {} is'.format(df_ts[i].RegionName[0]),forcast_today)
        #last known value
        last_price = df_ts[i].MeanValue.loc['2017-06-01']
        print('The last documented value for {} was'.format(df_ts[i].RegionName[0]),last_price)
        
def ARIMA_forecasts(df_ts, orders):
    warnings.filterwarnings("ignore")
    dfs=[]
    for i in range(len(df_ts)):
        try:
            model=ARIMA(df_ts[i].dropna().MeanValue,order=orders[i])
            model_fit = model.fit(disp=1)
        except:
            pass
        forecast = model_fit.forecast(161)
        actual_foreacst = forecast[0]
        forecast_conf_int = forecast[2]
        # make dataframe with forecast and 95% confidence interval 
        df_forecast = pd.DataFrame({'time': pd.date_range(start = '2017-07-01', end = '2030-11-01', freq = 'MS')})
        df_forecast['forecast'] = actual_foreacst
        df_forecast['lower_bound'] = forecast_conf_int[:, 0]
        df_forecast['upper_bound'] = forecast_conf_int[:, 1]
        df_forecast.set_index('time', inplace = True)
        # combine raw data dataframe and forecast dataframe
        df_new = pd.concat([df_ts[i].MeanValue, df_forecast])
        dfs.append(df_new)
    return(dfs)


def forecasts_cleaned(df_forecasts):
    dfs=[]
    for i in range(0,len(df_forecasts)):
        columns= df_forecasts[i].transpose().columns
        columns_2=[date_obj.strftime('%Y-%m') for date_obj in columns]
        values= df_forecasts[i].transpose().values[1]
        df_newest= pd.DataFrame(values).transpose()
        df_newest.columns=columns_2
        df_newest=df_newest.dropna(axis=1)
        dfs.append(df_newest)
    df_combined=pd.concat(dfs)
    AR_2017 = [col for col in df_combined.columns if '17' in col]
    AR_2018 = [col for col in df_combined.columns if '18' in col]
    AR_2019 = [col for col in df_combined.columns if '19' in col]
    AR_2020 = [col for col in df_combined.columns if '20' in col]
    AR_2021 = [col for col in df_combined.columns if '21' in col]
    AR_2022 = [col for col in df_combined.columns if '22' in col]
    AR_2023 = [col for col in df_combined.columns if '23' in col]
    AR_2024 = [col for col in df_combined.columns if '24' in col]
    AR_2025 = [col for col in df_combined.columns if '25' in col]
    AR_2026 = [col for col in df_combined.columns if '26' in col]
    AR_2027 = [col for col in df_combined.columns if '27' in col]
    AR_2028 = [col for col in df_combined.columns if '28' in col]
    AR_2029 = [col for col in df_combined.columns if '29' in col]
    AR_2030 = [col for col in df_combined.columns if '30' in col]

    df_combined['AR_2017']=df_combined[AR_2017].median(axis=1)
    df_combined['AR_2018']=df_combined[AR_2018].median(axis=1)
    df_combined['AR_2019']=df_combined[AR_2019].median(axis=1)
    df_combined['AR_2020']=df_combined[AR_2020].median(axis=1)
    df_combined['AR_2021']=df_combined[AR_2021].median(axis=1)
    df_combined['AR_2022']=df_combined[AR_2022].median(axis=1)
    df_combined['AR_2023']=df_combined[AR_2023].median(axis=1)
    df_combined['AR_2024']=df_combined[AR_2024].median(axis=1)
    df_combined['AR_2025']=df_combined[AR_2025].median(axis=1)
    df_combined['AR_2026']=df_combined[AR_2026].median(axis=1)
    df_combined['AR_2027']=df_combined[AR_2027].median(axis=1)
    df_combined['AR_2028']=df_combined[AR_2028].median(axis=1)
    df_combined['AR_2029']=df_combined[AR_2029].median(axis=1)
    df_combined['AR_2030']=df_combined[AR_2030].median(axis=1)
    return(df_combined)
    
    
def CAGR(df):
    df['2017_Median']= df[['2017-01','2017-02','2017-03','2017-04','2017-05']].median(axis=1)
    df['2014_Median']= df[['2014-01','2014-02','2014-03','2014-04','2014-05','2014-06','2014-07','2014-08','2014-09','2014-10','2014-11','2014-12']].median(axis=1)

    df['CAGR_3']=(((df['2017_Median']/df['2014_Median'])**(1/3))-1)*100

    df['2020_CAGR3']= df['2017_Median']*(1+(df['CAGR_3'])/100)**3
    df['2025_CAGR3']= df['2017_Median']*(1+(df['CAGR_3'])/100)**8
    df['2030_CAGR3']= df['2017_Median']*(1+(df['CAGR_3'])/100)**13
    
    return(df)

def clean_combined(df_combined):
    lst=(df_combined['AR_2020'].iloc[0:9].to_list(),float(df_combined['2020_CAGR3'].iloc[9]),df_combined['AR_2020'].iloc[10:13].to_list(),
    df_combined['2020_CAGR3'].iloc[13:15].to_list(),df_combined['AR_2020'].iloc[15:17].to_list(),float(df_combined['2020_CAGR3'].iloc[17]),
    df_combined['AR_2020'].iloc[18:20].to_list(),df_combined['2020_CAGR3'].iloc[20:25].to_list())
    def flatten(x):
        if isinstance(x, collections.Iterable):
            return [a for i in x for a in flatten(i)]
        else:
            return [x]
    new_list=flatten(lst)
    df_combined['Best_2020']=new_list
    return(df_combined)
    
       
    