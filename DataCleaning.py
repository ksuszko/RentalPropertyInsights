# Importing required libraries

#for data manipulation
import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from pandas import read_csv
from pandas import datetime

#for visualization
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates
import seaborn as sns
from prettytable import PrettyTable
from IPython.display import HTML, display

#for machine learning
from statsmodels.tsa.seasonal import seasonal_decompose
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

#statistics
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
#other
import requests
from operator import itemgetter
import datetime
import warnings


def strip_price(df,columns):
    for column in columns:
        df[column] = df[column].replace(to_replace={'\$':'',',':''}, regex = True).astype(float)
    return(df)

def equiv_converstion(df):
    warnings.filterwarnings("ignore")
    df.loc[:,'price_equiv'] = df['price']
    df.loc[:,'price_equiv'][df['room_type']=='Private room'] = df['price']*2
    return(df)

def zip_nulls(df):
    nulls=[]
    zip_codes=[]
    for column in df: #iterate through columns
        nulls.append(round((df[column].isna().sum()/len(df))*100)) #append percent null
        zip_codes.append(column) #append column name
        
    dict_nulls={'Zip': zip_codes, 'Nulls': nulls} #create dict
    zips=pd.DataFrame(data=dict_nulls) #create df
    zips=zips.sort_values(by='Nulls',ascending=False) #sort df    
    return(zips)

def null_percent_fig(zips,nulls):
    #display percent nulls with a visualization 
    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'

    fig = go.Figure(data=[go.Table(
      header=dict(
        values=['<b>Zip Code</b>','<b>% Null</b>'],
        line_color='darkslategray',
        fill_color=headerColor,
        align=['left','center'],
        font=dict(color='white', size=12)
      ),
      cells=dict(
        values=[ zips, nulls],
        line_color='darkslategray',
        # 2-D list of colors for alternating rows
        fill_color = [[rowOddColor,rowEvenColor]*(round(len(zips)/2))],
        align = ['left', 'center'],
        font = dict(color = 'darkslategray', size = 11)
        ))
    ])

    fig.show()
    
def roi(df):
    ROI=[]
    region=[]
    df=df.drop('Date',axis=1) #drop the date column
    for column in df: #iterate over all but last column
        first=int(df[column].iloc[df[column].first_valid_index()])#get first non-null item
        last=int(df[column].iloc[-1]) #get last item-- we know will not be null based on observation of data
        ROI.append((last/first)-1) #append calculated ROI
        region.append(int(df[column].name)) #append region name
    #dict of values   
    dict_region= {'RegionName': region, 'ROI': ROI}
    return(dict_region)   

def melt_data(df):
    melted = pd.melt(df, id_vars=['RegionID','RegionName', 'City', 'State', 'Metro',
                                'CountyName', 'SizeRank'], var_name='Month', value_name='MeanValue')
    melted['Month'] = pd.to_datetime(melted['Month'], infer_datetime_format=True)
    melted = melted.dropna(subset=['MeanValue'])
    return melted

def sentiment_analysis(df_2bed):
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 
    df_2bed['summary'].fillna('NA',inplace=True)
            
    sums=[]
    sent=[]
    for i in range(0,len(df_2bed)):
        sums.append(df_2bed['summary'].iloc[i])

    for summary in sums:
        sentiment_dict = sid_obj.polarity_scores(summary) 
        sent.append(sentiment_dict['compound'])
    df_2bed['sentiment']=sent
    return(df_2bed)


