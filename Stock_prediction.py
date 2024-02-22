

import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from datetime import date, timedelta
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import requests
# Specify the file path to your favicon
favicon_path = './favicon.ico'

# Set the page configuration including the favicon
st.set_page_config(
    page_title='Stock Prediction',
    page_icon=favicon_path,
    
    initial_sidebar_state='auto'
)
# Today Date Labaleing

today = date.today()

# start_date=d2
# end_date=d1




# company_name=''
st.title('Stock Market Trend Prediction using LSTM')
user_ip=st.text_input('Enter Company Name', 'GOOG')
start_date=st.text_input('Enter Start Date (DD-MM-YYYY)')

submit=st.button('Submit')
if submit:
        
    # ticker=get_ticker(user_ip.upper())
    # print(ticker)
    msft = yf.Ticker(user_ip)
    company_name = msft.info['longName']

    st.subheader('Stock Trend Prediction of '+ company_name)


    # ticker = get_ticker(user_ip)
    # ticker
    end_date=today
    ticker_data1 = yf.Ticker(user_ip)
    d1= end_date.strftime("%Y-%m-%d")
    d2 = (end_date - timedelta(days=365)).strftime("%Y-%m-%d")
    start_date=d2
    end_date=d1

    df1=ticker_data1.history(period='1d', start=start_date, end=end_date)


    #Describing data
    st.subheader(f'From {start_date} to {end_date}')
    st.write(df1.describe())

    st.subheader('Closing Price vs Time chart')
    fig=plt.figure(figsize=(12,6))
    plt.plot(df1.Close)
    st.pyplot(fig)


    # st.subheader('Closing Price vs Time chart with 100MA')
    # ma100=df1.Close.rolling(100).mean()

    # fig=plt.figure(figsize=(12,6))
    # plt.plot(ma100,'r')
    # plt.plot(df1.Close)
    # st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA and 200MA ')
    ma100=df1.Close.rolling(100).mean()
    ma200=df1.Close.rolling(200).mean()

    fig=plt.figure(figsize=(12,6))
    plt.plot(ma100,'r',label='100 Days Moving Average')
    plt.plot(ma200,'b',label='200 Days Moving Average')
    plt.plot(df1.Close,'g',label='Closing Price')
    plt.legend()
    plt.show()
    st.pyplot(fig)

    #Spliting data into training and testing
    d_train=pd.DataFrame(df1['Close'][0:int(len(df1)*0.70)])
    d_testing=pd.DataFrame(df1['Close'][int(len(df1)*0.70):int(len(df1))])


    scaler=MinMaxScaler(feature_range=(0,1))

    d_train_array=scaler.fit_transform(d_train)


    #Training Done in Model

    #Load model
    model=load_model('keras_model.h5')

    #test  Data
    paast_100_days=d_train.tail(100)
    final_df = pd.concat([paast_100_days, d_testing], ignore_index=True)
    input_data=scaler.fit_transform(final_df)
    x_test=[]
    y_test=[]

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])
    x_test,y_test=np.array(x_test),np.array(y_test)
    y_predicted=model.predict(x_test)

    scale_fact=scale_fact = 1 / scaler.scale_[0]
    y_predicted=y_predicted*scale_fact
    y_test=y_test*scale_fact


    #Graph plot



    st.subheader('Prediction vs Original of '+ company_name)
    fig2=plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='original price')
    plt.plot(y_predicted, 'r', label='predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.pyplot(fig2)