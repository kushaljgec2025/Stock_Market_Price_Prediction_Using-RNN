

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date, timedelta
import time
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import requests
import total_days


# Specify the file path to your favicon
favicon_path = './favicon.ico'

# Set the page configuration including the favicon
st.set_page_config(
    page_title='Stock Prediction',
    page_icon=favicon_path,
    
    initial_sidebar_state='auto'
)

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color:#04AA6D;
    color:white;
    border:none
}          
</style>""", unsafe_allow_html=True)



# Today Date Labaleing

today = date.today()

# start_date=d2
# end_date=d1




# company_name=''
# st.image('./banner.png')
st.title('Stock Market Trend Prediction using Long Short-Term Memory (LSTM)')
user_ip=st.text_input('Enter Company Ticker Symbol')
col1,col2 = st.columns(2,gap='medium')
start_date=col1.date_input('Enter Start Date (DD-MM-YYYY)',format="DD/MM/YYYY")

d2= start_date.strftime("%d-%m-%Y")
#d2
end_date=col2.date_input('Enter End Date (DD-MM-YYYY)',format="DD/MM/YYYY")

d1= end_date.strftime("%d-%m-%Y")
#d1
# start_date=start_date.replace('-','/')

# end_date=today



submit=st.button('Submit')
if submit:

    #progress bar
    progress_text = "Please wait, we are fetching the data..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()


    # ticker=get_ticker(user_ip.upper())
    # print(ticker)
    tot_days=total_days.calculate_days_between_dates(d2,d1)
    st.subheader("Total days")
    tot_days
    d1= end_date.strftime("%Y-%m-%d")
    d2 = (end_date - timedelta(days=tot_days)).strftime("%Y-%m-%d")
    start_date=d2
    end_date=d1
    msft = yf.Ticker(user_ip)
    
    company_name = msft.info['longName']
    if company_name:
        st.success("Got it")
    
       
    st.subheader('Stock Trend Prediction of '+ company_name)
    

    # ticker = get_ticker(user_ip)
    # ticker
   
    ticker_data1 = yf.Ticker(user_ip)
    
    
    df1=ticker_data1.history(period='1d', start=start_date, end=end_date)


    #Describing data
    st.subheader(f'From {start_date} to {end_date}')
    st.dataframe(df1, width=1000, height=300)
   

    st.subheader('Statistics of Data')
    st.dataframe(df1.describe(), width=1000)

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


    st.success("Predicted Stock")
    st.subheader('Prediction vs Original of '+ company_name)
    fig2=plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='original price')
    plt.plot(y_predicted, 'r', label='predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.pyplot(fig2)