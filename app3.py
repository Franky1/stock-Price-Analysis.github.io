# This Streamlit Web application has been created by ~ Arya Chakraborty [22MSD7020, VIT-AP]
# Contact me here ~ "https://www.linkedin.com/in/arya-chakraborty-95a8411b2/"
#  please visit the official streamlit documentation site to know in details ~ "https://docs.streamlit.io/"

# <=========================Importing all required Libraries ========================>
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from streamlit_option_menu import option_menu
import pandas
import datetime
from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override()
from PIL import Image 

im = Image.open('stock2.jpg')
st.set_page_config(layout="wide",page_title="Stock Price Analysis & Prediction App",page_icon = im)

# <--------------------For the main header ------------------>
html_temp = """ 
<div style ="background-color:black;padding:13px"> 
<h1 style ="color:green;text-align:center;">Stock Price Analysis & Prediction </h1> 
</div> 
"""
#Referance ~ "https://www.analyticsvidhya.com/blog/2020/12/deploying-machine-learning-models-using-streamlit-an-introductory-guide-to-model-deployment/"
st.markdown(html_temp, unsafe_allow_html = True)



# <--------------------------hide the right side streamlit menue button ------------------>
# Referance ~ "https://towardsdatascience.com/5-ways-to-customise-your-streamlit-ui-e914e458a17c"
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# <---------------For Condensing the Layout -------------->
#Referance ~ see above ...
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


# <-------------------sidebar for navigation --------------------->
with st.sidebar:  
    selected = option_menu('OPTIONS',
                          ['Dataset','Analysis',
                           'Prediction','Credits'],
                          icons=["house",'cast','gear','cloud-upload'],menu_icon="cast",
                          default_index=1,styles={
        "container": {"padding": "0!important", "background-color": "#0d0c0c"},
        "icon": {"color": "white", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#288014"},
        "nav-link-selected": {"background-color": "green"},
    }) 
# <-----------------Data Collection from the yahoo finance website using streamlit date_input method------------->
user_input=st.text_input('**:blue[Enter Stock Ticker]**','CIPLA.NS')
start = st.date_input(
    "Tell me the Start Date",
    datetime.date(2017,4,3))
st.write('The start Date is:', start)
end = st.date_input(
    "Tell me the End Date",
    datetime.date(2023,1,20))
st.write('The End Date is:', end)
data1=pdr.get_data_yahoo(user_input,start,end)

# <---------------------------For downloding the C.S.V file of the dataset----------------------->
# Refer to Streamlit Documentation Site
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
csv = convert_df(data1)
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='Y_finance_df.csv',
    mime='text/csv',
) 


import pandas as pd
data=pd.DataFrame(data1["Close"][0:1419]) # here we take first 1419 values 
validation_data=pd.DataFrame(data1['Close'][1419:]) # here we are taking next 7 values of the dataset for validating the next week.

data2=data1.reset_index().copy()
Total_data=data2.iloc[:,4:5].copy() #with this i am going to work
NCP1=Total_data.loc[0:742,:] #NCP1= Non covid period 1(2017-2020)
CP=Total_data.loc[742:988,:] #CP= Covid period(2020-2021)
NCP2=Total_data.loc[990:,:] #NCP=Non Covid period2(2021~present) here present time will vary



if (selected == 'Dataset'):

        
        st.title(':green[Dataset ~]')

        # <==========describing the data ================>

        st.header('Data from 2017 to 2022')
        col1, col2, col3 = st.columns(3)

        with col1:
                st.subheader("First 5 data ~")
                st.table(data.head())

        with col2:
                st.subheader("last 5 data ~")
                st.table(data.tail())

        with col3:
                st.subheader("Description")
                st.table(data.describe())
        
        # <===========modifying the dataset ===========>

        st.markdown('''Here we have considered **:green[(2017.April ~ 2020.March_end)]** & **:green[(2021.April ~ Present)]**
Datasetes as Non-Covid Period Datasets & (2020.April ~ 2021.March end ) Dataset as Covid period Dataset. 
For simplicity purpose here we are considering the "Close" price data as our target variable.''')
        
        # <==============Describing the whole datasets into three different part ================>
        col1, col2, col3 = st.columns(3)

        with col1:
                st.subheader("First Non-Covid Dataset ")
                st.table(NCP1.describe())

        with col2:
                st.subheader("Second Non-Covid Dataset ")
                st.table(NCP2.describe())

        with col3:
                st.subheader("Covid period Dataset ")
                st.table(CP.describe())

        

        # <================ Visualization ===================>

        st.subheader('**:green[Closing Price Vs Time Chart(2017 ~ Pressent)]**')
        fig=plt.figure(figsize=(12,6))
        plt.plot(data.Close)
        st._legacy_line_chart(data['Close'])
        st.subheader("Area Chart for Volume ~")
        st.area_chart(data1['Volume'])


if (selected == 'Analysis'):

        # <============================================================= Statistical Analysis ===============================================>
        st.subheader("Technical Analysis of the data ~")
        
        st.text("Plotting Bollinger Band for the dataset ~")
        df1 = data[['Close']]

        sma = df1.rolling(window=20).mean().dropna()
        rstd = df1.rolling(window=20).std().dropna()

        upper_band = sma + 2 * rstd
        lower_band = sma - 2 * rstd

        upper_band = upper_band.rename(columns={'Close': 'upper'})
        lower_band = lower_band.rename(columns={'Close': 'lower'})
        bb = df1.join(upper_band).join(lower_band)
        bb = bb.dropna()

        buyers = bb[bb['Close'] <= bb['lower']]
        sellers = bb[bb['Close'] >= bb['upper']]

        # <===========Plotting =============>

        import plotly.io as pio
        import plotly.graph_objects as go

        pio.templates.default = "plotly_dark"

        fig11 = go.Figure()
        fig11.add_trace(go.Scatter(x=lower_band.index, 
                                y=lower_band['lower'], 
                                name='Lower Band', 
                                line_color='rgba(173,204,255,0.2)'
                                ))
        fig11.add_trace(go.Scatter(x=upper_band.index, 
                                y=upper_band['upper'], 
                                name='Upper Band', 
                                fill='tonexty', 
                                fillcolor='rgba(173,204,255,0.2)', 
                                line_color='rgba(173,204,255,0.2)'
                                ))
        fig11.add_trace(go.Scatter(x=df1.index, 
                                y=df1['Close'], 
                                name='Close', 
                                line_color='#636EFA'
                                ))
        fig11.add_trace(go.Scatter(x=sma.index, 
                                y=sma['Close'], 
                                name='SMA', 
                                line_color='#FECB52'
                                ))
        fig11.add_trace(go.Scatter(x=buyers.index, 
                                y=buyers['Close'], 
                                name='Buyers', 
                                mode='markers',
                                marker=dict(
                                color='#00CC96',
                                size=10,
                                )
                                ))
        fig11.add_trace(go.Scatter(x=sellers.index, 
                                y=sellers['Close'], 
                                name='Sellers', 
                                mode='markers', 
                                marker=dict(
                                color='#EF553B',
                                size=10,
                                )
                                ))
        fig11.update_layout(
        autosize=False,
        width=1600,
        height=650,)
        st.plotly_chart(fig11, use_container_width=False, sharing="streamlit", theme="streamlit")
        if st.button("Explanation1"):
                st.markdown('''In the chart depicted above, **:green[Bollinger Bands]** bracket the :green[20-day SMA] of the stock with an upper and lower band 
along with the daily movements of the stock's price. Because standard deviation is a measure of volatility, 
when the markets become more volatile the bands widen; during less volatile periods, the bands contract. ''')


        st.subheader("Graphical Analysis of the data ~")
        st.subheader('ploting Closing Price of the 2 Non-Covid periods')

        st.text("Histogram")
        # <==============================1st non covid period==============>
        
        fig3=plt.figure(figsize=(10,4))
        sns.histplot(x='Close', data=NCP1,label="Daily Close Price", binwidth=15)
        plt.legend()
        st.pyplot(fig3)

        # <==============================2nd non covid period==============>
        fig4=plt.figure(figsize=(10,4))
        sns.histplot(NCP2["Close"],label="Daily Close Price", binwidth=15)
        plt.legend()
        st.pyplot(fig4)


        st.text("Box Plot")
        # <===============================1st non covid period==============>
        fig5=plt.figure(figsize=(10,4))
        sns.boxplot(NCP1["Close"],labels=["Daily Close Price"])
        plt.legend()
        st.pyplot(fig5)
        # <===============================2nd non covid period==============>
        fig6=plt.figure(figsize=(10,4))
        plt.boxplot(NCP2["Close"],labels=["Daily Close Price"])
        st.pyplot(fig6)
        if st.button("Explanation2"):
                st.text('''As we can see, it’s full of outliers. The interquartile range 
(i.e. the height of the box) is quite narrow if compared
with the distribution total range.This phenomenon is called 
fat tails and it’s very common in stock analysis.''')

        # <======================================================================Descriptive Analysis====================================================================>


        st.subheader('Descriptive Analysis of the data ~') 
        st.text("for first Non_Covid period~")
        st.write(NCP1["Close"].describe())
        st.text("For 2nd Non_Covid Period~")
        st.write(NCP2["Close"].describe())
        if st.button("Explanation3"):
                st.text('''here Positive Mean Price indicates Positive drift of the Closing Price time series.
standerd deviation is more than an order of magnitude higher than the mean value. It’s clearly the effect of outliers. 
In stock price analysis, the standard deviation is a measure of the risk and 
such a high standard deviation is the reason why stocks are considered risky assets.

Median is also not so different from the mean value, so we might think that the distribution is symmetrical.
Let’s check the skewness of the distribution to better assess the symmetry:''')

        from scipy.stats import skew
        st.text("Skewness of the both Closing Price datasets")
        st.write(skew(NCP1["Close"]))
        st.write(skew(NCP2["Close"]))
        if st.button("Explanation4"):       
                st.text('''For first Covid Period the skewnwss is negative, so we can assume that the distribution is not symmetrical
(i.e. null skewness) and that the left tail as a not neglectable weight.
and for the 2nd Covid Period the skewness is positive , so we can assume that the distribution is not symmetrical 
and the right tail as a not neglectable weight.
If we perform a test on the skewness, we find:''')

        st.text("Skewness Test~")
        from scipy.stats import skewtest
        st.write(skewtest(NCP1["Close"]))
        st.write(skewtest(NCP2["Close"]))
        if st.button("Explanation5"):
                st.text('''The very low p-value suggests that the skewness of the distribution can’t be neglected,
so we can’t assume that it’s symmetrical.
Finally, we can measure the kurtosis (scipy normalizes the kurtosis so that it is 0 for a normal distribution)''')

        from scipy.stats import kurtosis
        st.text('Kurtosis of the both Closing Price datasets~')
        st.write(skew(NCP1["Close"]))
        st.write(skew(NCP2["Close"]))
        if st.button("Explanation6"):
                st.text('''It’s very different from zero, so the distribution is quite different from a normal one. 
One is Platikurtic and other one is leptokurtic
A kurtosis test gives us these results:''')

        st.text("Kurtosis Test~")
        from scipy.stats import kurtosistest
        st.write(kurtosistest(NCP1["Close"]))
        st.write(kurtosistest(NCP2["Close"]))
        if st.button("Explanation7"):
                st.text('''Again, a very small p-value lets us reject the null hypothesis that the kurtosis is the same as a normal distribution (which is 0).
Are returns normally distributed? Although the statistically significant high values of kurtosis and skewness already tell us that the returns aren’t normally distributed,
a Q-Q plot will give us graphically clear information.''')


        # ADF Test 
        st.subheader("ADF Test for Stationarity checking ~")
        from statsmodels.tsa.stattools import adfuller
        st.markdown("**:blue[for first non covid period~]**")
        test_result=adfuller(NCP1["Close"])
        def adfuller_test(sales):
                result=adfuller(sales)
                labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
                for value,label in zip(result,labels):
                        st.write((label+' : '+str(value) ))
                if result[1] <= 0.05:
                        st.write("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
                else:
                        st.write("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
        
        adfuller_test(NCP1["Close"])
        st.markdown("**:blue[for Second non covid period~]**")
        adfuller_test(NCP2["Close"])
        if st.button("Explanation11"):
                st.markdown('''So we can see that in the both non covid period the data is **:green[Non-Stationary]** which indicates that
                in normal time although the closing price varies very frequently(In other words Volatility) there is a significant change in the mean closing price over time 
                ,which is suitable for long term investment, And the high volatility suggests that although short term investment can be risky but with 
                good prediction it can also be pretty profitable. ''')


        st.subheader("Seasonal Decomposition ~")
        st.set_option('deprecation.showPyplotGlobalUse', False)  #To ignore the error what was occuring
        from statsmodels.tsa.seasonal import seasonal_decompose
        result1=seasonal_decompose(NCP1['Close'], model='multiplicable', period=24)
        plt.figure(figsize=(20,8))
        result1.plot()
        st.pyplot()

        from statsmodels.tsa.seasonal import seasonal_decompose
        result2=seasonal_decompose(NCP2['Close'], model='multiplicable', period=24)
        plt.figure(figsize=(20,8))
        result2.plot()
        st.pyplot()
        if st.button("Explanation8"):
                st.text('''Here in the both non covid period we are totally unalbe to capture a proper trend 
for that reason although having a seasonality in the data we try to perform "Moving Average" Method to compare the trend''')

        #Moving Average plots
        st.subheader("Trend Analysis of the data ~")
        st.subheader('Closing Price Vs Time chart with 100MA')
        st.text('''In real life most of the Technical Analyst follows the strategy that if the 100day MA
crosses above the 200day MA then there is an upward trend and if the 100day MA crosses below the 200day MA 
then it is the starting of an downward trend''')

        ma100=data.Close.rolling(100).mean()
        fig=plt.figure(figsize=(12,6))
        sns.lineplot(ma100,label="MA100")
        sns.lineplot(data.Close,label="Close Price")
        plt.legend()
        st.pyplot(fig)


        # <-------------------Moving Average plots -------------------->

        st.subheader('Closing Price Vs Time chart with 100MA & 200MA')
        ma100=data.Close.rolling(100).mean()
        ma200=data.Close.rolling(200).mean()
        fig=plt.figure(figsize=(12,6))
        sns.lineplot(ma100,label="MA100")
        sns.lineplot(ma200,label="MA200") 
        sns.lineplot(data.Close,label="Close Price")
        plt.legend()
        st.pyplot(fig)

        #st.text_area("")


if (selected == 'Prediction'):
        # <===================================================== Using Deep Learning Model for Prediction & Validating =================================================>

        st.title("Implimenting L.S.T.M model for predicting the future ~")

        # <-------------Splitting Data into Training and Testing --------------->
        import pandas as pd
        data_training=pd.DataFrame(data["Close"][0:int(len(data)*0.80)])
        data_test=pd.DataFrame(data['Close'][int(len(data)*0.80):int(len(data))])

        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler(feature_range=(0,1))

        data_training_array=scaler.fit_transform(data_training)

        # <------------------Load My Model ------------------------>
 
        model=load_model('LSTM.h5')

        past_100_days=data_training.tail(100)
        final_df=past_100_days.append(data_test,ignore_index=True)
        input_data=scaler.fit_transform(final_df)


        # <-------------------Testing The MOdel [LSTM.h5] ------------->
        x_test=[]
        y_test=[]

        for i in range(100,input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i,0])
        x_test,y_test=np.array(x_test),np.array(y_test)

        y_predicted=model.predict(x_test)
        

        scaler=scaler.scale_
        scale_factor=1/scaler[0]
        y_predicted=y_predicted*scale_factor
        y_test=y_test*scale_factor
        
        Comparison=pd.DataFrame(y_predicted,y_test)
        
        if st.button("Show the table "):
                st.checkbox("Use container width", value=True, key="use_container_width")
                st.dataframe(Comparison, use_container_width=st.session_state.use_container_width) # st.dataframe for making interactive dataframe in streamlit 


        # <----------------Final Graph -------------------->
        st.subheader('Predictions Vs Original')
        fig2=plt.figure(figsize=(12,6))
        sns.lineplot(y_test,label='original price',color='green')
        sns.lineplot(y_predicted,label='Predicted Price',color='yellow')
        plt.xlabel('Time')
        plt.ylabel('Price')
       
        st.pyplot(fig2)

        from sklearn.metrics import mean_absolute_error as mae
        from sklearn.metrics import mean_squared_error as mse
        from sklearn.metrics import r2_score
  
        # <---------calculation of MAE & RMSE ------------>

        error = mae(y_test, y_predicted)
        error2=mse(y_test,y_predicted)
        error3=np.sqrt(error2)
        error4=r2_score(y_test, y_predicted)
       
        st.text("MAE ~")
        st.write(error)

        st.text("RMSE ~")
        st.write(error3)

        st.text("R2 Score")
        st.write(error4)



        # <---------- Validating The Model ---------------->
        
        int_val = st.slider('**:blue[Select the number of days you want to Predict]**', min_value=1, max_value=15, value=8, step=1)
        
        last_7_days=validation_data.head(7).values

        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler(feature_range=(0,1))

        last_7_days_scaled=scaler.fit_transform(last_7_days)
        list1=[]
        pred_list=[]
        list1.append(last_7_days_scaled)
        for i in range(int_val):
                list1=np.array(list1)
                list1=np.reshape(list1,(list1.shape[0],list1.shape[1],1))
                pred_price=model.predict(list1)
                
                
                list1=np.reshape(1,1)
                list1=list(list1)

                # Here i have to append the real values of the data i just predicted to the pred list and then predict the next value.
                list1.append(pred_price)
                
                pred_price=scaler.inverse_transform(pred_price)
                pred_price=pred_price.flatten()

                pred_list.append(pred_price[0])
                list1.pop(0)
                

        import pandas as pd
        valid = {'Actual': data1["Close"][1426:(1426+(int_val))],
        'Validation': pred_list}
  
        # <------Create DataFrame dor the 7 future days.------------>
        valid1 = pd.DataFrame(valid)
        valid1["Difference"]=valid1["Actual"]-valid1["Validation"]

        
        st.markdown("To know the predicted price press **:blue[Predict]**  ")
        if st.button("Predict"):

                st.markdown("Validation of the Model ~")
                st.table(valid1)

                fig108=plt.figure(figsize=(12,6))
                sns.lineplot(valid1["Actual"],color='green',label='original price')
                sns.lineplot(valid1["Validation"],color='Purple',label='Predicted Price')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.legend()
                st.pyplot(fig108)
                
        # <------Optional-------->
        if st.button("Tap this only after reading the whole article."):
                st.balloons()
                st.success('Thanks for reading the whole article. Hope you have liked it.', icon="✅")


        # <--------CREDITS------->
if (selected=='Credits'):
        st.markdown('''**:green[This Web Application is currently under devoloped & owned by Arya Chakraborty [22MSD7020][VIT-AP].\\
For more info please contact me here ~\\
www.linkedin.com/in/arya-chakraborty-95a8411b2]**''')
    