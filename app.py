import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.plot import plot, plot_components
import matplotlib.dates as mdates



st.title(':blue[Future 30 Days Stock Price Prediction]')
st.subheader(':red[Data is Download and Upload From Yahoo! Finance with CSV format]')
# Upload The File
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df1 = df.set_index('Date')
    st.write(df)
    # Plot the Historical Data
    # Close
    fig_close, ax = plt.subplots()
    ax.plot(df['Date'], df['Close'])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis='x', rotation=90)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Close Price over Time')
    st.pyplot(fig_close)
    
    # Volume 
    fig_volume , ax1 = plt.subplots()
    ax1.plot(df['Date'], df['Volume'])
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.tick_params(axis='x', rotation=90)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Share Buy')
    ax1.set_title('Share Buy Over Time')
    st.pyplot(fig_volume)

    #Create a Dataframe for Close and Date attributes
    columns = ['Date','Close']
    ndf = pd.DataFrame(df, columns=columns)
    #st.write(ndf)
    
    # Fit the model
    prophet_df = ndf.rename(columns={'Date':'ds', 'Close':'y'})
    #st.write(prophet_df)
    m = Prophet()
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods = 30)
    forecast = m.predict(future)
    #st.write(forecast)
    
    # Plot the Monthly Predicted Graph
    Predicted_data = pd.DataFrame({'Date': forecast['ds'], 'Predicted': forecast['yhat']})
    Predicted_data.set_index('Date', inplace=True)
    st.markdown(':green[**Future Trends**]')
    st.line_chart(Predicted_data)
    st.markdown(':green[**Forecast for future 30 Days**]')
    st.line_chart(Predicted_data[-30:])
    
    #Forecast Vs Real Close
    fig_forecast_close = m.plot(forecast, xlabel='Date', ylabel='Forecast Vs Close')
    st.markdown(':green[**Prediction Vs Real Close Value**]')
    st.pyplot(fig_forecast_close)
    st.markdown(':green[**Trend and Weekly**]')
    fig_trend_weekly = m.plot_components(forecast)
    st.pyplot(fig_trend_weekly)
    
    # Show The Predicted File 
    df_forecast = forecast[['ds', 'yhat']]
    df_forecast = df_forecast.rename(columns = {'ds':'Date', 'yhat':'Predicted Value'})
    df_forecast.index.freq = 'D'
    df_forecast_download = df_forecast[-30:]
    st.markdown(':green[**Future 30 Days Forecasting Price**]')
    st.write(df_forecast_download)
    
    # Download The Predicted File in CSV format
    def convert_df(df_forecast_download):
        return df_forecast_download.to_csv(index=False).encode('utf-8')
    
    csv = convert_df(df_forecast_download)
    st.download_button(
        "Press to Download",
        csv,
        "Prediction.csv",
        "text/csv",
        key='download-csv'
    )
