import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from datetime import datetime
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

filename = "G:/Mi unidad/Multiple_Year_Investment_Planning/modelo/Historical_Data_Granollers_1.xlsx"
df = pd.read_excel(filename)
df = df.set_index('Date_Time')
df.index = df.index.astype('datetime64[ns]')

# Yearly Values
Max_Winter_Yearly_Values = pd.DataFrame(index= [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
                                 columns=df.columns).applymap(lambda x: 0)
Min_Winter_Yearly_Values = pd.DataFrame(index= [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
                                 columns=df.columns).applymap(lambda x: 0)

for i in range(len(Max_Winter_Yearly_Values.columns)):
    df_max_winter = df.iloc[:, i].groupby(pd.Grouper(freq='M')).max()
    Max_Winter_Yearly_Values.iloc[:, i] = list(df_max_winter.values)

for i in range(len(Min_Winter_Yearly_Values.columns)):
    df_min_winter = df.iloc[:, i].groupby(df.index.month).min()
    Min_Winter_Yearly_Values.iloc[:, i] = list(df_min_winter.values)

# Day Values - Winter Scenario
Max_Winter_Daily_Values = pd.DataFrame(index= pd.date_range(start="2019-01-01",end="2019-12-30"),
                                 columns=df.columns).applymap(lambda x: 0)
Min_Winter_Daily_Values = pd.DataFrame(index= pd.date_range(start="2019-01-01",end="2019-12-30"),
                                 columns=df.columns).applymap(lambda x: 0)

for i in range(len(Max_Winter_Daily_Values.columns)):
    df_max_winter = df.iloc[:, i].groupby(pd.Grouper(freq='D')).max()
    Max_Winter_Daily_Values.iloc[:, i] = list(df_max_winter.values)

for i in range(len(Min_Winter_Daily_Values.columns)):
    df_max_winter = df.iloc[:, i].groupby(pd.Grouper(freq='D')).min()
    Min_Winter_Daily_Values.iloc[:, i] = list(df_max_winter.values)

Forecast_Results = pd.DataFrame(index= [2023, 2024, 2025, 2026, 'RMSE'],
                                 columns=df.columns).applymap(lambda x: 0)

for i in range(len(Forecast_Results.columns)):
    # close_data = df['E.T. LA TORRETA'].values
    close_data = Min_Winter_Yearly_Values.iloc[:, i].values
    close_data = close_data.reshape((-1,1))

    split_percent = 0.65
    split = int(split_percent*len(close_data))

    close_train = close_data[:split]
    close_test = close_data[split:]

    # date_train = df.index[:split]
    # date_test = df.index[split:]

    # Input Data
    date_train = Min_Winter_Yearly_Values.iloc[:, i].index[:split]
    date_test = Min_Winter_Yearly_Values.iloc[:, i].index[split:]

    look_back = 4

    train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=25)
    test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

    model = Sequential()
    model.add(
        LSTM(40,
            activation='relu',
            input_shape=(look_back,1))
    )
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    num_epochs = 70
    model.fit_generator(train_generator, epochs=num_epochs, verbose=1)
    prediction = model.predict_generator(test_generator)

    close_train = close_train.reshape((-1))
    close_test = close_test.reshape((-1))
    prediction = prediction.reshape((-1))
    print(close_train)

    # import plotly.graph_objects as go
    # trace1 = go.Scatter(
    #     x = date_train,
    #     y = close_train,
    #     mode = 'lines',
    #     name = 'Data'
    # )
    # trace2 = go.Scatter(
    #     x = date_test,
    #     y = prediction,
    #     mode = 'lines',
    #     name = 'Prediction'
    # )
    # trace3 = go.Scatter(
    #     x = date_test,
    #     y = close_test,
    #     mode='lines',
    #     name = 'Ground Truth'
    # )
    # layout = go.Layout(
    #     title = "Long-Term Forecast Validation",
    #     xaxis = {'title' : "Date"},
    #     yaxis = {'title' : "kWh"}
    # )
    # fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    # fig.show()

    close_data = close_data.reshape((-1))

    def predict(num_prediction, model):
        prediction_list = close_data[-look_back:]

        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back - 1:]

        return prediction_list

    def predict_dates(num_prediction):
        last_date = Max_Winter_Yearly_Values.index.values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction + 1, freq='Y').tolist()
        return prediction_dates

    num_prediction = 3
    forecast = predict(num_prediction, model)
    #forecast_dates = predict_dates(num_prediction)
    forecast_dates = [2023, 2024, 2025, 2026]

    # trace4 = go.Scatter(#     x = Max_Winter_Daily_Values.index,
    #     y = close_data,
    #     mode = 'lines',
    #     name = 'Trained'
    # )

    # trace5 = go.Scatter(
    #     x = forecast_dates,
    #     y = forecast,
    #     mode='lines',
    #     name = 'Predicted'
    # )
    # layout = go.Layout(
    #     title = "Long-Term Forecast",
    #     xaxis = {'title' : "Date"},
    #     yaxis = {'title' : "kWh"}
    # )
    # fig = go.Figure(data=[trace4, trace5], layout=layout)
    # fig.show()

    # plt.plot(date_test[0:1], prediction, label="Prediction Test")
    # plt.plot(date_test, close_test, label="Test Data")
    # plt.plot(date_train, close_train, label="Train Data")
    # plt.plot(forecast_dates, forecast, label="Long-Term Prediction")
    # plt.grid(True)
    # plt.title("Predicción Anual de Demanda")
    # plt.xlabel("Pico Máximo Anual de Invierno")
    # plt.ylabel("Potencia Activa (kWh)")
    # plt.legend()
    # plt.show()

    #calculate RMSE
    RMSE = sqrt(mean_squared_error(close_test[0:1], prediction))
    print("The RMSE is:", RMSE)
    Forecast_Results.iloc[:4,i] = forecast.tolist()
    Forecast_Results.iloc[4,i] = RMSE

