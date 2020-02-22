from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy
import time
from src import arima as common_methods

"""
Metodo 'forecast' utiliza o modelo Long Short-Term Memory network (LSTM), que e
um tipo de Recurrent Neural Network (RNN). Preve os meses seguintes e 
calcula o erro de previsao (RMSE e MAPE) 
e mostra o resultado graficamente.
Parametros do modelo:
- Conjunto de treino
- batch_size (deve ser 1)
- number_epochs
- number_neurons / units
"""


def forecast(file, column):
    # Read the data to a DataFrame
    df = DataFrame()
    df = read_csv(file)

    # Delete remaining columns
    columns_list = list(df.columns.values)
    columns_list.remove(column)
    for i in range(len(columns_list)): df.drop(columns_list[i], axis=1, inplace=True)

    series = Series(list(df[column]), index=list(df.index))
    series.index.name = 'Data'

    # Split data into train and test-data frames
    df_test = df[(len(df) - 12):]
    df_train = df[:-12]
    y_true = list(df_test[column])

    # Transform data to be stationary
    raw_values = series.values
    print(type(raw_values))
    diff_values = difference(raw_values, 1)

    # Transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values

    # Split data into train and test-sets
    train, test = supervised_values[0:-12], supervised_values[-12:]

    # Transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)

    # Building the model - parametrization:
    # 1. train set
    # 2. batch_size
    # 3. number_epochs
    # 4. number_neurons
    start = time.time()
    lstm_model = fit_lstm(train_scaled, 1, 250, 24)
    # Forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)
    end = time.time()
    print("Execution time: " + str(end - start) + " s")

    # Walk-forward validation on the test data
    y_pred = list()
    for i in range(len(test_scaled)):
        # Make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        # Invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # Invert differencing
        yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
        # Store forecast
        y_pred.append(yhat)
        expected = raw_values[len(train) + i + 1]
        print('Month=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

    common_methods.model_evaluation(y_true, y_pred)

    common_methods.plot(column, df_train, y_true, y_pred)


"""
Metodos auxiliares
"""


# Date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


# Frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# Create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# Invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# Scale train and test data to [-1, 1]
def scale(train, test):
    # Fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)

    # Transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)

    # Transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)

    return scaler, train_scaled, test_scaled


# Inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)

    return inverted[0, -1]


# Fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    model = Sequential()
    model.add(LSTM(neurons, activation='tanh', batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=False))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()

    return model


# Make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


if __name__ == '__main__':
    """ 
    --- Coluna com o tipo de dados existente no ficheiro no segundo parametro
           recharges
           SMS
           voice_calls
           data
    """
    forecast('TelecomData.csv', 'recharges')