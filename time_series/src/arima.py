import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn import metrics


"""
Metodo 'time_series_analysis': determina de forma manual os parametros (p, d, q) do modelo 
ARIMA. O parametro d com recurso a dois metodos. O primeiro pela visualizacao grafica da 
media movel e desvio padrao movel. O segundo pelo teste estatistico de Dickey-Fuller.
Os parametros p e q com recurso aos graficos de ACF e PACF respectivamente.
"""


def time_series_analysis(file, column):
    # Read the data to a DataFrame
    df = pd.DataFrame()
    df = pd.read_csv(file)
    
    # Delete remaining columns
    columns_list = list(df.columns.values)
    columns_list.remove(column)
    for i in range(len(columns_list)): df.drop(columns_list[i], axis=1, inplace=True)    

    # Logarithmic transformation
    ts_log = np.log(df)

    # Run the series stationarity test
    stationarity_test(df, column)

    # Make the time series stationary with the application of a differentiation transformation
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)

    # After the transformation, it runs a new seasonality test of the series
    stationarity_test(ts_log_diff, column)

    # Estimate the parameters p, q using the ACF and PACF functions
    lag_acf = acf(ts_log_diff, nlags=20)
    lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
    
    # Plot ACF function
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / math.sqrt((len(ts_log_diff))), linestyle='--', color='gray')
    plt.axhline(y=1.96 / math.sqrt((len(ts_log_diff))), linestyle='--', color='gray')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.title('Auto correlation Function')
    
    # Plot PACF function
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / math.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / math.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.xlabel('Lag')
    plt.ylabel('PACF')
    plt.title('Partial Auto correlation Function')
    plt.tight_layout()
    plt.show()


"""
Metodo 'stationarity_test': chamado pelo metodo anterior para realizar o teste de 
estacionariedade a serie temporal com recurso a visualizacao grafica da media movel 
e desvio padrao movel, e ao teste Dickey-Fuller.
"""


def stationarity_test(ts, column):
    # Determing rolling statistics
    rolmean = ts.rolling(12).mean()
    rolstd = ts.rolling(12).std()
    legend_xx_aux = [i for i in range(len(rolmean))]
    legend_xx = ['Jan 2014', '', '', '', '', '', '', '', '', '', '', '', 'Jan 2015', '', '',
                '', '', '', '', '', '', '', '', '', 'Jan 2016', '', '', '', '', '', '', '', 
                '', '', '', '', 'Jan 2017', '', '', '', '', '', '', '', '', '', '', '']

    # Plot rolling statistics
    plt.xticks(legend_xx_aux, legend_xx)
    orig = plt.plot(legend_xx_aux, ts[column], color='blue', label='Original')
    mean = plt.plot(legend_xx_aux, rolmean, color='red', label='Rolling Mean')
    std = plt.plot(legend_xx_aux, rolstd, color='black', label='Rolling Std')
    plt.xlabel('Time (months)')
    plt.ylabel('Value')
    plt.title('Statistical properties of the time series')
    plt.legend(loc='best')
    plt.show()

    # Perform Dickey-Fuller test on the timeseries and verify the null hypothesis that the TS is non-stationary
    print('Results of Dickey-Fuller Test:')
    df_test = adfuller(ts.iloc[:, 0].values, autolag='AIC')
    df_output = pd.Series(df_test[0:4], index=['Test Statistic', 'p-value', '#lags Used', 'Number of Observations Used'])
    for key, value in df_test[4].items():
        df_output['Critical Value (%s)' % key] = value
    print(df_output)


"""
Metodo 'forecast': preve os meses seguintes, calcula o erro de previsao (RMSE e MAPE) 
e mostra o resultado graficamente.
"""


def forecast(file, column):
    # Read the data to a DataFrame
    df = pd.DataFrame()
    df = pd.read_csv(file)
    
    # Delete remaining columns
    columns_list = list(df.columns.values)
    columns_list.remove(column)
    for i in range(len(columns_list)): df.drop(columns_list[i], axis=1, inplace=True)

    # Split data into train and test-sets
    df_test = df[(len(df) - 12):]
    df_train = df[:-12]
    y_true = list(df_test[column])

    # Logarithmic transformation
    ts_log = np.log(df_train)

    # Building the model - parametrization:
    # 1. order = (p, d, q)
    # 2. seasonal_order configured with the value of 12 corresponding to the seasonal cycle in the fourth parameter
    model = sm.tsa.statespace.SARIMAX(ts_log, order=(1, 1, 0), seasonal_order=(1, 0, 0, 12))
    results = model.fit(disp=-1)
    pred_uc = results.get_forecast(steps=12)  # steps: number of months predicted
    pred_ci = pred_uc.conf_int() # exponential transformation (inverse of logarithmic)
    predictions = np.exp(pred_ci)
    minimum = list(predictions['lower ' + column])
    maximum = list(predictions['upper ' + column])
    y_pred = [(minimum[i] + ((maximum[i] - minimum[i]) / 2)) for i in range(len(minimum))]

    model_evaluation(y_true, y_pred)

    plot(column, df_train, y_true, y_pred)
    

def model_evaluation(y_true, y_pred):
    rmse = math.sqrt(metrics.mean_squared_error(y_true, y_pred))
    mape = round(((metrics.mean_absolute_error(y_true, y_pred)) / (sum(y_true) / len(y_true))) * 100, 2)

    print('RMSE = ' + str(rmse))
    print('MAPE (%) = ' + str(mape))	


def plot(column, df_train, y_true, y_pred):
    legend_xx_aux = [(i + 1) for i in range(48)]
    legend_xx = ['Jan 2014', '', '', '', '', '', '', '', '', '', '', '', 'Jan 2015', '', '', '',
            '', '', '', '', '', '', '', '', 'Jan 2016', '', '', '', '', '', '', '', '', '',
            '', '', 'Jan 2017', '', '', '', '', '', '', '', '', '', '', 'Dez 2017']
    pred_values_list = list(df_train[column])
    pred_values_list.extend(y_pred)
    true_values_list = list(df_train[column])
    true_values_list.extend(y_true)

    plt.xticks(legend_xx_aux, legend_xx)
    plt.plot(legend_xx_aux, pred_values_list, label='Forecast')
    plt.plot(legend_xx_aux, true_values_list, label='Real')
    plt.xlabel('Time (months)')
    plt.ylabel('Data volume')
    plt.title('Forecasting of data consumption')
    plt.legend()
    plt.show()


if __name__ == '__main__': 
    """ 
    --- Coluna com o tipo de dados existente no ficheiro no segundo parametro
           recharges
           SMS
           voice_calls
           data
    """
    #time_series_analysis('TelecomData.csv', 'data')
    forecast('TelecomData.csv', 'data')