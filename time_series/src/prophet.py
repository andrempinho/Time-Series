import pandas as pd
from fbprophet import Prophet
from src import arima as common_methods

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
    
    # Prophet requires columns named with the values 'ds' and 'y'
    df_train = pd.DataFrame({'ds': list(df_train.index), 'y': list(df_train[column])})
    
    # Building the model - parametrization:
    # 1. changepoint_prior_scale configurado com o valor default de 0.5. Permite ajustar a 
    #    flexibilidade da componente da tendencia ao longo do tempo. Quanto maior o seu valor maior sera o ajuste da curva a serie temporal
    # 2. add_seasonality configurado com o valor de 12 correspondente ao ciclo sazonal
    model = Prophet(weekly_seasonality=False, changepoint_prior_scale=0.5)
    model.add_seasonality(name='monthly', period=12)
    model.fit(df_train)
    
    # Forecast
    future = model.make_future_dataframe(periods=12) # number of months predicted
    predictions = model.predict(future)
    y_pred = list(predictions['yhat'])
    y_pred = y_pred[len(df_train)::]
    
    common_methods.model_evaluation(y_true, y_pred)
    
    common_methods.plot(column, df_train, y_true, y_pred)


if __name__ == '__main__': 
    """ 
    --- Coluna com o tipo de dados existente no ficheiro no segundo parametro
           recharges
           SMS
           voice_calls
           data
    """    
    forecast('TelecomData.csv', 'recharges')