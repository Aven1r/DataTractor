from datetime import timedelta
import pandas as pd
import numpy as np
import re
from prophet import Prophet


def fillna_mean(df):
    for column in df.columns:
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)

    return df

def nan_handler(df: pd.DataFrame):
        datetime = ['Дата и время']
        zero_data = [
            'Нагрузка на двигатель',
            'Стояночный тормоз',
            'Засоренность фильтра рулевого управления',
            'Засоренность фильтра навесного оборудования',
            'Отопитель',
            'Выход блока управления двигателем',
            'Включение тормозков',
            'Засоренность фильтра слива',
            'Аварийное давление масла КПП',
            'Аварийная температура масла ДВС',
            'Неисправность тормозной системы',
            'Термостарт',
            'Разрешение запуска двигателя',
            'Низкий уровень ОЖ',
            'Аварийная температура масла ГТР',
            'Необходимость сервисного обслуживания',
            'Подогрев топливного фильтра',
            'Вода в топливе',
            'Холодный старт',
            'Крутящий момент' ,
            'Положение рейки ТНВД' ,
            'Расход топлива' ,
            'Давление наддувочного воздуха двигателя',
            'Температура масла гидравлики' ,
            'Педаль слива',
            'iButton2']
        
        regex = '|'.join(map(re.escape, zero_data))  # Escape special characters
        print("Regex pattern:", regex)

        # Check if columns exist before dropping
        columns_to_drop = df.filter(regex=regex).columns
        if not columns_to_drop.empty:
            df = df.drop(columns_to_drop, axis=1)

        for col in df.columns:
            if col in datetime:
                df[col] = pd.to_datetime(df[col], format='%d/%m/%Y %H:%M:%S')
            if col not in datetime and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.').str.replace(':', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df
    
def train_params(df):
    # Предполагается, что функция fillna_mean уже определена
    # Если нет, вам нужно будет определить эту функцию.
    
    features = {
        "engine": "ДВС. Температура охлаждающей жидкости",
        "transmission": "КПП. Давление масла в системе смазки",
        "hydraulics": "Аварийная температура масла в гидросистеме (spn3849)",
        "brake": "Давление в пневмостистеме (spn46), кПа",
        "electric": "Электросистема. Напряжение"
    }
    
    # Инициализируем DataFrame для будущих прогнозов
    final_forecasts = pd.DataFrame()
    
    time_series = df["Дата и время"]
    
    future_dates = [time_series.max() + timedelta(seconds=30*x) for x in range(1, 2*24*60*2 + 1)]
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Перебор всех характеристик для создания и обучения моделей
    for feature_name, column_name in features.items():
        feature_df = pd.DataFrame({
            "ds": time_series,
            "y": df[column_name]
        })
        
        feature_df = fillna_mean(feature_df)
        feature_df.sort_values(by="ds", inplace=True)
        
        model = Prophet(changepoint_prior_scale=0.1)
        model.fit(feature_df)
        
        forecast = model.predict(future_df)
        forecast_filtered = forecast[forecast['ds'] > time_series.max()][['ds', 'yhat']]
        
        # Переименовываем 'yhat' для каждой характеристики
        forecast_filtered.rename(columns={'yhat': feature_name}, inplace=True)
        
        if final_forecasts.empty:
            final_forecasts = forecast_filtered
        else:
            # Объединяем по 'ds'
            final_forecasts = final_forecasts.merge(forecast_filtered, on='ds', how='outer')
    
    return final_forecasts

class AnomalyMLService():
    def __init__(self):
        self.model = Prophet()
        self.data = pd.DataFrame()
        self.predicted_params = pd.DataFrame()

    def preprocess(self, data):
        data = pd.read_csv(data, sep=';')
        data = data.iloc[0:10000, :]
        self.data = nan_handler(data)

    def train(self):
        self.predicted_params = train_params(self.data)
        return self.predicted_params
