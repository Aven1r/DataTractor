def drop_column(df):

    import pandas as pd
    import numpy as np
    pd.set_option('future.no_silent_downcasting', True)
    
    to_drop = []

    row = df.shape[0]
    for column in df.columns:
        if df[column].isna().sum() == row:
            to_drop.append(column)
        
        elif len(df[column][~df[column].isna()].value_counts().index) == 1:
            to_drop.append(column)

    nan_ratio = df.isna().sum() / len(df) * 100
    list_drop = list(nan_ratio.sort_values(ascending=True)[nan_ratio > 60].index)
    
    to_drop.extend(list_drop)
    
    return to_drop

def to_numeric(df):
    import pandas as pd

    datetime = ['Дата и время']
    for col in df.columns:
        df[col].fillna(0, inplace=True)
        if col not in datetime and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.').str.replace(':', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif df[col].dtype == bool:
            df[col] = df[col].astype(int)


def replace_to_nan(df):
    import numpy as np

    for column in df.columns:
        df[column].replace('        -', np.nan, inplace=True)
        df[column].replace('-', np.nan, inplace=True)
        #df[column].str.replace(',', '.', inplace=True)

def fillna_mean(df):
    drop_columns = []
    for column in df.columns:
        if df[column].isna().all():
            drop_columns.append(column)
        
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)
    return drop_columns
        

def time_data(df):

    df["День"] = df["Дата и время"].dt.day
    df["Месяц"]= df["Дата и время"].dt.month
    df["Час"] = df["Дата и время"].dt.hour

def split_dataframe(df, chunk_size=5000):
    
    chunks = []  # Список для хранения маленьких DataFrame
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)  # Вычисляем количество частей
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunks.append(df.iloc[start:end])  # Добавляем часть DataFrame в список
    
    return chunks

def tsfresh_feature(dfs, flag=0):
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_extraction import MinimalFCParameters

    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    
    extracted_dfs = []
    fc_parameters = MinimalFCParameters()

    for df in tqdm(dfs):
        if 'id' not in df.columns:
            df['id'] = 1

        # Предполагается, что функция fillna_mean(df) уже определена где-то в вашем коде
        fillna_mean(df)

        # Извлечение признаков с tsfresh
        extracted_features = extract_features(df, column_id='id', column_sort="Дата и время", 
                                              default_fc_parameters=fc_parameters)
        
        # Замена NaN значений после извлечения признаков
        impute(extracted_features)
        
        # Здесь нужно добавить столбец target непосредственно к extracted_features
        extracted_features['target'] = 0 if flag == 0 else 1
        
        extracted_dfs.append(extracted_features)

    return extracted_dfs

def get_tsfresh(df):
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_extraction import MinimalFCParameters

    fc_parameters = MinimalFCParameters()

    if 'id' not in df.columns:
            df['id'] = 1
    extracted_features = extract_features(df, column_id='id', column_sort="Дата и время", 
                                              default_fc_parameters=fc_parameters)
        
        # Замена NaN значений после извлечения признаков
    impute(extracted_features)    

    return extracted_features

def remove_highly_correlated_features_np(df, target_column="target", correlation_threshold=0.8):
    import numpy as np
    import pandas as pd

    # Выделяем матрицу признаков и вектор целевой переменной
    features_df = df.drop(columns=[target_column])
    df_values = features_df.values
    target_values = df[target_column].values

    # Вычисляем матрицу корреляций
    corr_matrix = np.corrcoef(df_values, rowvar=False)
    corr_with_target = np.corrcoef(np.vstack([df_values.T, target_values]))[:-1, -1]

    # Маска для отслеживания удаляемых колонок
    cols_to_drop = np.zeros(corr_matrix.shape[0], dtype=bool)
    
    for i in range(len(corr_matrix) - 1):
        for j in range(i + 1, len(corr_matrix)):
            if (np.abs(corr_matrix[i, j]) >= correlation_threshold) and not cols_to_drop[j]:
                # Определяем, какой из двух признаков меньше коррелирует с таргетом, и помечаем его для удаления
                if corr_with_target[i] > corr_with_target[j]:
                    cols_to_drop[j] = True
                else:
                    cols_to_drop[i] = True

    # Создаем список колонок для удаления
    cols_to_drop_list = features_df.columns[cols_to_drop].tolist()

    return cols_to_drop_list



def merge_dataframes_from_lists(list1, list2):
    import pandas as pd
    combined_list = list1 + list2
    merged_df = pd.concat(combined_list, axis=0).reset_index(drop=True)
    return merged_df

def nan_handler(df):
    import re
    import pandas as pd

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
