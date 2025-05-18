import glob
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from data_preparing.feature_selection import selection_for_clusterisation
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def clusterization_iforest(client_df, threshold=2.0):
    """
    Обнаружение аномалий с использованием Z-score.
    
    Параметры:
    -----------
    client_df : pandas.DataFrame
        DataFrame с данными клиентов.
    threshold : float
        Пороговое значение Z-score для определения аномалий (по умолчанию 3.0).
    
    Возвращает:
    --------
    pandas.DataFrame
        DataFrame с добавленным столбцом 'anomaly', где True - аномалия.
    """
    # Выбор признаков для кластеризации
    features = selection_for_clusterisation(client_df)
    
    # Масштабирование признаков
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Вычисление Z-score
    z_scores = np.abs((scaled_features - np.mean(scaled_features, axis=0)) / np.std(scaled_features, axis=0))
    
    # Определение аномалий
    client_df['anomaly'] = (z_scores > threshold).any(axis=1)
    
    return client_df


def process_all_clients(clients_dir, min_points=144, max_points=1000):
    """
    Обрабатывает все файлы клиентов в указанной директории,
    применяя к каждому Isolation Forest для обнаружения аномалий.
    
    Parameters:
    -----------
    clients_dir : str
        Путь к директории с файлами клиентов
        
    Returns:
    --------
    list
        Список DataFrame'ов с размеченными аномалиями
    """
    # Получаем список всех файлов клиентов
    client_files = [f for f in os.listdir(clients_dir) if f.endswith('.csv')]
    
    # Список для хранения обработанных датафреймов
    processed_dfs = []
    
    # Обрабатываем каждый файл
    for file_name in tqdm(client_files):
        try:
            # Загружаем данные клиента
            file_path = os.path.join(clients_dir, file_name)
            client_df = pd.read_csv(file_path)
            client_len = len(client_df)

            if client_len > max_points:
                client_df = client_df.iloc[int(client_len * 0.6):]

            # Проверяем, есть ли достаточно данных для кластеризации
            if len(client_df) > min_points:  # минимум n записи для кластеризации
                # Масштабирование данных
                client_df = clusterization_iforest(client_df)
                # Добавляем в список результатов
                processed_dfs.append(client_df)
            else:
                continue
                
        except Exception as e:
            print(f"Ошибка при обработке файла {file_name}: {e}")
            continue
    
    return processed_dfs


def filter_files_by_uids(file_names, uid_dataframe):
    """
    Фильтрует список имен файлов, оставляя только те, идентификаторы которых 
    присутствуют в датафрейме с колонкой 'UID' или в Series.
    
    Параметры:
    -----------
    file_names : list
        Список имен файлов вида 'client_uuid.csv'
    uid_dataframe : DataFrame или Series
        Датафрейм с колонкой 'UID' или Series, содержащие идентификаторы
        
    Возвращает:
    --------
    list
        Отфильтрованный список имен файлов
    """
    # Функция для извлечения идентификатора из имени файла
    def extract_client_id(filename):
        try:
            client_id_with_ext = filename.split('_')[1]
            client_id = client_id_with_ext.split('.')[0]
            return client_id
        except (IndexError, AttributeError):
            return None
    
    # Проверяем, является ли uid_dataframe Series или DataFrame
    if isinstance(uid_dataframe, pd.Series):
        uid_set = set(uid_dataframe.values)
    else:  # DataFrame
        if 'UID' in uid_dataframe.columns:
            uid_set = set(uid_dataframe['UID'])
        else:
            # Если колонки UID нет, используем первую колонку DataFrame
            uid_set = set(uid_dataframe.iloc[:, 0])
    
    # Фильтруем файлы, оставляя только те, идентификаторы которых есть в датафрейме
    filtered_files = []
    for file_name in file_names:
        client_id = extract_client_id(file_name)
        if client_id is not None and client_id in uid_set:
            filtered_files.append(file_name)
    
    return filtered_files


def process_files_in_batches(directory='telecom100k/psx', batch_size=6):
    """
    Получает список всех файлов в указанной директории, читает их в pandas DataFrames 
    и группирует в батчи заданного размера.
    
    Параметры:
    -----------
    directory : str
        Путь к директории с файлами (по умолчанию: 'psx').
    batch_size : int
        Размер пачки файлов для обработки (по умолчанию: 6).
        
    Возвращает:
    --------
    list
        Список батчей, где каждый батч - это список pandas DataFrames.
    """
    # Получаем абсолютный путь к директории
    abs_directory = os.path.abspath(directory)
    
    # Получаем список всех файлов в директории с полными путями
    file_paths = glob.glob(os.path.join(abs_directory, '*'))
    
    # Сортируем файлы по имени
    file_paths.sort()
    
    # Разбиваем список файлов на пачки по batch_size
    batches = []
    for i in range(0, len(file_paths), batch_size):
        # Получаем пути к файлам для текущего батча
        batch_paths = file_paths[i:i+batch_size]
        
        # Читаем каждый файл в DataFrame
        batch_dfs = []
        for file_path in batch_paths:
            try:
                # Определяем формат файла и читаем его соответствующим методом
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                elif file_path.endswith('.json'):
                    df = pd.read_json(file_path)
                else:
                    # Для неизвестных форматов пробуем csv
                    df = pd.read_csv(file_path)
                
                # Сохраняем путь к файлу как атрибут DataFrame
                df.file_path = file_path
                
                batch_dfs.append(df)
            except Exception as e:
                print(f"Ошибка при чтении файла {file_path}: {e}")
        
        # Добавляем батч с DataFrames в результат
        batches.append(batch_dfs)
    
    return batches