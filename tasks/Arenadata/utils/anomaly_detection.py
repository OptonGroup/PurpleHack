import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def process_clients_files_except(folder_path, exclude_ids):
    """
    Функция проходит по всем файлам в папке folder_path, кроме тех, чьи ID находятся в exclude_ids.
    Для каждого файла выполняет обработку и обнаружение аномалий с использованием Isolation Forest.
    
    Параметры:
      folder_path : str
          Путь к папке, содержащей файлы клиентов.
      exclude_ids : list
          Список ID клиентов, которые нужно исключить из обработки.
    """
    # Получаем список всех файлов в директории
    all_files = [f for f in os.listdir(folder_path) if f.startswith('client_') and f.endswith('.csv')]
    
    # Преобразуем exclude_ids в set для быстрого поиска
    exclude_set = set(exclude_ids)
    
    for filename in all_files:
        # Извлекаем ID клиента из имени файла (убираем 'client_' и '.csv')
        client_id = filename[7:-4]  # client_XXXXX.csv -> XXXXX
        
        # Пропускаем файлы, которые есть в exclude_ids
        if client_id in exclude_set:
            continue
            
        file_path = os.path.join(folder_path, filename)
        try:
            # Загрузка данных
            client = pd.read_csv(file_path)
            
            # Удаление лишних столбцов
            client = client.drop(columns=['IdPSX', 'IdSubscriber', 'Name', 'Id', 
                                       'CreatedAt', 'UpdatedAt', 'ClosedAt'], errors='ignore')
            
            # Предварительная обработка данных
            from utils.feature_generation import process_client_data, process_session_columns, add_endpoint_column
            
            client = process_client_data(client)
            client = process_session_columns(client)
            client = add_endpoint_column(client)
            
            # Масштабирование столбцов 'EndPoint' и 'AvgPackets' в диапазоне [0, 1]
            for col in ['EndPoint', 'AvgPackets']:
                min_val = client[col].min()
                max_val = client[col].max()
                if max_val != min_val:
                    client[col] = (client[col] - min_val) / (max_val - min_val)
            
            print(f'Обрабатываем файл {file_path}')
            
            # === Обнаружение аномалий с Isolation Forest ===
            # 1. Формирование матрицы признаков
            features = client[['AvgPackets', 'AvgUp', 'AvgDown', 'EndPoint']].copy()
            
            # 2. Обучение модели Isolation Forest
            model = IsolationForest(contamination=0.05, random_state=42)
            model.fit(features)
            
            # 3. Получение предсказаний (-1 — аномалия, 1 — нормальная точка)
            predictions = model.predict(features)
            
            # 4. Добавление колонки с флагом аномалии
            client['anomaly'] = (predictions == -1)
            
            # === Визуализация результатов ===
            plt.figure(figsize=(10, 6))
            
            # Нормальные точки
            plt.scatter(
                client.loc[client['anomaly'] == False, 'EndPoint'],
                client.loc[client['anomaly'] == False, 'AvgPackets'],
                c='blue',
                label='Normal'
            )
            
            # Аномальные точки
            plt.scatter(
                client.loc[client['anomaly'] == True, 'EndPoint'],
                client.loc[client['anomaly'] == True, 'AvgPackets'],
                c='red',
                label='Anomaly'
            )
            
            plt.xlabel('EndPoint')
            plt.ylabel('AvgPackets')
            plt.title(f'Isolation Forest - Anomaly Detection for {client_id}')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")

# Пример использования:
# result = pd.read_csv('telecom100k/telecom100k/telecom100k/RESULT', sep=',')
# exclude_ids = result['UID'].unique()
# process_clients_files_except('clients', exclude_ids) 