import pandas as pd
import os
from utils.clusterization_longdist import process_single_client_file_dbscan
from data_preparing.feature_selection import final_feature_selection
from data_preparing.feature_generation import process_client_data, process_session_columns, add_endpoint_column
from utils.data_extraction import data_preparation, update_client_files
from utils.clusterization_realtime import process_all_clients
from utils.clusterization_realtime import filter_files_by_uids
from tqdm import tqdm
from utils.clusterization_longdist import find_closest_point_to_max_mean_cluster


def update_data(files, clinets_path='new_clients'):

        # Подгружаемые файлы для парсинга в итоговую таблицу пользователя
        subscribers = pd.read_csv('telecom100k/data/subscribers.csv')
        client_df = pd.read_parquet('telecom100k/data/client.parquet')
        plan_df = pd.read_json('telecom100k/data/plan.json')
        psxattrs_df = pd.read_csv('telecom100k/data/psxattrs.csv')

        j = pd.read_parquet('telecom100k/data/company.parquet')['Id'].to_numpy()
        p = pd.read_parquet('telecom100k/data/physical.parquet')['Id'].to_numpy()


        data = data_preparation(files, subscribers, client_df, plan_df, psxattrs_df, id_p=p, id_j=j)
        print('Data prepared')

        data = process_client_data(data)
        print('Client data processed')
        data = process_session_columns(data)
        print('Session columns processed')
        data = add_endpoint_column(data)
        print('Endpoint column added')

        final_df = final_feature_selection(data)
        print('Features selected')

        stats = update_client_files(final_df, clinets_path)
        print('Files updated')
        return stats


def detection_realtime(files, clinets_path='new_clients', update=True):
    '''
    Обрабатывает файлы с данными пользователей, обновляет информацию о пользователях и выполняет кластеризацию
    для обнаружения потенциально взломанных пользователей в реальном времени. Возвращает витрину с данными о
    взломанных пользователях.

    Параметры:
      files : list
          Список файлов с данными пользователей.
      clinets_path : str
          Путь к папке, где хранятся файлы клиентов.
      update : bool
          Флаг, указывающий, нужно ли обновлять данные пользователей.

    Возвращает:
      anomalies : DataFrame
          Датафрейм с информацией о взломанных пользователях.
      dfs : list
          Список датафреймов с данными пользователей.
    '''
    # Обновление данных пользователей
    if update:
        stats = update_data(files, clinets_path)
        print(f'Data updated : {stats}')


    dfs = process_all_clients(clinets_path, 10)
    anomalies = pd.DataFrame()

    print('Processing clients')
    for df in tqdm(dfs):
            line = max(len(df) - 10, 0)
            part = df.iloc[line:]
            if part['anomaly'].sum() > 5:
                seria = part[part['anomaly'] == True].head(1)
                anomalies = pd.concat([anomalies, seria])
    print(f'Аномалий предсказанных как новые возможные взломы: {len(anomalies)}')
    try:
        anomalies = anomalies[['IdSession', 'IdClient', 'Type', 'IdPlan', 'Enabled',
                           'anomaly', 'AvgPackets']].reset_index(drop=True)
        anomalies.columns = ['Id', 'UID', 'Type', 'IdPlan', 'TurnOn', 'Hacked', 'Traffic']
    
        logits = update_probably_hacked(anomalies['UID'])
        return logits

    except Exception as e:
        pass


def update_probably_hacked(uids):
    """
    Обновляет файл с потенциально взломанными пользователями, добавляя только новые идентификаторы.
    
    Параметры:
      uids : pandas.Series или list
          Серия или список идентификаторов пользователей для добавления в файл.
    
    Возвращает:
      dict
          Словарь с информацией о количестве добавленных новых идентификаторов.
    """
    # Преобразуем входные данные в список, если они еще не в этом формате
    if hasattr(uids, 'tolist'):
        uids_list = uids.tolist()
    else:
        uids_list = list(uids)
        
    # Проверяем, существует ли файл probably_hacked.csv
    if os.path.exists('probably_hacked.csv'):
        # Файл существует, читаем текущие идентификаторы
        existing_df = pd.read_csv('probably_hacked.csv')
        existing_uids = existing_df['UID'].tolist()
        
        # Находим только новые идентификаторы (те, которых еще нет в файле)
        new_uids = [uid for uid in uids_list if uid not in existing_uids]
        
        # Если есть новые идентификаторы, добавляем их
        if new_uids:
            new_df = pd.DataFrame({'UID': new_uids})
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_csv('probably_hacked.csv', index=False)
            added_count = len(new_uids)
        else:
            # Новых идентификаторов нет, файл остается без изменений
            added_count = 0
    else:
        # Файл не существует, создаем новый файл со всеми идентификаторами
        new_df = pd.DataFrame({'UID': uids_list})
        new_df.to_csv('probably_hacked.csv', index=False)
        added_count = len(uids_list)
    
    return {'added': added_count, 'total': len(uids_list)}
    


def showcase(clients_path='new_clients', merge_threshold=0.05, eps=0.1, min_samples=10, min_distance=0.65):
    """
    Выполняет кластеризацию данных клиентов, чьи UID содержатся в probably_hacked.csv или already_hacked.csv,
    и определяет текущие аномалии. Обновляет файлы probably_hacked.csv и already_hacked.csv на основе результатов.
    
    Параметры:
      clients_path : str
          Путь к папке с данными клиентов.
      merge_threshold : float
          Пороговое значение для слияния близких кластеров.
      eps : float
          Максимальное расстояние между двумя образцами для DBSCAN.
      min_samples : int
          Минимальное количество образцов в окрестности точки для DBSCAN.
      min_distance : float
          Минимальное расстояние между кластерами для обнаружения аномалий.
          
    Возвращает:
      anomalies : DataFrame
          Датафрейм с информацией о текущих аномалиях.
    """
    clients = os.listdir(clients_path)
    
    # Получаем UID из обоих файлов
    all_uids = set()
    
    # Получаем UID из probably_hacked.csv
    if os.path.exists('probably_hacked.csv'):
        probably_hacked_df = pd.read_csv('probably_hacked.csv')
        probably_uids = set(probably_hacked_df['UID'].tolist())
        all_uids.update(probably_uids)
    
    # Получаем UID из already_hacked.csv
    if os.path.exists('already_hacked.csv'):
        already_hacked_df = pd.read_csv('already_hacked.csv')
        already_uids = set(already_hacked_df['UID'].tolist())
        all_uids.update(already_uids)
    
    # Преобразуем множество UID обратно в DataFrame для функции filter_files_by_uids
    all_uids_df = pd.DataFrame({'UID': list(all_uids)})
    
    # Фильтруем клиентов
    filtered_clients = filter_files_by_uids(clients, all_uids_df)

    anomalies = pd.DataFrame()

    # Обрабатываем всех потенциально взломанных клиентов
    for filtered_client in tqdm(filtered_clients):
        distance, client = process_single_client_file_dbscan(clients_path + '/' + filtered_client, merge_threshold, eps, min_samples)
        if distance > min_distance:
            point = find_closest_point_to_max_mean_cluster(client)
            point['anomaly'] = True
            point = pd.DataFrame(point).T
            anomalies = pd.concat([anomalies, point])
    
    if not anomalies.empty:
        anomalies = anomalies[['IdSession', 'IdClient', 'Type', 'IdPlan', 'Enabled',
                              'anomaly', 'AvgPackets']].reset_index(drop=True)
        anomalies.columns = ['Id', 'UID', 'Type', 'IdPlan', 'TurnOn', 'Hacked', 'Traffic']
        
        # Получаем список UID, которые являются текущими аномалиями
        current_anomaly_uids = set(anomalies['UID'].tolist())
        
        # Обновляем файл already_hacked.csv - записываем туда только текущие аномалии
        current_anomalies_df = pd.DataFrame({'UID': list(current_anomaly_uids)})
        current_anomalies_df.to_csv('already_hacked.csv', index=False)
        print(f"Файл already_hacked.csv обновлен: {len(current_anomaly_uids)} текущих аномалий")
        
        # Удаляем все подтвержденные аномалии из probably_hacked.csv
        if os.path.exists('probably_hacked.csv'):
            probably_hacked_df = pd.read_csv('probably_hacked.csv')
            updated_probably_hacked = probably_hacked_df[~probably_hacked_df['UID'].isin(current_anomaly_uids)]
            updated_probably_hacked.to_csv('probably_hacked.csv', index=False)
            removed_count = len(probably_hacked_df) - len(updated_probably_hacked)
            print(f"Удалено {removed_count} подтвержденных UID из файла probably_hacked.csv")
    else:
        # Если аномалий не обнаружено, создаем пустой файл already_hacked.csv
        pd.DataFrame(columns=['UID']).to_csv('already_hacked.csv', index=False)
        print("Файл already_hacked.csv обновлен: 0 текущих аномалий")

    return anomalies


def detection(clients_path='new_clients', merge_threshold=0.05, eps=0.1, min_samples=10, min_distance=0.65):
    """
    Выполняет кластеризацию данных клиентов из указанной папки с использованием алгоритма DBSCAN
    и обнаруживает аномалии. Если расстояние между двумя самыми большими кластерами превышает
    заданный порог, функция находит ближайшую точку из кластера с наименьшим средним значением
    к кластеру с наибольшим средним значением и помечает её как аномалию.

    Параметры:
      clients_path : str
          Путь к папке с данными клиентов.
      merge_threshold : float
          Пороговое значение для слияния близких кластеров.
      eps : float
          Максимальное расстояние между двумя образцами для того, чтобы один считался в пределах другого.
      min_samples : int
          Минимальное количество образцов в окрестности точки для того, чтобы она считалась основной точкой.
      min_distance : float
          Минимальное расстояние между двумя самыми большими кластерами для обнаружения аномалий.

    Возвращает:
      anomalies : DataFrame
          Датафрейм с информацией о пользователях, помеченных как аномалии в виде той же витрины.
    """
    clients = os.listdir(clients_path)
    anomalies = pd.DataFrame()


    for client in tqdm(clients):
        distance, client = process_single_client_file_dbscan(clients_path + '/' + client, merge_threshold, eps, min_samples)
        if distance > min_distance:
            point = find_closest_point_to_max_mean_cluster(client)
            point['anomaly'] = True
            point = pd.DataFrame(point).T
            anomalies = pd.concat([anomalies, point])


    anomalies = anomalies[['IdSession', 'IdClient', 'Type', 'IdPlan', 'Enabled',
                            'anomaly', 'AvgPackets']].reset_index(drop=True)
    anomalies.columns = ['Id', 'UID', 'Type', 'IdPlan', 'TurnOn', 'Hacked', 'Traffic']

    return anomalies