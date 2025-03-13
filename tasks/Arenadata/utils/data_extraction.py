import pandas as pd
import os
import glob
from tqdm import tqdm


def load_client_dataframes(dir='telecom100k/psx', num_files=None):
    """
    Загружает указанное количество CSV файлов из директории клиентов в pandas DataFrame.

    Параметры:
    -----------
    dir : str
        Путь к директории, содержащей CSV файлы клиентов.
    num_files : int, optional
        Количество файлов для загрузки. По умолчанию загружаются все файлы.

    Возвращает:
    --------
    list
        Список pandas DataFrame, загруженных из CSV файлов.
    """
    # Получаем все файлы в директории с поддерживаемыми расширениями
    csv_files = glob.glob(os.path.join(dir, '*.csv'))
    parquet_files = glob.glob(os.path.join(dir, '*.parquet'))
    json_files = glob.glob(os.path.join(dir, '*.json'))
    txt_files = glob.glob(os.path.join(dir, '*.txt'))
    
    # Объединяем все файлы
    all_files = csv_files + parquet_files + json_files + txt_files
    
    # Если num_files не указан, загружаем все файлы
    if num_files is None:
        num_files = len(all_files)
    
    # Выбираем нужное количество файлов
    selected_files = all_files[:num_files]
    
    dataframes = []
    print(f"Загружаем {num_files} файлов из {len(all_files)}")
    for file_path in tqdm(selected_files):
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            elif file_path.endswith('.txt'):
                df = pd.read_csv(file_path, sep='|')
            else:
                print(f"Пропуск файла с неподдерживаемым расширением: {file_path}")
                continue
                
            # Добавляем DataFrame только если он был успешно загружен
            dataframes.append(df)
        except Exception as e:
            print(f"Ошибка при чтении файла {file_path}: {str(e)}")
            continue
    
    return dataframes

def data_preparation(files, subscribers, client_df, plan_df, psxattrs_df, id_p=None, id_j=None):
    """
    Подготавливает данные для обучения, объединяя несколько датафреймов сессий и добавляя дополнительную информацию.

    Параметры:
    -----------
    files : list
        Список путей к CSV файлам, содержащим данные сессий с колонками:
        IdSubscriber, Duration, UpTx, DownTx, StartSession, IdSession.
    subscribers : pandas.DataFrame
        DataFrame с информацией о подписчиках, содержащий колонки:
        IdClient, Status, IdOnPSX (соответствует IdSubscriber в файлах сессий).
    client_df : pandas.DataFrame
        DataFrame с информацией о клиентах, содержащий колонки:
        Id, IdPlan.
    plan_df : pandas.DataFrame
        DataFrame с информацией о планах, содержащий колонки:
        Id (соответствует IdPlan в client_df), Enabled, Attrs.
    psxattrs_df : pandas.DataFrame
        DataFrame с атрибутами PSX, содержащий колонки:
        DateFormat, TZ, TransmitUnits, Delimiter, Id (соответствует IdPSX).
    id_p : list, optional
        Список ID клиентов, которые будут помечены типом 'P'.
    id_j : list, optional
        Список ID клиентов, которые будут помечены типом 'J'.
        
    Возвращает:
    --------
    pandas.DataFrame
        Объединенный DataFrame с колонками из всех источников.
    """

    dataframes = []
    for file in files:
        dataframes.append(file)
    
    combined_sessions = pd.concat(dataframes, ignore_index=True)
    
    result = combined_sessions.merge(
        subscribers,
        left_on='IdSubscriber',
        right_on='IdOnPSX',
        how='left'
    )
    
    result = result.merge(
        client_df,
        left_on='IdClient',
        right_on='Id',
        how='left',
        suffixes=('', '_client')
    )
    
    result = result.merge(
        plan_df,
        left_on='IdPlan',
        right_on='Id',
        how='left',
        suffixes=('', '_plan')
    )
    
    result = result.merge(
        psxattrs_df,
        left_on='IdPSX',
        right_on='Id',
        how='left',
        suffixes=('', '_psx')
    )
    
    columns_to_drop = [col for col in result.columns if col.endswith(('_client', '_plan', '_psx'))]
    result = result.drop(columns=columns_to_drop)
    result = result.rename(columns={'Duartion': 'Duration'})
    
    if id_p is not None or id_j is not None:
        id_p_set = set(id_p) if id_p is not None else set()
        id_j_set = set(id_j) if id_j is not None else set()
        
        def determine_type(client_id):
            if client_id in id_p_set:
                return 'P'
            elif client_id in id_j_set:
                return 'J'
            else:
                return None
                
        result['Type'] = result['IdClient'].apply(determine_type)
    
    final_columns = [
        'IdClient', 'Status', 'IdSubscriber', 'Duration', 
        'UpTx', 'DownTx', 'StartSession', 'EndSession', 'IdSession',
        'IdPlan', 'Enabled', 'Attrs',
        'DateFormat', 'TZ', 'TransmitUnits', 'Delimiter', 'Type'
    ]
    
    existing_columns = [col for col in final_columns if col in result.columns]
    return result[existing_columns]

def update_client_files(df, output_dir='clients'):
    """
    Группирует данные по IdClient и обновляет/создает индивидуальные файлы клиентов.
    После обновления всех файлов, сортирует данные в каждом файле по EndPoint.

    Параметры:
    -----------
    df : pandas.DataFrame
        DataFrame, содержащий данные клиентов с колонками, включая EndPoint.
    output_dir : str
        Директория, где будут храниться/обновляться файлы клиентов.
        
    Возвращает:
    --------
    dict
        Словарь со статистикой об обновленных и созданных файлах.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {'updated': 0, 'created': 0}
    
    # Группировка данных по клиентам
    grouped = df.groupby('IdClient')
    
    # Этап 1: Обновление или создание файлов без сортировки
    for client_id, client_data in tqdm(grouped):
        file_path = os.path.join(output_dir, f'client_{client_id}.csv')
        
        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path)
            updated_data = pd.concat([existing_data, client_data], ignore_index=True)
            updated_data.to_csv(file_path, index=False)
            stats['updated'] += 1
        else:
            client_data.to_csv(file_path, index=False)
            stats['created'] += 1
    
    # Этап 2: Сортировка всех файлов в папке по EndPoint
    all_client_files = glob.glob(os.path.join(output_dir, 'client_*.csv'))
    for file_path in tqdm(all_client_files):
        try:
            client_df = pd.read_csv(file_path)
            if 'EndPoint' in client_df.columns:
                client_df = client_df.sort_values('EndPoint').reset_index(drop=True)
                client_df.to_csv(file_path, index=False)
        except Exception as e:
            print(f"Ошибка при сортировке файла {file_path}: {e}")
    
    return stats
