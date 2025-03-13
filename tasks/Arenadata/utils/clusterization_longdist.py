from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from data_preparing.feature_selection import selection_for_clusterisation
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min


def process_single_client_file_dbscan(file_path, merge_threshold=0.05, eps=0.1, min_samples=10):
    """
    Функция загружает данные из одного файла, выполняет обработку и обнаружение аномалий
    с использованием DBSCAN, сливает кластеры до двух, если они ближе порогового значения,
    и возвращает датафрейм с метками класса и расстояние между двумя самыми большими кластерами.
    
    Параметры:
      file_path : str
          Путь к файлу клиента.
      merge_threshold : float
          Пороговое значение для слияния близких кластеров.
      eps : float
          Максимальное расстояние между двумя образцами для того, чтобы один считался в пределах другого.
      min_samples : int
          Минимальное количество образцов в окрестности точки для того, чтобы она считалась основной точкой.
    
    Возвращает:
      client : DataFrame
          Датафрейм с метками класса для каждой записи.
      final_distance : float
          Расстояние между двумя самыми большими кластерами.
    """
    try:
        # Загрузка данных
        client = pd.read_csv(file_path)
        
        # Выбор признаков для кластеризации
        features = selection_for_clusterisation(client)
        
        # Масштабирование данных
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        
        # === Обнаружение аномалий с использованием DBSCAN ===
        # Настройка и обучение модели DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(scaled_features)
        
        # Вычисление центроидов кластеров
        unique_labels = set(labels) - {-1}  # Исключаем аномалии
        while len(unique_labels) > 2:
            # Определение размеров кластеров
            cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
            largest_clusters = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)[:2]
            
            # Вычисление центроидов для двух самых больших кластеров
            centroids = np.array([scaled_features[labels == label].mean(axis=0) for label in largest_clusters])
            distances = cdist(centroids, centroids)
            
            # Слияние кластеров только если расстояние меньше порогового значения
            if distances[0, 1] < merge_threshold:
                labels[labels == largest_clusters[1]] = largest_clusters[0]
                unique_labels = set(labels) - {-1}
            else:
                break  # Прекращаем слияние, если ближайшие кластеры слишком далеки
        
        # Обновление меток в датафрейме
        client['cluster'] = labels
        
        # Вычисление расстояния между двумя самыми большими кластерами
        if len(unique_labels) >= 2:
            cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
            largest_clusters = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)[:2]
            centroids = np.array([scaled_features[labels == label].mean(axis=0) for label in largest_clusters])
            final_distance = cdist(centroids, centroids)[0, 1]
        else:
            final_distance = 0.0
        
        return final_distance, client
    
    except Exception:
        return None, None, None
    

def find_closest_point_to_max_mean_cluster(client_df):
    """
    Находит первую самую близкую точку к кластеру с наибольшим средним значением
    из кластера с наименьшим средним значением.

    Параметры:
      client_df : DataFrame
          Датафрейм с метками кластеров и признаками.
      feature_columns : list
          Список названий столбцов, которые используются в качестве признаков.

    Возвращает:
      closest_point : Series
          Строка датафрейма, представляющая ближайшую точку.
    """
    feature_columns = ['AvgPackets', 'AvgUp', 'AvgDown']
    # Вычисление средних значений для каждого кластера
    cluster_means = client_df.groupby('cluster')[feature_columns].mean()
    
    # Определение кластеров с наибольшим и наименьшим средним значением
    max_mean_cluster = cluster_means.mean(axis=1).idxmax()
    min_mean_cluster = cluster_means.mean(axis=1).idxmin()
    
    # Поиск ближайшей точки из кластера с наименьшим средним значением к центроиду кластера с наибольшим средним значением
    min_cluster_points = client_df[client_df['cluster'] == min_mean_cluster][feature_columns].values
    max_cluster_centroid = cluster_means.loc[max_mean_cluster].values.reshape(1, -1)
    
    closest_index, _ = pairwise_distances_argmin_min(min_cluster_points, max_cluster_centroid)
    closest_point = client_df[client_df['cluster'] == min_mean_cluster].iloc[closest_index[0]]
    
    return closest_point
