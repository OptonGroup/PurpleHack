import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def clusterization(final):
    # 1. Обучение модели Isolation Forest
    model = IsolationForest(contamination=0.12, random_state=42)
    model.fit(final)

    # 2. Получение предсказаний (-1 — аномалия, 1 — нормальная точка)
    predictions = model.predict(final)

    # 3. Добавление колонки с флагом аномалии
    final['anomaly'] = (predictions == -1)
    
    return final
