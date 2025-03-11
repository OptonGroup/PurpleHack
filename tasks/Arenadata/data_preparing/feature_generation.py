import pandas as pd
from datetime import datetime, timezone, timedelta

def process_client_data(df):
    # Если в колонке TransmitUnits указано 'bytes', умножаем UpTx и DownTx на 8
    mask = df['TransmitUnits'] == 'bytes'
    df.loc[mask, 'UpTx'] = df.loc[mask, 'UpTx'] * 8
    df.loc[mask, 'DownTx'] = df.loc[mask, 'DownTx'] * 8

    # Функция для расчёта среднего количества пакетов за длительность
    def compute_avg_packets(row):
        if row['Duration'] <= 0:
            return 0
        return (row['UpTx'] + row['DownTx']) / row['Duration']
    
    # Применяем функцию и создаём столбец 'AvgPackets'
    df['AvgPackets'] = df.apply(compute_avg_packets, axis=1)

    # Функция для расчёта среднего исходящего трафика
    def compute_avg_up(row):
        if row['Duration'] <= 0:
            return 0
        return row['UpTx'] / row['Duration']
    
    # Функция для расчёта среднего входящего трафика
    def compute_avg_down(row):
        if row['Duration'] <= 0:
            return 0
        return row['DownTx'] / row['Duration']
    
    # Создаём столбцы 'AvgUp' и 'AvgDown'
    df['AvgUp'] = df.apply(compute_avg_up, axis=1)
    df['AvgDown'] = df.apply(compute_avg_down, axis=1)

    # Функция для извлечения второго значения из строки в колонке Attrs
    def get_second_attr(row):
        # Если Attrs или Delimiter отсутствуют, возвращаем None
        if pd.isna(row['Attrs']) or pd.isna(row['Delimiter']):
            return None
        parts = row['Attrs'].split(row['Delimiter'])
        # Проверяем, что есть как минимум два элемента
        if len(parts) >= 2:
            return parts[1]
        else:
            return None

    # Добавляем новый столбец с вторым значением
    df['EthernetPlan'] = df.apply(get_second_attr, axis=1)
    df = df.drop(columns=['Attrs', 'TransmitUnits', 'Delimiter'])
    
    return df


def convert_to_utc(date_str, date_format, tz_str):
    """
    Парсит дату по date_format и локализует по временной зоне из tz_str (например, 'GMT-6'),
    затем конвертирует время в UTC.
    """
    if pd.isna(date_str):
        return None
    dt = datetime.strptime(date_str, date_format)
    offset = int(tz_str.replace("GMT", ""))  # извлекаем смещение (например, -6)
    dt = dt.replace(tzinfo=timezone(timedelta(hours=offset)))
    return dt.astimezone(timezone.utc)

def process_session_columns(df):
    """
    1. Добавляет в DataFrame колонки StartSessionUTC и EndSessionUTC (переведённые в UTC).
    2. Создаёт колонку SessionDuration:
       - Если EndSessionUTC != 0, вычисляет разницу (в секундах) между EndSessionUTC и StartSessionUTC.
       - Если EndSessionUTC == 0 (EndSession NaN), берёт исходный Duration.
    3. Удаляет оригинальные колонки StartSession, EndSession, TZ, DateFormat.
    """

    # Преобразуем StartSession в UTC
    df['StartSessionUTC'] = df.apply(
        lambda row: convert_to_utc(row['StartSession'], row['DateFormat'], row['TZ']), axis=1
    )
    # Преобразуем EndSession в UTC или ставим 0
    df['EndSessionUTC'] = df.apply(
        lambda row: convert_to_utc(row['EndSession'], row['DateFormat'], row['TZ'])
        if pd.notna(row['EndSession']) else 0,
        axis=1
    )

    # Функция для вычисления фактической длительности сессии
    def compute_session_duration(row):
        # Если EndSessionUTC == 0, используем исходный Duration
        if row['EndSessionUTC'] == 0 or row['StartSessionUTC'] is None:
            return row['Duration']
        else:
            # Вычисляем разницу во времени (в секундах)
            return (row['EndSessionUTC'] - row['StartSessionUTC']).total_seconds()

    # Добавляем новую колонку с рассчитанной длительностью
    df['SessionDuration'] = df.apply(compute_session_duration, axis=1)

    # Удаляем ненужные колонки
    df = df.drop(columns=['StartSession', 'EndSession', 'TZ', 'Duration'])

    return df


def add_endpoint_column(df):
    """
    Функция принимает DataFrame с колонками:
      - StartSessionUTC: строковое представление даты/времени,
      - DateFormat: формат даты/времени (например, '%Y-%m-%d %H:%M:%S'),
      - duration: целое число секунд.
      
    Функция вычисляет новую колонку endpoint:
      endpoint = (StartSessionUTC + duration) в виде количества секунд с эпохи Unix.
    """
    
    def calc_endpoint(row):
        # Преобразуем StartSessionUTC в datetime с использованием указанного формата
        start_dt = pd.to_datetime(row['StartSessionUTC'], format=row['DateFormat'])
        # Прибавляем к дате duration в секундах
        end_dt = start_dt + pd.Timedelta(seconds=row['SessionDuration'])
        # Преобразуем итоговую дату в Unix timestamp (секунды с 1970-01-01)
        return int(end_dt.timestamp())
    
    # Применяем функцию построчно
    df['EndPoint'] = df.apply(calc_endpoint, axis=1)
    df = df.drop(columns=['EndSessionUTC', 'DateFormat'])
    return df


