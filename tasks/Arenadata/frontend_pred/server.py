from flask import Flask, jsonify, send_from_directory
import pandas as pd
import os
import glob

app = Flask(__name__, static_folder='.')

# Маршрут для обслуживания основной HTML страницы
@app.route('/')
def index():
    return send_from_directory('.', 'dashboard.html')

# Маршрут для обслуживания статических файлов
@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

# Маршрут для получения подтвержденных аномалий
@app.route('/api/confirmed-anomalies')
def get_confirmed_anomalies():
    try:
        # Проверяем существование файла already_hacked.csv
        if os.path.exists('already_hacked.csv'):
            df = pd.read_csv('already_hacked.csv')
            
            # Присоединяем дополнительную информацию из файлов клиентов
            result = enrich_anomaly_data(df, 'already_hacked')
            
            return jsonify({
                'status': 'success',
                'data': result.to_dict(orient='records')
            })
        else:
            return jsonify({
                'status': 'success',
                'data': []
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

# Маршрут для получения потенциальных аномалий
@app.route('/api/potential-anomalies')
def get_potential_anomalies():
    try:
        # Проверяем существование файла probably_hacked.csv
        if os.path.exists('probably_hacked.csv'):
            df = pd.read_csv('probably_hacked.csv')
            
            # Присоединяем дополнительную информацию из файлов клиентов
            result = enrich_anomaly_data(df, 'potential')
            
            return jsonify({
                'status': 'success',
                'data': result.to_dict(orient='records')
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Файл с потенциальными аномалиями не найден'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

# Маршрут для получения всех витрин данных
@app.route('/api/showcases')
def get_showcases():
    try:
        # Получаем все файлы с витринами данных
        showcase_files = glob.glob('clients_shocases_per_time/anomalies*.csv')
        
        # Сортируем файлы по номеру
        showcase_files.sort(key=lambda x: int(os.path.basename(x).replace('anomalies', '').replace('.csv', '')))
        
        # Создаем список с данными из всех витрин
        showcases = []
        
        for file in showcase_files:
            if os.path.exists(file):
                df = pd.read_csv(file)
                
                # Получаем номер витрины из имени файла
                showcase_num = os.path.basename(file).replace('anomalies', '').replace('.csv', '')
                
                # Добавляем информацию о витрине
                showcases.append({
                    'id': showcase_num,
                    'name': f'Витрина {showcase_num}',
                    'timestamp': os.path.getmtime(file),
                    'count': len(df),
                    'data': df.to_dict(orient='records')
                })
        
        return jsonify({
            'status': 'success',
            'data': showcases
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

# Вспомогательная функция для обогащения данных об аномалиях
def enrich_anomaly_data(anomalies_df, anomaly_type):
    # Получаем список UID
    uids = anomalies_df['UID'].tolist()
    
    # Создаем список для хранения обогащенных данных
    enriched_data = []
    
    # Создаем базовый DataFrame для результата
    result_df = pd.DataFrame()
    
    # Пытаемся найти файлы клиентов для каждого UID
    for uid in uids:
        client_file = f'clients/client_{uid}.csv'
        if os.path.exists(client_file):
            # Читаем данные клиента
            client_df = pd.read_csv(client_file)
            
            # Если в файле есть данные, берем последнюю запись
            if len(client_df) > 0:
                # Получаем информацию о последней сессии
                last_session = client_df.iloc[-1]
                
                # Создаем запись с обогащенными данными
                enriched_record = {
                    'Id': last_session.get('IdSession', f'S{uid}'),
                    'UID': uid,
                    'Type': last_session.get('Type', 'Unknown'),
                    'IdPlan': last_session.get('IdPlan', 'Basic'),
                    'TurnOn': bool(last_session.get('Enabled', True)),
                    'Hacked': anomaly_type == 'already_hacked',
                    'Traffic': float(last_session.get('AvgPackets', 0)),
                    'status': 'confirmed' if anomaly_type == 'already_hacked' else 'potential'
                }
                
                enriched_data.append(enriched_record)
        else:
            # Если файл клиента не найден, создаем запись с базовой информацией
            enriched_record = {
                'Id': f'S{uid}',
                'UID': uid,
                'Type': 'Unknown',
                'IdPlan': 'Basic',
                'TurnOn': True,
                'Hacked': anomaly_type == 'already_hacked',
                'Traffic': 0.0,
                'status': 'confirmed' if anomaly_type == 'already_hacked' else 'potential'
            }
            
            enriched_data.append(enriched_record)
    
    # Преобразуем обогащенные данные в DataFrame
    if enriched_data:
        result_df = pd.DataFrame(enriched_data)
    
    return result_df

# Маршрут для подтверждения аномалии
@app.route('/api/confirm/<uid>', methods=['POST'])
def confirm_anomaly(uid):
    try:
        # Проверяем, существует ли файл probably_hacked.csv
        if os.path.exists('probably_hacked.csv'):
            # Читаем файл с потенциальными аномалиями
            probably_df = pd.read_csv('probably_hacked.csv')
            
            # Проверяем, существует ли UID в списке потенциальных аномалий
            if uid in probably_df['UID'].values:
                # Удаляем UID из списка потенциальных аномалий
                probably_df = probably_df[probably_df['UID'] != uid]
                probably_df.to_csv('probably_hacked.csv', index=False)
                
                # Проверяем, существует ли файл already_hacked.csv
                if os.path.exists('already_hacked.csv'):
                    # Читаем файл с подтвержденными аномалиями
                    already_df = pd.read_csv('already_hacked.csv')
                    
                    # Добавляем UID в список подтвержденных аномалий, если его там еще нет
                    if uid not in already_df['UID'].values:
                        new_df = pd.DataFrame({'UID': [uid]})
                        already_df = pd.concat([already_df, new_df], ignore_index=True)
                        already_df.to_csv('already_hacked.csv', index=False)
                else:
                    # Создаем новый файл с подтвержденными аномалиями
                    new_df = pd.DataFrame({'UID': [uid]})
                    new_df.to_csv('already_hacked.csv', index=False)
                
                return jsonify({
                    'status': 'success',
                    'message': f'Аномалия с UID {uid} подтверждена'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'UID {uid} не найден в списке потенциальных аномалий'
                })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Файл с потенциальными аномалиями не найден'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

# Маршрут для отклонения аномалии
@app.route('/api/reject/<uid>', methods=['POST'])
def reject_anomaly(uid):
    try:
        # Проверяем, существует ли файл probably_hacked.csv
        if os.path.exists('probably_hacked.csv'):
            # Читаем файл с потенциальными аномалиями
            probably_df = pd.read_csv('probably_hacked.csv')
            
            # Проверяем, существует ли UID в списке потенциальных аномалий
            if uid in probably_df['UID'].values:
                # Удаляем UID из списка потенциальных аномалий
                probably_df = probably_df[probably_df['UID'] != uid]
                probably_df.to_csv('probably_hacked.csv', index=False)
                
                return jsonify({
                    'status': 'success',
                    'message': f'Аномалия с UID {uid} отклонена'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'UID {uid} не найден в списке потенциальных аномалий'
                })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Файл с потенциальными аномалиями не найден'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'True').lower() in ('true', '1', 't')
    app.run(debug=debug_mode, host='0.0.0.0', port=port)