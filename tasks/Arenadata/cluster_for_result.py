from utils.data_extraction import load_client_dataframes
from utils.main import detection, update_data


files = load_client_dataframes()
stats = update_data(files, clinets_path='clients')

result = detection(clients_path='clients', merge_threshold=0.1, eps=0.25, min_samples=10, min_distance=0.6)
result.to_csv('RESULT.csv', index=False)