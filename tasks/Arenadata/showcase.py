from utils.clusterization_realtime import process_files_in_batches
from utils.main import showcase, detection_realtime


batches = process_files_in_batches(directory='telecom100k/psx', batch_size=6)

print(len(batches))
for i in range(len(batches)):
    logits = detection_realtime(batches[i], update=True, clinets_path='clients_showcase')
    anomalies = showcase(clients_path='clients_showcase')
    anomalies.to_csv(f'clients_shocases_per_time/anomalies{i}.csv', index=False)
    print(logits)
    print('--------------------------------')
    print(f'{i} of {len(batches)} done')
    print('--------------------------------')
