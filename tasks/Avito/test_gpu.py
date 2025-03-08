import torch
print(f"CUDA доступен: {torch.cuda.is_available()}")
print(f"Количество GPU: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Текущая GPU: {torch.cuda.get_device_name(0)}")