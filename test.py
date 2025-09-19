import torch
print(torch.cuda.is_available())  # Должно вернуть True
print(torch.cuda.get_device_name(0))  # Должно показать "NVIDIA RTX 1650"