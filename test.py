import torch
print("CUDA elérhető:", torch.cuda.is_available())
print("GPU eszköz neve:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nincs GPU")