import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


# if you want to use gpu in the project you can use this command for installation
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
