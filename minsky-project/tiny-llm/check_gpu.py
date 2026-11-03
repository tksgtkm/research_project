"""GPU が PyTorch から使えるか確認する。"""
import torch

print("torch       :", torch.__version__)
print("CUDA build  :", torch.version.cuda)
print("available   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("arch list   :", torch.cuda.get_arch_list())
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f"GPU{i}        : {torch.cuda.get_device_name(i)} (sm_{cap[0]}{cap[1]})")
    x = torch.randn(1024, 1024, device="cuda")
    print("matmul OK   :", (x @ x).shape)
