import torch
import time

print("Is CUDA available: ", torch.cuda.is_available())


size = 5000

a = torch.rand(size, size)
b = torch.rand(size, size)

torch.device("cpu")
start_cpu = time.time()
for i in range(10):
    result_cpu = torch.mm(a, b)
end_cpu = time.time()

print(f"CPU time: {end_cpu - start_cpu:.4f} seconds")


device = torch.device("cuda:0")
a_cuda = a.to(device)
b_cuda = b.to(device)

# Warm-up (important to avoid first-time CUDA overhead skewing results)
torch.mm(a_cuda, b_cuda)
torch.cuda.synchronize() # Ensure warm-up is complete

start_gpu = time.time()
for i in range(10):
    result_gpu = torch.mm(a_cuda, b_cuda)
torch.cuda.synchronize()
end_cpu = time.time()
print(f"GPU time: {end_cpu - start_gpu:.4f} seconds")
