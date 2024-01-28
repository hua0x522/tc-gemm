import torch 

def random_data(size):
    l = []
    sign = 1
    for i in range(size):
        l.append(sign * (i % 10) / 10)
        sign = -sign
    return torch.tensor(l).to(torch.half)

m = 2048
n = 2048
k = 2048

A = random_data(m * k)
B = random_data(n * k)

A = A.reshape([m, n]).to('cuda')
B = B.reshape([n, k]).to('cuda')

C = A @ B 

print(C[0, :10])