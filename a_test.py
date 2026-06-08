import torch
import numpy
import math


a = torch.zeros(100)
c = a.clone()
for i in range(0,100):
    a[i] = i
b = a[[10,20]]

d = a.view((10,10))
e = d[1:2]
print(0.95**50)


"""
A = torch.zeros([2,2], dtype=torch.float)
A[0][0] = 1
A[0][1] = 2
# A[1][0] = 5
A[1][1] = 1
A.requires_grad_()
U, S, VT = torch.linalg.svd(A)
B = torch.matmul(U, VT)
B[0][1] = -B[0][1]
B[0][0] = -B[0][0]

loss = torch.linalg.det(B)
loss.backward()

print(U, VT)
print(B)
print(A.grad)
"""

"""
class Young_Moudulous_Map(torch.nn.Module):
    def __init__(self, target, inverted_index, gs_num):
        super(Young_Moudulous_Map, self).__init__()
        self.target = torch.nn.Parameter(target)
        self.inverted_index = inverted_index
        self.gs_num = gs_num
    def forward(self):
        E_out = torch.zeros(self.gs_num, dtype=torch.float)
        for i in range(0, self.gs_num):
            E_out[i] = self.target[self.inverted_index[i]]
        return E_out

op = 10

filling_mask = torch.zeros(10, dtype=torch.bool)
print(filling_mask)

mask = torch.zeros(3, dtype=torch.int8)
mask[0] = 2
mask[1] = 1
mask[2] = 2
target = torch.ones(3)

test = Young_Moudulous_Map(target, mask, 3)

optimizer = torch.optim.SGD(test.parameters(),
    lr=1e-2, # 学习率
    )

tain = torch.zeros((10, *target.shape[1:]))
print(tain.shape)

for i in range(1,10):
    another_target = test.forward()
    loss = another_target[0] * another_target[0] + another_target[1] * another_target[1] + another_target[2] * another_target[2]
    loss.backward()

    torch.matmul()

    print(f"iteration {i}")
    print(test.target)
    print(test.target.grad)

    if i % 2 == 0:
        optimizer.step()
        optimizer.zero_grad()
        print("after stepping:")
        print(test.target.grad)

    

"""