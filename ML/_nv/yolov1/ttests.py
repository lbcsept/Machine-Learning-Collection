import torch
x = torch.as_tensor(
        [
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], [11, 12, 13, 14, 15]],
            [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]],
        ]
    )

index = [[0, 0, 1], [1, 2, 0]]
# tensor([[ 6,  7,  8,  9,  0],
#         [11, 12, 13, 14, 15],
#         [16, 17, 18, 19, 20]])
print(x[index])

index_t = torch.as_tensor(index)
# tensor([[ 6,  7,  8,  9,  0],
#         [11, 12, 13, 14, 15],
#         [16, 17, 18, 19, 20]])
x = x.index_select(0, index_t[0])
x = x[torch.arange(x.shape[0]).unsqueeze(-1), index_t[1].unsqueeze(-1)].squeeze()
print(x)

print("=="*20)

x = torch.as_tensor([[1,2,3,4,5], [6,7,8,9,0]])
print(x)
index = [[0, 1, 1], [1, 1, 2]]
print(index)
# tensor([2, 7, 8])
print(x[index])