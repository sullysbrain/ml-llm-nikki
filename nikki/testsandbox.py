
import torch

# Creating a PyTorch tensor (1D)
torch_tensor_1d = torch.tensor([1, 2, 3])
print(torch_tensor_1d)  # Output: tensor([1, 2, 3])

# Creating a 2D PyTorch tensor (Matrix)
torch_tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(torch_tensor_2d)
# Output:
# tensor([[1, 2, 3],
#         [4, 5, 6]])

# Element-wise operation
result_tensor = torch_tensor_1d + 2
print(result_tensor)  # Output: tensor([3, 4, 5])

