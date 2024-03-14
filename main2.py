import torch
import torch.nn as nn
import numpy as np

# Input Volume, Filter
input_data1 = np.array([
    [0, 0, 0, 1, 0],
    [0, 1, 1, 2, 0],
    [0, 0, 1, 2, 0],
    [0, 1, 1, 0, 0],
    [2, 0, 0, 0, 0],
])

input_data2 = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 2, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
])

input_data3 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 2, 0],
    [0, 0, 0, 0, 0],
])

filter11 = np.array([
  [1, 1, -1],
  [-1, 0, 1],
  [-1, 0, -1],
])

filter12 = np.array([
  [-1, 0, -1],
  [0, 0, 0],
  [1, 0, 1],
])

filter13 = np.array([
  [0, 1, 0],
  [-1, 0, -1],
  [0, -1, 1],
])

filter21 = np.array([
    [0, -1, -1],
    [0, -1, 1],
    [0, -1, 1],
])

filter22 = np.array([
    [1, -1, 0],
    [-1, 0, 0],
    [-1, 0, 0],
])

filter23 = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [-1, 1, 1],
])

# numpy array to torch tensor
input_data1_tensor = torch.tensor(input_data1, dtype=torch.float).unsqueeze(0).unsqueeze(0)
input_data2_tensor = torch.tensor(input_data2, dtype=torch.float).unsqueeze(0).unsqueeze(0)
input_data3_tensor = torch.tensor(input_data3, dtype=torch.float).unsqueeze(0).unsqueeze(0)
filter11_tensor = torch.tensor(filter11, dtype=torch.float).unsqueeze(0).unsqueeze(0)
filter12_tensor = torch.tensor(filter12, dtype=torch.float).unsqueeze(0).unsqueeze(0)
filter13_tensor = torch.tensor(filter13, dtype=torch.float).unsqueeze(0).unsqueeze(0)
filter21_tensor = torch.tensor(filter21, dtype=torch.float).unsqueeze(0).unsqueeze(0)
filter22_tensor = torch.tensor(filter22, dtype=torch.float).unsqueeze(0).unsqueeze(0)
filter23_tensor = torch.tensor(filter23, dtype=torch.float).unsqueeze(0).unsqueeze(0)

# convolutional layer
conv2d11 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
conv2d12 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
conv2d13 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
conv2d21 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
conv2d22 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
conv2d23 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)

# reset the convolutional weight
conv2d11.weight.data = filter11_tensor
conv2d12.weight.data = filter12_tensor
conv2d13.weight.data = filter13_tensor
conv2d21.weight.data = filter21_tensor
conv2d22.weight.data = filter22_tensor
conv2d23.weight.data = filter23_tensor

# convolutional operation
output11 = conv2d11(input_data1_tensor)
output12 = conv2d12(input_data2_tensor)
output13 = conv2d13(input_data3_tensor)
output21 = conv2d21(input_data1_tensor)
output22 = conv2d22(input_data2_tensor)
output23 = conv2d23(input_data3_tensor)

output_volume1 = output11 + output12 + output13 + 1
output_volume2 = output21 + output22 + output23

max_pooling = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
avg_pooling = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)

output_max_polling = max_pooling(output_volume2)
output_avg_polling = avg_pooling(output_volume2)

print("-----output_volume1-----")
print(output_volume1)
print("-----output_volume2-----")
print(output_volume2)
print("-----output_max_polling2-----")
print(output_max_polling)
print("-----output_avg_polling2-----")
print(output_avg_polling)