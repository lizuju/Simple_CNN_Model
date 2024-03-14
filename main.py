import numpy as np

def conv2d(input_data, filter, stride, padding):

  input_data_padded = np.pad(input_data, padding, 'constant')

  output_size = (input_data_padded.shape[0] - filter.shape[0] + 2 * padding) // stride + 1

  output_data = np.zeros((output_size, output_size))

  for i in range(output_size):
    for j in range(output_size):
      input_data_region = input_data_padded[i * stride:(i * stride + filter.shape[0]), j * stride:(j * stride + filter.shape[1])]

      output_data[i, j] = np.sum(input_data_region * filter)

  return output_data

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

stride = 1
padding = 0

output_data1_1 = conv2d(input_data1, filter11, stride, padding)
# print(output_data1_1)
output_data1_2 = conv2d(input_data2, filter12, stride, padding)
# print(output_data1_2)
output_data1_3 = conv2d(input_data3, filter13, stride, padding)
# print(output_data1_3)

output_data2_1 = conv2d(input_data1, filter21, stride, padding)
# print(output_data2_1)
output_data2_2 = conv2d(input_data2, filter22, stride, padding)
# print(output_data2_2)
output_data2_3 = conv2d(input_data3, filter23, stride, padding)
# print(output_data2_3)

print("-----output_volume1_1-----")
for i in range(0, 3):
    output_volume1_1 = output_data1_1[i] + output_data1_2[i] + output_data1_3[i] + 1
    print(output_volume1_1)

print("-----output_volume2_1-----")
for i in range(0, 3):
    output_volume2_1 = output_data2_1[i] + output_data2_2[i] + output_data2_3[i]
    print(output_volume2_1)