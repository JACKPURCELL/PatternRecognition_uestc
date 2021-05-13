
import numpy as np

k = np.zeros((10,10,3))
dim_3 = np.zeros((10,3))

temp_y = np.array([1,2,3,4,5,6,7,8,9,10])
out = np.random.rand(3,10)
layer3 = np.random.rand(10,10)
for i in range(10):# L-2(l)
    dim_3[i] = out.dot(layer3[:,i])
    for j in range(10):
        k[i,j] = dim_3[i].dot(temp_y[j])
print(out)
print(layer3)
print(dim_3)
print(temp_y)
print(k)