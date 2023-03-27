import pandas as pd
import numpy as np
# Arrat of 4*5
array1_1 = [1, 2, 3, 4, 5]
array1_2 = [2, 3, 4, 5, 6]
array1_3 = [3, 4, 5, 6, 7]
array1_4 = [4, 5, 6, 7, 8]
nparray_1 = np.array([array1_1, array1_2, array1_3, array1_4])
# Arrat of 5*4
array2_1 = [7, 8, 9, 10]
array2_2 = [11, 36, 20, 32]
array2_3 = [73, 48, 79, 56]
array2_4 = [59, 80, 90, 10]
array2_5 = [32, 56, 69, 74]
nparray_2 = np.array([array2_1, array2_2, array2_3, array2_4, array2_5])
print("\nMatrix 1 :-")
print(nparray_1)
print("\nMatrix 2 :-")
print(nparray_2)
# Matrix multiplication
mulMatrixs = np.matmul(nparray_1, nparray_2)
print("\nMatrix multiplication :-")
print(mulMatrixs)
# Elementwise matrix multiplication
print("\nElementwise matrix multiplication :- ")
x = nparray_1*nparray_1
print(x)
# Mean and median of first matrix
meanOfMat_1 = np.mean(nparray_1)
print("\nMedian of the matrix :- ", meanOfMat_1)
medianOfMat_1 = np.median(nparray_1)
print("\nMedian of the matrix :- ", medianOfMat_1)
# Transpose of the matrixs 1
transOfmat_1 = nparray_1.T
print("\nTranspose of the matrix :-")
print(transOfmat_1)
# Numeric centric data
df = pd.read_csv('mtcars.csv')
cars = np.array([df.mpg, df. cyl, df.disp, df.hp, df.drat,
                df.wt, df.qsec, df.vs, df.vs, df.am, df.gear, df.crab])
nparraySelf = cars - np.mean(cars, axis=0)
print(nparraySelf)
