import numpy as np
matrix = [1,2,3]
matrix2 = np.array([[1,2,3],[1,2,3]])
matrix3 = np.array([[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]]])
# print(matrix.shape)
# print(matrix1.shape)
matrix_1d_pading = np.pad(matrix, (2,4), "constant", constant_values = (3,1))
matrix_2d_pading = np.pad(matrix2, ((2,1), (1,3)), "constant", constant_values = ((1,2),(2,1)))
matrix_3d_pading = np.pad(matrix3, ((1,1),(1,2),(2,2)),"constant")
# print(matrix_2d_pading)
# print(matrix_1d_pading)
print(matrix_3d_pading)