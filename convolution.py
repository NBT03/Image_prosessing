import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\BaTien\Downloads\anh_the_3x4.jpg")
img = cv2.resize(img, (200, 200))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255
print(img_gray)


# np.random.seed(5)

class Conv2D:
    def __init__(self, input1, kernel_size, stride, padding, Numofkernel):
        self.stride = stride
        self.Numofkernel = Numofkernel
        self.kernel_size = kernel_size
        self.padding = padding
        self.input1 = np.pad(input1, ((padding, padding), (padding, padding)), "constant", constant_values=1)
        self.kernel = np.random.randn(self.Numofkernel, self.kernel_size, self.kernel_size)
        self.results = np.zeros((int((self.input1.shape[1] - self.kernel.shape[2]) / self.stride) + 1,
                                 int((self.input1.shape[0] - self.kernel.shape[1]) / self.stride) + 1, self.Numofkernel))
        # print(self.results, self.results.shape)
        # print(self.kernel)
        # print(self.input1)
        print(self.kernel.shape)
        print(self.results.shape)

    def Get_Roi(self):
        for row in range(int((self.input1.shape[0] - self.kernel.shape[1]) / self.stride + 1)):
            for col in range(int((self.input1.shape[1] - self.kernel.shape[2]) / self.stride + 1)):
                roi = self.input1[row * self.stride: row * self.stride + self.kernel.shape[1],
                      col * self.stride: col * self.stride + self.kernel.shape[2]]
                yield row, col, roi

    def operation(self):
        for layer in range(self.kernel.shape[0]):
            for row, col, roi in self.Get_Roi():
                self.results[row, col, layer] = np.sum(roi * self.kernel[layer,:,:])
        return self.results


class ReLU:
    def __init__(self, input):
        self.input = input
        self.results = np.zeros((self.input.shape[0], self.input.shape[1], self.input.shape[2]))
        print(self.input)

    def operation(self):
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    self.results[row, col, layer] = 0 if self.input[row, col, layer] < 0 else self.input[row, col, layer]
        return self.results
class Maxpolling:
    def __init__(self, input, Maxpolling_size):
        self.Maxpolling_size = Maxpolling_size
        self.input = input
        self.results = np.zeros((int(self.input.shape[0]/self.Maxpolling_size),
                                int(self.input.shape[1]/self.Maxpolling_size),
                                self.input.shape[2]))
    def operation(self):
        for layer in range(self.input.shape[2]):
            for row in range(int(self.input.shape[0]/self.Maxpolling_size)):
                for col in range(int(self.input.shape[1]/self.Maxpolling_size)):
                    self.results[row, col, layer] = np.max(self.input[row * self.Maxpolling_size:row*self.Maxpolling_size + self.Maxpolling_size,
                                                           col * self.Maxpolling_size:col*self.Maxpolling_size + self.Maxpolling_size,
                                                            layer])
        return self.results


conv2d = Conv2D(img_gray, kernel_size=3, stride=1, padding=1,Numofkernel=8).operation()
conv2d_relu = ReLU(conv2d).operation()
conv2d_polling = Maxpolling(conv2d, 3).operation()

for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(conv2d_polling[:,:,i], cmap = "gray")
# img_gray_conv = conv2d.operation()
# conv2d1 = Conv2D(img_gray_conv, 3)
# img_gray_conv1 = conv2d.operation()
# plt.imshow(img_gray_conv1, cmap='gray')
# plt.imshow(img_gray_conv, cmap='gray')

plt.show()
