import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
(X_train, Y_train),(X_test, Y_test) = cifar10.load_data()
classes = ['airplane', 'automobile', 'bird', 'cat','deer','dog','frog','horse','ship','truck']
X_train, X_test = X_train/255, X_test/255
Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)
print(X_train.shape)

# for i in range(50):
#     print(i)
#     plt.subplot(5, 10, i + 1)
#     plt.imshow(X_train[200+i])
#     plt.title(classes[Y_train[200+i][0]])
#     plt.axis("off")
# plt.show()
# model_traning_first = models.Sequential([
#     layers.Conv2D(32, (3,3), input_shape = (32,32,3), activation = "relu"),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.15),
#     layers.Conv2D(64, (3,3), activation = "relu"),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.15),
#     layers.Conv2D(128, (3,3), activation = "relu"),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.15),
#      layers.Flatten(input_shape=(32,32,3)),
#      layers.Dense(1000, activation="relu"),
#      layers.Dense(256,activation="relu"),
#      layers.Dense(10,activation="softmax")]
# )
# # model_traning_first.summary()
# model_traning_first.compile(optimizer="SGD",
#                             loss="categorical_crossentropy",
#                             metrics=["accuracy"])

# model_traning_first.fit(X_train,Y_train,epochs=10)
# model_traning_first.save('mode_training_cifar10.h5')
models = models.load_model("mode_training_cifar10.h5")
pred = models.predict(X_test[20].reshape(-1,32,32,3))
print(classes[np.argmax(pred)])
acc = 0
for i in range(100):
  plt.subplot(10, 10, i+1)
  plt.imshow(X_test[i])
  if np.argmax(classes[np.argmax(models.predict(X_test[i].reshape(-1,32,32,3)))])==Y_test[i][0]:
      acc = acc + 1
  plt.title(classes[np.argmax(models.predict(X_test[i].reshape(-1,32,32,3)))])
  plt.axis("off")
print(acc)
plt.show()
