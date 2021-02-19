import numpy as np
import os
import matplotlib.pyplot as plt
from model import *
from keras.layers import Input, add, Multiply, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from datetime import datetime
import cv2

time1= datetime.now()
file_path = "./data/train/"
xpath = os.path.join(file_path, "images")
ypath = os.path.join(file_path, "manual")

data = os.listdir(xpath)
label = os.listdir(ypath)

x_train = np.zeros((len(data), 256, 256))
y_train = np.zeros((len(data), 256, 256))

for idx, imname in enumerate(data):
    print(os.path.join(xpath, imname))
    img = cv2.imread(os.path.join(xpath, imname))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray/255
    x_train[idx, :, :] = gray


for idx, imname in enumerate(label):
    print(os.path.join(ypath, imname))
    img = cv2.imread(os.path.join(ypath, imname))
    y_train[idx, :, :] = img[:,:,0]/255

x_train = np.expand_dims(x_train, axis=3)
y_train = np.expand_dims(y_train, axis=3)


inpt = Input(shape=(256, 256, 1))
print(inpt.shape)

output1 = unet(inpt)

output = Activation('sigmoid')(output1)
model = Model(inpt, output)
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
result = model.fit(x_train, y_train, batch_size=4, epochs=50, validation_split=0.2)
plt.plot(np.arange(len(result.history['accuracy'])), result.history['accuracy'], label='training')
plt.plot(np.arange(len(result.history['val_accuracy'])), result.history['val_accuracy'], label='validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
model.save('acne_Unet.h5')
time2 = datetime.now()

for i in range(4):
    output = model.predict(np.expand_dims(x_train[3+i, :, :, :], axis=0))
    plt.figure()
    plt.subplot(131)
    plt.imshow(np.squeeze(output), cmap="gray")
    plt.subplot(132)
    plt.imshow(y_train[3+i, :, :, 0], cmap="gray")
    plt.subplot(133)
    plt.imshow(x_train[3+i, :, :, 0], cmap="gray")
    plt.show()
