import numpy as np
import os
from keras.models import load_model
from datetime import datetime
import cv2

a = datetime.now()
file_path = "./data/test/"
xpath = os.path.join(file_path, "images")
data = os.listdir(xpath)

x_train = np.zeros((len(data), 256, 256))
org = np.zeros((len(data),256,256,3))

for idx, imname in enumerate(data):
    img = cv2.imread(os.path.join(xpath, imname))
    org[idx, :, :, :] = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255
    x_train[idx, :, :] = gray

x_train = np.expand_dims(x_train, axis=3)


model = load_model('acne_Unet.h5')
#
predict = model.predict(x_train, verbose=1)
print(predict.shape)
for i in range(len(predict)):
    img = predict[i, :, :, 0]*255
    a, thresh = cv2.threshold(img, 35, 255, cv2.THRESH_BINARY)
    cv2.imwrite('./result/predict_%d.png'%i, img)
    cv2.imwrite('./result/original_%d.png' % i, org[i, :, :, :])

