# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:41:39 2024

@author: sevi
"""

from dnet_model import multi_dnet_model
from tensorflow.keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

SIZE_X = 256
SIZE_Y = 256
n_classes = 3

train_imgs = []
for dir_path in glob.glob("data/images/"):
  for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    train_imgs.append(img)

train_imgs = np.array(train_imgs)


mask_imgs = []

for dir_path in glob.glob("data/masks/"):
  for mask_path in glob.glob(os.path.join(dir_path, "*.png")):
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
    mask_imgs.append(mask)

mask_imgs = np.array(mask_imgs)


print(mask_imgs.shape)
print(train_imgs.shape)

from sklearn.preprocessing import LabelEncoder
encodedLabel = LabelEncoder()
n, h, w = mask_imgs.shape 
mask_imgs_reshape = mask_imgs.reshape(-1,1)
mask_imgs_reshape_encoded = encodedLabel.fit_transform(mask_imgs_reshape)
mask_imgs_encoded_original_shape = mask_imgs_reshape_encoded.reshape(n, h, w)


###
np.unique(mask_imgs_encoded_original_shape)




train_imgs = np.expand_dims(train_imgs, axis=3)
train_imgs = normalize(train_imgs, axis=1)

mask_imgs_input = np.expand_dims(mask_imgs_encoded_original_shape, axis=3)
     

print(train_imgs.shape)
print(mask_imgs_input.shape)



from sklearn.model_selection import train_test_split

X1, X_TEST, Y1, Y_TEST = train_test_split(train_imgs, mask_imgs_input, test_size=0.10, random_state=0)



X1_train, X_NONE, Y1_train, Y_NONE = train_test_split(X1, Y1, test_size=0.20, random_state=0)
print("Classes values in the dataset are ", np.unique(Y1_train) , " where 0 is background....")
     


from tensorflow.keras.utils import to_categorical
mask_imgs_cat = to_categorical(Y1_train, num_classes=n_classes)
Y1_train_cat = mask_imgs_cat.reshape(Y1_train.shape[0], Y1_train.shape[1], Y1_train.shape[2], n_classes)


mask_test_cat = to_categorical(Y_TEST, num_classes=n_classes)
Y_TEST_CAT = mask_test_cat.reshape((Y_TEST.shape[0], Y_TEST.shape[1], Y_TEST.shape[2], n_classes))


Y1_train_cat.shape



IMAGE_H = X1_train.shape[1]
IMAGE_W = X1_train.shape[2]
IMAGE_C = X1_train.shape[3]
     


def get_model():
  return multi_dnet_model(n_classes=n_classes, IMG_HEIGHT=IMAGE_H, IMG_WIDTH=IMAGE_W, IMG_CHANNELS=IMAGE_C)


model = get_model()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X1_train, Y1_train_cat, batch_size=16, verbose=1, epochs=250, validation_data=(X_TEST, Y_TEST_CAT), shuffle=False)


model.save('mydnet.hdf5')

_, accur = model.evaluate(X_TEST, Y_TEST_CAT)
print("the accuracy is = ", (accur * 100, "%"))
     
#plot the train and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

y_pred=model.predict(X_TEST)
y_pred_argmax=np.argmax(y_pred, axis=3)

from keras.metrics import MeanIoU
n_classes = 3
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(Y_TEST[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())



# To calculate IoU for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)

class1_IoU = values[0, 0] / (values[0, 0] + values[0, 1] + values[1, 0])
class2_IoU = values[1, 1] / (values[1, 1] + values[1, 0] + values[0, 1])
class3_IoU = values[2, 2] / (values[2, 2] + values[2, 0] + values[0, 2])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)




plt.imshow(train_imgs[0, :,:,0], cmap='gray')

plt.imshow(mask_imgs[0], cmap='gray')

print(len(X_TEST))

import random
test_img_number = random.randint(0, len(X_TEST)-1)
test_img = X_TEST[test_img_number]
ground_truth=Y_TEST[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()
     
test_img_number2 = random.randint(0, len(X_TEST)-1)
test_img2 = X_TEST[test_img_number2]
ground_truth2=Y_TEST[test_img_number2]
test_img_norm2=test_img2[:,:,0][:,:,None]
test_img_input2=np.expand_dims(test_img_norm2, 0)
prediction2 = (model.predict(test_img_input2))
predicted_img2=np.argmax(prediction2, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img2[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth2[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img2, cmap='jet')
plt.show()

test_img_number3 = random.randint(0, len(X_TEST)-1)
test_img3 = X_TEST[test_img_number3]
ground_truth3=Y_TEST[test_img_number3]
test_img_norm3=test_img3[:,:,0][:,:,None]
test_img_input3=np.expand_dims(test_img_norm3, 0)
prediction3 = (model.predict(test_img_input3))
predicted_img3=np.argmax(prediction3, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img3[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth3[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img3, cmap='jet')
plt.show()

test_img_number4 = random.randint(0, len(X_TEST)-1)
test_img4 = X_TEST[test_img_number4]
ground_truth4=Y_TEST[test_img_number4]
test_img_norm4=test_img4[:,:,0][:,:,None]
test_img_input4=np.expand_dims(test_img_norm4, 0)
prediction4 = (model.predict(test_img_input4))
predicted_img4=np.argmax(prediction4, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img4[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth4[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img4, cmap='jet')
plt.show()

test_img_number5 = random.randint(0, len(X_TEST)-1)
test_img5 = X_TEST[test_img_number5]
ground_truth5=Y_TEST[test_img_number5]
test_img_norm5=test_img5[:,:,0][:,:,None]
test_img_input5=np.expand_dims(test_img_norm5, 0)
prediction5 = (model.predict(test_img_input5))
predicted_img5=np.argmax(prediction5, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img5[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth5[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img5, cmap='jet')
plt.show()
