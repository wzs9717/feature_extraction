#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as K
# Helper libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
EAGER = True
 
fashion_mnist = keras.datasets.fashion_mnist
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
 
 
print(train_images.shape,train_labels.shape)
 

train_images = train_images.reshape([-1,28,28,1]) / 255.0
test_images = test_images.reshape([-1,28,28,1]) / 255.0
classes=10
chanDim=-1#TF BK
inputShape=(28, 28, 1)
model = tf.keras.Sequential()
model.add(K.Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
model.add(K.Activation("relu"))
model.add(K.BatchNormalization(axis=chanDim))
model.add(K.Conv2D(32, (3, 3), padding="same"))
model.add(K.Activation("relu"))
model.add(K.BatchNormalization(axis=chanDim))
model.add(K.MaxPooling2D(pool_size=(2, 2)))
model.add(K.Dropout(0.25))

# second CONV => RELU => CONV => RELU => POOL layer set
model.add(K.Conv2D(64, (3, 3), padding="same"))
model.add(K.Activation("relu"))
model.add(K.BatchNormalization(axis=chanDim))
model.add(K.Conv2D(64, (3, 3), padding="same"))
model.add(K.Activation("relu"))
model.add(K.BatchNormalization(axis=chanDim))
model.add(K.MaxPooling2D(pool_size=(2, 2)))
model.add(K.Dropout(0.25))

# first (and only) set of FC => RELU layers
model.add(K.Flatten())
model.add(K.Dense(512))
model.add(K.Activation("relu"))
model.add(K.BatchNormalization())
model.add(K.Dropout(0.5))

# softmax classifier
model.add(K.Dense(classes))
model.add(K.Activation("softmax"))
 
print(model.summary())
model.load_weights("./weights/fashion_minist3.h5")
# lr = 0.001
epochs = 10
checkpoint = keras.callbacks.ModelCheckpoint("./weights/fashion_minist3.h5",
            verbose=0,
            save_best_only=False,
            monitor="val_accuracy",
            mode='max'
        )
opt = tf.keras.optimizers.Adam(lr=1e-5, decay=0)
for i in range(1, 15):
    model.layers[i].trainable = False
# 编译模型
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
 
# 拟合数据
model.fit(train_images, train_labels,
    epochs=epochs,
    validation_data=[test_images[:5000],test_labels[:5000]],
    batch_size=8,
    callbacks=[checkpoint]
    )

# 模型评测
test_loss, test_acc = model.evaluate(test_images, test_labels,verbose=0)
 
print('the model\'s test_loss is {} and test_acc is {}'.format(test_loss, test_acc))
 
#部分预测结果展示
show_images = test_images[:10]
print(show_images.shape)
predictions = model.predict(show_images)
predict_labels = np.argmax(predictions, 1)
 
plt.figure(figsize=(10,5)) #显示前10张图像，并在图像上显示类别
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.grid(False)
    plt.imshow(show_images[i,:,:,0],cmap=plt.cm.binary)
    plt.title(class_names[predict_labels[i]])
 
plt.show()

