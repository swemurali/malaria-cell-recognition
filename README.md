# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## DESIGN STEPS

### STEP 1:

Download and load the dataset to colab. After that mount the drive in your colab workspace to access the dataset.

### STEP 2:

Use ImageDataGenerator to augment the data and flow the data directly from the dataset directory to the model.

### STEP 3:

Split the data into train and test.

### STEP 4:

Build the convolutional neural network

### STEP 5:

Train the model with training data

### STEP 6:

Evaluate the model with the testing data

### STEP 7:

Plot the performance plot

## PROGRAM

```
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
from google.colab import drive
import tensorflow as tf
drive.mount('/content/drive')
!tar --skip-old-files -xvf '/content/drive/MyDrive/Dataset/cell_images.tar.xz' -C '/content/drive/MyDrive/Dataset/'
my_data_dir = '/content/drive/MyDrive/Dataset/cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'
os.listdir(train_path)
len(os.listdir(train_path+'/uninfected/'))
len(os.listdir(train_path+'/parasitized/'))
os.listdir(train_path+'/parasitized')[122]
para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[122])
plt.imshow(para_img)
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
sns.jointplot(x=dim1,y=dim2)
image_shape = (130,130,3)
help(ImageDataGenerator)
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
model = models.Sequential()
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(128))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
batch_size = 16

help(image_gen.flow_from_directory)
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')

train_image_gen.batch_size
len(train_image_gen.classes)
train_image_gen.total_batches_seen
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)


train_image_gen.class_indices
results = model.fit(train_image_gen,epochs=20,
                              validation_data=test_image_gen
                             )
model.save('cell_model.h5')
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
model.metrics_names
model.evaluate(test_image_gen)
pred_probabilities = model.predict(test_image_gen)
test_image_gen.classes
predictions = pred_probabilities > 0.5
confusion_matrix(test_image_gen.classes,predictions)
plt.imshow(predictions)
print[predictions]
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![194067076-bf956f1e-9996-462e-97e7-f4427bc20d12](https://user-images.githubusercontent.com/94165336/195599529-2a1e601c-3c9d-498b-ad3c-e7958168bf91.png)

### Classification Report

![194067099-5c583a7d-659f-42d5-a399-3de38d641070](https://user-images.githubusercontent.com/94165336/195599705-76f64f38-5655-4277-9807-5efeb1c20f58.png)

### Confusion Matrix

![194067118-a5bb4f4d-d7af-4a55-817a-9ff8d16cd20d](https://user-images.githubusercontent.com/94165336/195599747-bca32097-612f-4744-8990-f493d405fe58.png)

### New Sample Data Prediction

![194067149-4ba8071f-c626-4673-ad06-4e6813ecfcd2](https://user-images.githubusercontent.com/94165336/195599799-d76f8a12-d9ed-4403-81a2-a549db8b08c6.png)


## RESULT

Successfully developed a convolutional deep neural network for Malaria Infected Cell Recognition.
