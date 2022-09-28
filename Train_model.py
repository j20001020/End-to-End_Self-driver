# In[]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
import numpy as np
import os, cv2

# In[]
def prepare_data(data):
    imageFile = np.array(list(data['img_name']))  # 從CSV抓資料#imgeFile是整列資料
    image_label = np.array(list(map(int, data['lable'])))  # 對應結果

    image_array = []  # 建立空陣列

    for file_name in imageFile:  # 從整列資料抓取單張圖片

        img = cv2.imread("C:/Users/Admin/Desktop/NCKU/20220520110654/20220520110654/" + file_name)
        img = cv2.resize(img, (80, 80))
        # img = cv2.resize(img, (224, 224))

        image_array.append(img)

    image_array = np.array(image_array)
    image_array = image_array / 255
    image_label = to_categorical(image_label)

    return image_array, image_label

# In[]
# 建立CNN模型
def build_model():
    model = Sequential()

    model.add(Conv2D(filters=64,
                     kernel_size=(5, 5),
                     padding='same',
                     strides=1,
                     input_shape=(224, 224, 3),
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(filters=128,
                     kernel_size=(5, 5),
                     padding='same',
                     strides=1,
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(filters=256,
                     kernel_size=(5, 5),
                     padding='same',
                     strides=1,
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    
    model.add(Conv2D(filters=512,
                     kernel_size=(5, 5),
                     padding='same',
                     strides=1,
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Flatten())
    # model.add(Dropout(0.2))

    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=21, activation='softmax'))  #分類數

    return model

# In[]
df_raw = pd.read_csv("C:/Users/Admin/Desktop/NCKU/20220520110654/"+"csv_change.csv")

# 資料切割(訓練、驗證、測試)
df_train = df_raw[df_raw['Usage'] == 'Training']
df_test = df_raw[df_raw['Usage'] == 'Test']

X_train, y_train = prepare_data(df_train)
X_test, y_test = prepare_data(df_test)

# In[]
CNN_model = build_model()

# In[]
CNN_model.summary()

# In[]
# 訓練模型
CNN_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
CNN_model.fit(x=X_train, y=y_train, validation_split=0.2, epochs=20, batch_size=5)

# In[]
CNN_model.save('car_model.h5')

# In[]
scores = CNN_model.evaluate(X_test[:2000], y_test[:2000])
print(scores[1])