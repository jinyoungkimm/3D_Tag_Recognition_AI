from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


def load_images_from_alphabat_folder(folder_path):

    images = []
    labels = []

    for filename in os.listdir(folder_path):
        
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('L')  # 이미지를 흑백으로 변환

        # 이미지 크기를 target_size로 변경
        img = img.resize((28,28))

        img = np.array(img)  # 이미지를 numpy 배열로 변환

        label = (filename[0])  # 파일명에서 라벨 추출
        print("filename[0]",filename[0])
        images.append(img)
        labels.append(label)

        images, labels = shuffle(images, labels, random_state=42)

    return np.array(images), np.array(labels)


def load_images_from_numeric_folder(folder_path):

    images = []
    labels = []

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('L')  # 이미지를 흑백으로 변환

        # 이미지 크기를 target_size로 변경
        img = img.resize((28,28))

        img = np.array(img)  # 이미지를 numpy 배열로 변환
        label = (filename[0])  # 파일명에서 라벨 추출
        
        print("filname[0]",filename[0])

        images.append(img)
        labels.append(label)

        images, labels = shuffle(images, labels, random_state=42)

    return np.array(images), np.array(labels)  # 리스트를 numpy.ndarray 타입으로 변경


########알파벳(A~Z, 26개) 데이터 #####

# 데이터 폴더 경로 설정
train_folder_path = './numeric'

# 데이터 불러오기
X_train, y_train = load_images_from_numeric_folder(train_folder_path)


# 데이터를 훈련 세트와 테스트 세트로 분할 (예: 테스트 세트는 전체 데이터의 20%)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.01, random_state=42)


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255.0


# 예시 데이터
# labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

labels = y_train
_labels = y_test
# LabelEncoder를 사용하여 문자열 라벨을 정수로 매핑
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels)

_label_encode = LabelEncoder()
_integer_labels = _label_encode.fit_transform(_labels)

# 정수로 매핑된 라벨을 원-핫 인코딩
y_train = to_categorical(integer_labels)
y_test = to_categorical(_integer_labels)


#print("Original Labels:", labels)
#print("Encoded Labels:", integer_labels)
#print("One-Hot Encoded Labels:\n", y_train)

# CNN Model

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) # 26 : A~Z까지는 총 26개, 10 : 0~9까지는 총 10개


# Model option
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

# Model Optimization
modelPath = "./Numeric_CNN.hdf5"
checkpointer = ModelCheckpoint(filepath=modelPath, monitor='val_loss', verbose=True, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

# Executing Learning
history = model.fit(X_train, y_train, epochs=300, validation_split=0.25, batch_size=200, verbose=1,
                    callbacks=[early_stopping_callback, checkpointer])

# Test Data Accuracy
print("\n Test Accuracy: %.4f" % (model.evaluate(X_train, y_train)[1]))

# 검증셋(테스트셋)과 학습셋의 Error를 저장
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_vloss))
plt.plot(x_len, y_vloss, marker='.', c='red', label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


trained_model = load_model("./Numeric_CNN.hdf5")
loss,accuracy = trained_model.evaluate(X_train, y_train)
print("Test Acccuracy By Saved Model",accuracy)

