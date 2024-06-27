import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Tentando remover o warning

diretorio_dados_idades = [
    'D:/Python Projetos/Visao Computacional 2.0/Dataset/6-20',
    'Dataset/25-30',
    'Dataset/42-48',
    'Dataset/60-98'
]


def carregar_dados(diretorio: list[str]) -> tuple[str, str]:
    imagens = list()
    labels = list()
    # for x, y in enumerate(diretorio):
    #     print(x, ':', y)
    for tag, idade in enumerate(diretorio):
        for file in os.listdir(idade):
            try:
                imagem = cv2.imread(os.path.join(idade, file),
                                    cv2.IMREAD_GRAYSCALE)
                imagem = cv2.resize(imagem, (48, 48))
                imagens.append(imagem)
                labels.append(tag)
            except Exception as e:
                print(
                    f'Erro ao carregar imagem: {os.path.join(idade, file): {e}}')
    return np.array(imagens), np.array(labels)


images, labels = carregar_dados(diretorio_dados_idades)
# print(images, labels)

X_train, X_test, Y_train, Y_test = train_test_split(
    images, labels, test_size=0.3, random_state=42)
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1).astype('float32')/255
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1).astype('float32')/255
print('antes', Y_train)
Y_train = to_categorical(Y_train)
print('pós', Y_train)
Y_test = to_categorical(Y_test)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
          activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3),
          activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, batch_size=15, epochs=20,
          verbose=1, validation_data=(X_test, Y_test))

model.save('previsor_idade_v1_1.0.h5')
