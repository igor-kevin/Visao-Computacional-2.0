import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Tentando remover o warning

olhos_esquerdo = "Eye dataset/left_look"
olhos_direito = "Eye dataset/right_look"


def carregar_dados(diretorio: list[str]) -> tuple[str, str]:
    imagens = list()
    labels = list()
    # for x, y in enumerate(diretorio):
    #     print(x, ':', y)

    for file in os.listdir(diretorio):
        try:
            imagem = cv2.imread(os.path.join(diretorio, file),
                                cv2.IMREAD_GRAYSCALE)
            imagem = cv2.resize(imagem, (48, 48))
            imagens.append(imagem)
            label = 0 if diretorio.endswith("left_look") else 1
            labels.append(label)
        except Exception as e:
            print(
                f'Erro ao carregar imagem: {os.path.join(diretorio, file): {e}}')
    return np.array(imagens), np.array(labels)


esquerdo_imagens, esquerdo_labels = carregar_dados(olhos_esquerdo)
direito_imagens, direito_labels = carregar_dados(olhos_direito)

imagens_combinadas = np.concatenate(
    (esquerdo_imagens, direito_imagens), axis=0)
labels = np.concatenate((esquerdo_labels, direito_labels), axis=0)

imagens_normalizadas = imagens_combinadas.reshape(
    -1, 48, 48, 1).astype("float32")/255
labels_binarios = to_categorical(labels)

# Separando em teste e treino

X_train, X_test, Y_train, Y_test = train_test_split(
    imagens_normalizadas, labels_binarios, test_size=0.3, random_state=42)

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
model.add(Dense(2, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


model.summary()
model.fit(X_train, Y_train, batch_size=25, epochs=40,
          verbose=1, validation_data=(X_test, Y_test))

model.save('eye_model.keras')
