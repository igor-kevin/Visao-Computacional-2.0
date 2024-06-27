import cv2
import cv2.data
from keras.models import load_model
import numpy as np


cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
modelo_idade = load_model('previsor_idade_v1_1.0.h5')

IDADES = ['6-20',
          '25-30',
          '42-48',
          '60-98'
          ]


def detect_idade(imagem_face):
    imagem_face = cv2.resize(imagem_face, (48, 48))
    imagem_face = cv2.cvtColor(imagem_face, cv2.COLOR_BGR2GRAY)
    imagem_face = np.reshape(
        imagem_face, [1, imagem_face.shape[0], imagem_face.shape[1], 1])
    predict = np.argmax(modelo_idade.predict(imagem_face))
    return IDADES[predict]


capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_region = frame[y:y+h, x:x+w]
        idade = detect_idade(face_region)
        cv2.putText(frame, idade, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Detecção', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
