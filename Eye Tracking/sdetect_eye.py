import cv2
import cv2.data
from keras.models import load_model
import numpy as np

modelo_eye = load_model('eye_model.keras')
treshold = 0.9


def detect_eye(imagem_eye):
    imagem_eye = cv2.resize(imagem_eye, (48, 48))
    imagem_eye = cv2.cvtColor(imagem_eye, cv2.COLOR_BGR2GRAY)
    imagem_eye = np.expand_dims(imagem_eye, axis=0)
    imagem_eye = np.expand_dims(imagem_eye, axis=-1)
    # normalizando os valores
    imagem_eye = imagem_eye.astype("float32")/255

    direita_probabilidade = modelo_eye.predict(imagem_eye)[0][1]
    if direita_probabilidade > treshold:
        print(direita_probabilidade)
        return True
    else:
        return False


cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    if not ret:
        break
    # Transforma o frame em gray
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pega as faces do frame
    faces = cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #
    for (x, y, w, h) in faces:
        regiao_eye = frame[y:y+h, x:x+w]
        movimento = detect_eye(regiao_eye)
        if movimento:
            cv2.putText(frame, "Olhando para os lados!",
                        (x, y-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Ok", (x, y-10),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Detecção', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
