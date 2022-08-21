import cv2
import numpy as np
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential


def emotion_recognition(image):
    emotion_result = ""
    # Creamos el modelo
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.load_weights('media/service/model.h5')

    # Diccionario que asigna a cada etiqueta una emoción (orden alfabético)
    emotion_dict = {0: "Enojado", 1: "Disgustado", 2: "Asustado", 3: "Feliz", 4: "Neutral", 5: "Triste",
                    6: "Sorprendido"}

    image_array = np.array(image)
    # Cargamos el clasificador
    facecasc = cv2.CascadeClassifier('media/service/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Obtenemos la región de interés (rostro)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        # El modelo realiza la predicción
        prediction = model.predict(cropped_img)

        # Obtenemos la emoción con mayor porcentaje
        maxindex = int(np.argmax(prediction))
        emotion_result = emotion_dict[maxindex]

        # Agregamos el texto de la emoción identificada
        cv2.putText(image_array, emotion_dict[maxindex], (x + 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

    return emotion_result
