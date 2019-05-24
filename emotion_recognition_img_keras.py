# -*- coding: utf-8 -*-
# https://github.com/windandscreen/face_detection
from keras.models import load_model
import numpy as np
import datetime
import cv2
from PIL import Image, ImageDraw


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text((left, top), text, textColor)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def main():
    # loading emotion model
    emotion_classifier = load_model('classifier/emotion_models/simple_CNN.530-0.65.hdf5')
    emotion_labels = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'calm'
    }

    img = cv2.imread("img/emotion/emotion.png")
    face_classifier = cv2.CascadeClassifier("opencv_xml/haarcascade_frontalface_default.xml")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))
    color = (255, 0, 0)

    for (x, y, w, h) in faces:
        gray_face = gray[(y):(y + h), (x):(x + w)]
        gray_face = cv2.resize(gray_face, (48, 48))
        gray_face = gray_face / 255.0
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion = emotion_labels[emotion_label_arg]
        cv2.rectangle(img, (x + 10, y + 10), (x + h - 10, y + w - 10), (255, 255, 255), 2)
        img = cv2ImgAddText(img, emotion, x + h * 0.3, y, color, 20)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    tic = datetime.datetime.now()
    main()
    toc = datetime.datetime.now()
    print(toc - tic)
