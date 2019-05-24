# -*- coding: utf-8 -*-
# https://github.com/windandscreen/face_detection
import face_recognition
from PIL import Image, ImageDraw
import datetime


def main():
    image = face_recognition.load_image_file("img/ag.png")

    # 查找图像中所有面部的所有面部特征
    face_landmarks_list = face_recognition.face_landmarks(image)

    for face_landmarks in face_landmarks_list:
        facial_features = [
            'chin',  # 下巴
            'left_eyebrow',  # 左眉毛
            'right_eyebrow',  # 右眉毛
            'nose_bridge',  # 鼻樑
            'nose_tip',  # 鼻尖
            'left_eye',  # 左眼
            'right_eye',  # 右眼
            'top_lip',  # 上嘴唇
            'bottom_lip'  # 下嘴唇
        ]
        pil_image = Image.fromarray(image)
        d = ImageDraw.Draw(pil_image)
        for facial_feature in facial_features:
            d.line(face_landmarks[facial_feature], fill=(255, 255, 255), width=2)
        pil_image.show()
    return 0


if __name__ == '__main__':
    tic = datetime.datetime.now()
    main()
    toc = datetime.datetime.now()
    print(toc - tic)
