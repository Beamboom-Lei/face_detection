# -*- coding: utf-8 -*-
# https://github.com/windandscreen/face_detection
import cv2
import dlib
import datetime


def main():
    filepath = "img/ag.png"

    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 人脸正脸分类器
    detector = dlib.get_frontal_face_detector()

    tic = datetime.datetime.now()
    dets = detector(gray, 1)
    toc = datetime.datetime.now()
    print(toc - tic)

    for face in dets:
        # 在图片中标注人脸，并显示
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imshow("image", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    tic = datetime.datetime.now()
    main()
    toc = datetime.datetime.now()
    print(toc - tic)
