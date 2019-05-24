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

    # 获取人脸关键点检测器
    predictor = dlib.shape_predictor("dlib_dat/shape_predictor_68_face_landmarks.dat")
    tic = datetime.datetime.now()
    dets = detector(gray, 1)
    toc = datetime.datetime.now()
    print(toc - tic)

    for face in dets:
        shape = predictor(img, face)  # 寻找人脸的68个标定点
        # 遍历所有点，打印出其坐标，并圈出来
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 1, (0, 255, 0), 2)
        cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    tic = datetime.datetime.now()
    main()
    toc = datetime.datetime.now()
    print(toc - tic)
