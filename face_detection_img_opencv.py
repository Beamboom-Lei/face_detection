# -*- coding: utf-8 -*-
# https://github.com/windandscreen/face_detection
import datetime
import cv2


def main():
    # 图片路径
    filepath = "img/xingye-1.png"

    # OpenCV人脸识别分类器
    classifier = cv2.CascadeClassifier("opencv_xml/haarcascade_frontalface_default.xml")

    # 读取图片
    img = cv2.imread(filepath)  # 读取图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰色
    color = (0, 255, 0)  # 定义绘制颜色

    # 调用识别人脸
    tic = datetime.datetime.now()
    faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    toc = datetime.datetime.now()
    print(toc - tic)

    if len(faceRects):  # 大于0则检测到人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            # 框出人脸
            cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
            # 左眼
            cv2.circle(img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
            # 右眼
            cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
            # 嘴巴
            cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4), (x + 5 * w // 8, y + 7 * h // 8), color)

    # 绘制图像
    cv2.imshow("image", img)  # 显示图像
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    tic = datetime.datetime.now()
    main()
    toc = datetime.datetime.now()
    print(toc - tic)
