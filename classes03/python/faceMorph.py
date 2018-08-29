#!/usr/bin/python
# 张晓民
#
import sys
import cv2
import dlib
import numpy as np
import faceBlendCommon as fbc



if __name__ == '__main__':

  # 训练模型
  PREDICTOR_PATH = "../../common/resources/shape_predictor_68_face_landmarks.dat"

  # 脸部检测器
  faceDetector = dlib.get_frontal_face_detector()
  # 关键点检测器
  landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

  # 要合成的两张图片
  im1 = cv2.imread("../data/images/yangzi.jpg")
  im2 = cv2.imread("../data/images/zhangyishan.jpg")

  # 检测两张图片的关键点
  points1 = fbc.getLandmarks(faceDetector, landmarkDetector, im1)
  points2 = fbc.getLandmarks(faceDetector, landmarkDetector, im2)

  points1 = np.array(points1)
  points2 = np.array(points2)

  # 对齐
  im1 = np.float32(im1)/255.0
  im2 = np.float32(im2)/255.0

  # 调整输出尺寸
  h = 600
  w = 600

  # 规范坐标系
  imNorm1, points1 = fbc.normalizeImagesAndLandmarks((h, w), im1, points1)
  imNorm2, points2 = fbc.normalizeImagesAndLandmarks((h, w), im2, points2)

  # 求取平均脸.
  pointsAvg = (points1 + points2)/2.0

  # 变形网格
  boundaryPoints = fbc.getEightBoundaryPoints(h, w)
  points1 = np.concatenate((points1, boundaryPoints), axis=0)
  points2 = np.concatenate((points2, boundaryPoints), axis=0)
  pointsAvg = np.concatenate((pointsAvg, boundaryPoints), axis=0)

  rect = (0, 0, w, h)
  dt = fbc.calculateDelaunayTriangles(rect, pointsAvg)

  cv2.namedWindow("Morphed Face", cv2.WINDOW_AUTOSIZE)

  def adjustBlenAlpha(*args):
      global points1, points2, dt, imNorm1, imNorm2
      alpha = float(args[0])/10
      pointsMorph = (1 - alpha) * points1 + alpha * points2
      imOut1 = fbc.warpImage(imNorm1, points1, pointsMorph.tolist(), dt)
      imOut2 = fbc.warpImage(imNorm2, points2, pointsMorph.tolist(), dt)

      imMorph = (1 - alpha) * imOut1 + alpha * imOut2

      cv2.imshow("Morphed Face", imMorph)
      

  cv2.createTrackbar("Alpha","Morphed Face",1,10,adjustBlenAlpha)
  
  cv2.waitKey(0)
  cv2.destroyAllWindows()
