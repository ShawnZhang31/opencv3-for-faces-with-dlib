import cv2
import argparse
import numpy as np

img=cv2.imread("../data/images/capsicum.jpg")
hsvImage=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

hsvImage=np.float32(hsvImage)

H,S,V=cv2.split(hsvImage)

plot_width = 540
plot_height = 400
actualRange = 180
rangeRatio = int(plot_width/actualRange)
bufferHeight = 30

histImage = 255*np.ones((plot_height + bufferHeight, plot_width, 3))
xAxisValues = np.arange(plot_width)

histogram = cv2.calcHist([H*rangeRatio],[0],None,[plot_width],[0,plot_width])

cv2.normalize(histogram, histogram, 0, plot_height, cv2.NORM_MINMAX, -1)

histogram = np.max(histogram)-histogram

points=np.column_stack((xAxisValues, histogram.T[0]))

cv2.polylines(histImage, np.int32([points]), False, (0,0,255), 2, cv2.LINE_AA)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.4
fontWidth = 1
xInterval = 20

cv2.putText(histImage, "0", (0, int(histImage.shape[0] - (bufferHeight/2)) ), font, fontScale, (0, 0, 0), fontWidth)

for i in range(0,plot_width,rangeRatio*xInterval):
  
  # Specify the position ( xval, yval )for putting the point on x axis 
  xval = i - 7  
  yval = int(histImage.shape[0] - (bufferHeight/2))
  cv2.putText(histImage, str( i/rangeRatio ), (xval, yval ), font, fontScale, (0, 0, 0), fontWidth)

cv2.namedWindow("Original Image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Histogram of Hue channel", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image",img)
cv2.imshow("Histogram of Hue channel",histImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

