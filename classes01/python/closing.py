import cv2

image=cv2.imread("../../data/images/closing.png",cv2.IMREAD_GRAYSCALE)

if image is None:
    print("文件读取失败")

closingSize=10
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*closingSize+1,2*closingSize+1),(closingSize,closingSize))

imageMorphClosed= cv2.morphologyEx(image,cv2.MORPH_CLOSE,element,iterations=1)

cv2.imshow("Origin", image)
cv2.imshow("Closed", imageMorphClosed)

cv2.waitKey(0)
cv2.destroyAllWindows()