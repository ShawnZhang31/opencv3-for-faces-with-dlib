import cv2

image=cv2.imread("../../data/images/opening.png",cv2.IMREAD_GRAYSCALE)

# 检查是否是有效的输入
if image is None:
    print("读取文件失败")

openingSize=3
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*openingSize+1,2*openingSize+1),(openingSize,openingSize))

imageMorphOpened=cv2.morphologyEx(image,cv2.MORPH_OPEN,element,iterations=3)

cv2.imshow("Original", image)
cv2.imshow("Opening", imageMorphOpened)

# 测试代码
imageMorphClosed=cv2.morphologyEx(image,cv2.MORPH_CLOSE,element,iterations=3)
cv2.imshow("Closed",imageMorphClosed)

cv2.waitKey(0)
cv2.destroyAllWindows()

