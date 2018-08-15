import cv2

imageName="../../data/images/truth.png"
image=cv2.imread(imageName,cv2.IMREAD_COLOR)

if image is None:
    print("图片加载失败")

dilationSize=6
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(2*dilationSize+1,2*dilationSize+1),(dilationSize,dilationSize))

imageDialted=cv2.dilate(image,element)

cv2.imshow("Original",image)
cv2.imshow("Dilated Image",imageDialted)

cv2.waitKey(0)
cv2.destroyAllWindows()

