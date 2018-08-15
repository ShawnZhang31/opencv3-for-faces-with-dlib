import cv2

image=cv2.imread("../../data/images/truth.png",cv2.IMREAD_COLOR)

erosionSize = 6
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(erosionSize*2+1,erosionSize*2+1),(erosionSize,erosionSize))
cv2.imshow("element",element)

imageEroded=cv2.erode(image,element)

cv2.imshow("origin",image)
cv2.imshow("Eroded Image",imageEroded)

cv2.waitKey(0)
cv2.destroyAllWindows()