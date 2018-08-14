import cv2

# 读取文件
image=cv2.imread("../../data/images/sample.jpg")

# 检查是否读取成功
if image is None:
    print("文件读取失败")

# 将图片转换为灰度图片
grayImage=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# 保存结果
cv2.imwrite("imageGray.jpg",grayImage)

# 创建创空用于显示图片
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.namedWindow("gray Image",cv2.WINDOW_AUTOSIZE)

# 显示图像
cv2.imshow("image",image)
cv2.imshow("gray Image",grayImage)

# 等待用户的输入销毁窗框
cv2.waitKey(0)
cv2.destroyAllWindows()