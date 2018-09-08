import cv2
import numpy as np

def warpTriangle(img1, img2, tri1, tri2):
    """
    变换三角形
    """
    # 先求出三角形的包围盒
    r1 = cv2.boundingRect(tri1)
    r2 = cv2.boundingRect(tri2)

    print("tri1:{}".format(tri1))
    print("r1:{}".format(r1))
    print("tri2:{}".format(tri2))
    print("r2:{}".format(r2))

    # 使用包围盒剪裁图形
    img1Cropped = img1[r1[1]:r1[1]+r1[3],r1[0]:r1[0]+r1[2]]

    # 调整坐标系
    tri1Cropped = []
    tri2Cropped = []
    for i in range(0,3):
        tri1Cropped.append(((tri1[0][i][0] - r1[0]),(tri1[0][i][1] - r1[1])))
        tri2Cropped.append(((tri2[0][i][0] - r1[0]),(tri2[0][i][1] - r2[1])))
    
    # 获取变换矩阵
    warpMat = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))

    print("warpMat:{}".format(warpMat))

    # 进行仿射变换
    img2Cropped = cv2.warpAffine(img1Cropped,warpMat,(r2[2],r2[3]),None,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)

    # 进行alpha混合
    mask = np.zeros((r2[3],r2[2],3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri2Cropped),(1.0,1.0,1.0), 16, 0)

    img2Cropped = img2Cropped * mask

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped

if __name__ == '__main__' :
    imgIn = cv2.imread("../data/images/kingfisher.jpg")
    imgOut = 255 * np.ones(imgIn.shape, dtype = imgIn.dtype)

    triIn = np.float32([[[360,50], [60,100], [300,400]]])
    triOut = np.float32([[[400,200], [160,270], [400,400]]])
    warpTriangle(imgIn, imgOut, triIn, triOut)

    color = (255, 150, 0)
    cv2.polylines(imgIn, triIn.astype(int), True, color, 2, cv2.LINE_AA)
    cv2.polylines(imgOut, triOut.astype(int), True, color, 2, cv2.LINE_AA)
  
    cv2.namedWindow("Input")
    cv2.imshow("Input", imgIn)

    cv2.namedWindow("Output")
    cv2.imshow("Output", imgOut)

    cv2.waitKey(0)