import os
import sys
import cv2
import random

# 如果不存在文件夹则创建一个
def create_dir(folder):
    try:
        os.makedirs(folder)
    except:
        print('{} 已经存在!'.format(folder))

# 在图片上绘制脸部的包围框
def drawRectangle(im, bbox):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(im, (x1,y1), (x2, y2), (0,255,0), thickness=5, lineType=cv2.LINE_8)

# 在图片上绘制关键特征点
def drawLandmarks(im, parts):
    for i, part in enumerate(parts):
        px, py = part
        cv2.circle(im, (px, py), 1, (0,0,255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(im, str(i+1),(px, py), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,100), 4)

# opencv中字体的比例是偏大的，对整个图片进行放大处理，避免字体叠加在一起
scale = 4
# 随机采样50张照片
numSamples = 100
# 面部关键点数据集的文件路径
fldDir = sys.argv[1]
print("文件路径:{}".format(fldDir))
# 面部关键特征点的额数量
numPoints = sys.argv[2]
print("关键点数量:{}".format(numPoints))

outputDir = os.path.join(fldDir, "output")
outputMirrorDir = os.path.join(outputDir, 'mirror')
outputOriginalDir = os.path.join(outputDir, 'original')
create_dir(outputMirrorDir)
create_dir(outputOriginalDir)

imageNamesFilepath = os.path.join(fldDir, 'image_names.txt')
print("images_names:{}".format(imageNamesFilepath))

if os.path.exists(imageNamesFilepath):
    with open(imageNamesFilepath) as d:
        imageNames = [x.strip() for x in d.readlines()]
else:
    print("文件路径有误")

random.shuffle(imageNames)
imageNamesSampled = imageNames[:numSamples]


# 迭代
for k, imageName in enumerate(imageNamesSampled):
    print("准备处理图片:{}".format(imageName))

    # 创建文件路径
    imagePath = os.path.join(fldDir, imageName)
    im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    # 放大图片
    im = cv2.resize(im,(0,0), fx=scale, fy=scale)

    # 创建脸部的包围框
    rectPath = os.path.splitext(imageName)[0]+'_rect.txt'
    rectPath = os.path.join(fldDir, rectPath)
    # 打开包围框文件并读取里面的内容
    with open(rectPath) as f:
        line = f.readline()
        left, top, width, height = [float(n) for n in line.strip().split()]
        right = left + width
        bottom = top + height
        # 进行放大
        x1, y1, x2, y2 = int(scale*left), int(scale*top), int(scale*right), int(scale*bottom)
        bbox = [x1, y1, x2, y2]
        # 读取特征点的坐标
        pointsPath = os.path.splitext(imagePath)[0] + "_bv" + str(numPoints) + ".txt"
        parts = []
        with open(pointsPath) as g:
            lines = [x.strip() for x in g.readlines()]
            for line in lines:
                left, right = [float(n) for n in line.split()]
                px, py = int(scale*left), int(scale*right)
                parts.append([px,py])
    
    drawRectangle(im, bbox)
    drawLandmarks(im, parts)

    imageBasename = os.path.basename(imagePath)

    if 'mirror' in imageBasename:
        outputImagePath = os.path.join(outputMirrorDir, imageBasename)
    else:
        outputImagePath = os.path.join(outputOriginalDir, imageBasename)
    
    # 保存文件
    cv2.imwrite(outputImagePath, im)

