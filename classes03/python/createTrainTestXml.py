import sys
import os
import random
try:
    from lxml import etree as ET
except ImportError:
    print("install lxml using pip")
    print("pip install lxml")

# 从标准文件创建XML
def createXml(imageNames, xmlName, numPoints,fldDatadir):
    # 创建一个根节点
    dataset = ET.Element('dataset')
    # 在根节点下创建一个name节点
    ET.SubElement(dataset, "name").text = "Training Faces"
    # 在根节点下创建一个images节点
    images = ET.SubElement(dataset, "images")

    # 创建一个xml文件名
    numFiles = len(imageNames)
    print("{0} : {1} 文件".format(xmlName, numFiles))

    # 迭代所有的图片
    for k, imageName in enumerate(imageNames):
        # 打印进度
        print("{}:{} - {}".format(k+1, numFiles, imageName))
        # 读取图片的rectangle文件
        rect_name = os.path.splitext(imageName)[0] + "_rect.txt"
        with open(os.path.join(fldDatadir,rect_name), "r") as file:
            rect = file.readline()
        rect = rect.split()
        left, top, width, height = rect[0:4]

        # 创建image节点
        image = ET.SubElement(images, "image", file=imageName)
        # 创建box节点
        box = ET.SubElement(image, "box", top=top, left=left, width=width, height=height)

        # 读取points文件
        points_name = os.path.splitext(imageName)[0] + "_bv" + numPoints + ".txt"
        with open(os.path.join(fldDatadir, points_name), "r") as file:
            for i, point in enumerate(file):
                x, y = point.split()

                x = str(int(float(x)))
                y = str(int(float(y)))

                # 点的名称
                name = str(i).zfill(2)
                # 创建part节点
                ET.SubElement(box, "part", name=name, x=x, y=y)
    
    # 最后创建XML tree并保存到硬盘
    tree = ET.ElementTree(dataset)

    # 保存到硬盘
    print("将文件写到硬盘: {}".format(xmlName))
    tree.write(xmlName, pretty_print=True, xml_declaration = True, encoding="UTF-8")

# 入口函数
if __name__ == "__main__":
    fldDatadir = sys.argv[1]
    numPoints = sys.argv[2]

    # 读取所有图片的文件名
    with open(os.path.join(fldDatadir, "image_names.txt")) as d:
        imageNames = [x.strip() for x in d.readlines()]
    

    totalNumFiles = len(imageNames)
    print("图片的数量为:{}".format(totalNumFiles))
    numTestFiles = int(0.05 * totalNumFiles)
    print("测试集的图片数量为:{}".format(numTestFiles))

    # 随机采样5%的图片作为测试集
    testFiles = random.sample(imageNames, numTestFiles)
    # 剩下的作为训练集
    trainFiles = list(set(imageNames) - set(testFiles))

    # 生成XML数据
    createXml(trainFiles, os.path.join(fldDatadir, "zxm_training_with_face_landmarks.xml"), numPoints, fldDatadir)
    createXml(testFiles, os.path.join(fldDatadir, "zxm_testing_with_face_landmarks.xml"), numPoints, fldDatadir)