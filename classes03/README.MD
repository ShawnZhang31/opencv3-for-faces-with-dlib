# 脸部关键点点检测
## 步骤
脸部关键点检测需要分为两步进行    
1. **面部检测**:在面部检测和关键点检测的时候最好使用同样的检测器;
2. **关键点检测**:在第一步检测到的矩形区域内进行关键点检测;   
**注意:OpenCV和Dlib使用了不同的面部检测器。OpenCV的面部检测器是基于HAAR或者LBP，Dlib的检测器是基于梯度方向的直方图特征(Histogram of Oriented Gradients features)和支持向量机(SVM:Support Vector Machines)**