/**
 * @brief 使用光流法增加追踪的稳定性
 * 
 * @file webcamStabilizedFLDcpp
 * @author 张晓民
 * @date 2018-08-20
 */
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <opencv2/opencv.hpp>
#include "renderFace.hpp"

using namespace dlib;
using namespace std;
using namespace cv;

#define RESIZE_HEIGHT 360
#define SKIP_FRAMES 1
#define OPENCV_FACE_RENDER
#define NUM_FRAMES_FOR_FPS 100

void constrainPoint(cv::Point2f &p, cv::Size sz)
{
    p.x = cv::min(cv::max( (double)p.x,0.0), (double)(sz.width - 1));
    p.y = cv::min(cv::max( (double)p.y,0.0), (double)(sz.height - 1));
}

double interEyeDistance(dlib::full_object_detection &shape)
{
    cv::Point2f leftEyeLeftCorner(shape.part(36).x(), shape.part(36).y());
	cv::Point2f rightEyeRightCorner(shape.part(45).x(), shape.part(45).y());
	double distance = norm(rightEyeRightCorner - leftEyeLeftCorner); 
	return distance;
}

int main(void)
{
    // 创建一个命名window来显示结果
    string winName("Stabilized Facial Landmark Detector");
    cv::namedWindow(winName, cv::WINDOW_NORMAL);

    // 打开摄像头
    cv::VideoCapture cap(0);
    if(!cap.isOpened()) // 检查摄像头是否成功打开
    {
        std::cout<<"摄像头打开失败!"<<std::endl;
        return -1;
    }
    
    std::cout<<"摄像头打开成功！"<<std::endl;
    
    // 设置光流法的参数
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
    cv::Size winSize(101,101);
    double eyeDistance, dotRadius, sigma;
    bool eyeDistanceNotCalculated = true;
    int maxLevel = 5;
    std::vector<uchar> status;
    std::vector<float> err;


    double fps=30.0;

    // 当前帧、上一帧及其灰度版本
    cv::Mat im, imPrev, imGray, imGrayPrev;
    // 图像金字塔
    std::vector<cv::Mat> imGrayPyr, imGrayPrevPyr;

    // 获取第一帧
    cap >> imPrev;

    // 转换成灰度图像
    cv::cvtColor(imPrev, imGrayPrev, cv::COLOR_BGR2GRAY);

    // 构建图像金字塔
    cv::buildOpticalFlowPyramid(imGrayPrev, imGrayPrevPyr, winSize, maxLevel);

    // 获取图像的尺寸
    cv::Size size=imPrev.size();

    // 使用固定高度的图片作为face detector的输入图像
    cv::Mat imSmall, imDisplay;
    float height=imPrev.rows;
    std::cout<<"摄像头原始图像的高度为:"<<height<<std::endl;

    // 对原始图像进行处理
    float RESIZE_SCALE= height/RESIZE_HEIGHT;
    // float RESIZE_SCALE= 1.0;
    

    dlib::frontal_face_detector detector=get_frontal_face_detector();
    dlib::shape_predictor predictor;
    dlib::deserialize("../../common/resources/shape_predictor_68_face_landmarks.dat") >> predictor;

    // 初始化计时器
    double t = (double)cv::getTickCount();
    int count=0;

    std::vector<dlib::rectangle> faces;

    // 创建一些变量来存储光流法需要的数据
    std::vector<cv::Point2f> points, pointsPrev, pointsDetectedCur, pointsDetectedPrev;
    // 初始化这些数据
    for(int k=0; k<predictor.num_parts(); ++k)
    {
        pointsPrev.push_back(cv::Point2f(0,0));
        points.push_back(cv::Point2f(0,0));
        pointsDetectedCur.push_back(cv::Point2f(0,0));
        pointsDetectedPrev.push_back(cv::Point2f(0,0));
    }

    // 第一帧没啥子好处理的
    bool isFirstFrame=true;

    // 是否启用光流法
    bool showStabilized = false;


    while(true)
    {
        if(count==0)
          t=cv::getTickCount();

        // 获取图像
        cap >> im;
        // 转为灰度图像
        cv::cvtColor(im, imGray, cv::COLOR_BGR2GRAY);


        cv::resize(im, imSmall, cv::Size(), 1.0/RESIZE_SCALE, 1.0/RESIZE_SCALE);

        // 将图片从opencv格式更改为dlib格式
        dlib::cv_image<bgr_pixel> cimSmall(imSmall);
        dlib::cv_image<bgr_pixel> cimg(im);

        if(count % SKIP_FRAMES ==0)
        {
            faces=detector(cimSmall);
        }

        if(faces.size()<1) //没有检测到脸的时候不处理
            continue;

        // 获取每张脸的特征点
        std::vector<dlib::full_object_detection> shapes;
        for(int i=0;i<faces.size();i++)
        {
            dlib::rectangle r((long)(faces[i].left()*RESIZE_SCALE),(long)(faces[i].top()*RESIZE_SCALE),(long)(faces[i].right()*RESIZE_SCALE),(long)(faces[i].bottom()*RESIZE_SCALE));

            // 找出每张脸的shape
            dlib::full_object_detection shape=predictor(cimg,r);
            shapes.push_back(shape);

            // 循环处理每一个点
            for(int k=0;k<shape.num_parts();k++)
            {
                if(isFirstFrame)
                {
                    pointsPrev[k].x = pointsDetectedPrev[k].x = shape.part(k).x();
                    pointsPrev[k].y = pointsDetectedPrev[k].y = shape.part(k).y();
                }
                else
                {
                    pointsPrev[k]=points[k];
                    pointsDetectedPrev[k]=pointsDetectedCur[k];
                }

                points[k].x = pointsDetectedCur[k].x=shape.part(k).x();
                points[k].y = pointsDetectedCur[k].y=shape.part(k).y();
            }

            // 计算Sigma的值
            if(eyeDistanceNotCalculated)
            {
                eyeDistance=interEyeDistance(shape);
                winSize = cv::Size(2 * int(eyeDistance/4) + 1,  2 * int(eyeDistance/4) + 1);
                eyeDistanceNotCalculated = false;
                dotRadius = eyeDistance > 100 ? 3 : 2;
                sigma = eyeDistance * eyeDistance / 400;
            }

            // 计算图像金字塔
            cv::buildOpticalFlowPyramid(imGray, imGrayPyr, winSize, maxLevel);
            // 使用光流法预测关键点的位置
            cv::calcOpticalFlowPyrLK(imGrayPrevPyr, imGrayPyr, pointsPrev, points, status, err, winSize, maxLevel, termcrit, 0, 0.0001);

            // 获取权重
            for(int k=0;k<shape.num_parts();k++)
            {
                double n= norm(pointsDetectedPrev[k]-pointsDetectedCur[k]);
                double alpha = exp(-n*n/(sigma*1.0));
                std::cout<<"sigma="<<sigma<<std::endl;
                std::cout<<"alpha="<<alpha<<std::endl;
                points[k]=(1-alpha)*pointsDetectedCur[k]+alpha*points[k];
            }

            if(showStabilized)
            {
                renderFacePoints(im, points, cv::Scalar(255,0,255),dotRadius);
            }
            else
            {
                renderFacePoints(im, points, cv::Scalar(0,255,255), dotRadius);
            }

        }


        // 切换
        char key=cv::waitKey(1);
        if(key==27)
        {
            return EXIT_SUCCESS;
        }
        else if(key==32) // 点击空格键切换
        {
            showStabilized = !showStabilized;
        }

        // 准备下一帧
        imPrev=im.clone();
        imGrayPrev=imGray.clone();
        imGrayPrevPyr=imGrayPyr;
        imGrayPyr= std::vector<cv::Mat>();

        isFirstFrame=false;

        count ++;

        if(count ==100)
        {
            t=((double)cv::getTickCount() -t)/cv::getTickFrequency();
            fps=100.0/t;
            count=0;
        }

        // 显示帧率
        cv::putText(im, cv::format("fps %.2f", fps), cv::Point(50, size.height-50),cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0,0,255),3);
        cv::imshow(winName, im);
    }

    cap.release();
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}

