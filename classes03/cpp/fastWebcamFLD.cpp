/**
 * @brief 提升Facial Landmark Detector的识别速度
 * 
 * @file fastWebcamFLD.cpp
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

#define RESIZE_HEIGHT 480
#define SKIP_FRAMES 2
#define OPENCV_FACE_RENDER

int main(void)
{
    // 创建一个命名window来显示结果
    string winName("Fast Facial Landmark Detector");
    cv::namedWindow(winName, cv::WINDOW_NORMAL);

    // 打开摄像头
    cv::VideoCapture cap(0);
    if(!cap.isOpened()) // 检查摄像头是否成功打开
    {
        std::cout<<"摄像头打开失败!"<<std::endl;
        return -1;
    }
    
    std::cout<<"摄像头打开成功！"<<std::endl;
    
    double fps=30.0;

    // 从摄像头获取图像
    cv::Mat im;
    cap >> im;

    // 使用固定高度的图片作为face detector的输入图像
    cv::Mat imSnall, imDisplay;
    float height=im.rows;
    std::cout<<"摄像头原始图像的高度为:"<<height<<std::endl;

    // 对原始图像进行处理
    float RESIZE_SCALE= height/RESIZE_HEIGHT;
    // float RESIZE_SCALE= 1.0;
    cv::Size size=im.size();

    dlib::frontal_face_detector detector=get_frontal_face_detector();
    dlib::shape_predictor predictor;
    dlib::deserialize("../../common/resources/shape_predictor_68_face_landmarks.dat") >> predictor;

    // 初始化计时器
    double t = (double)cv::getTickCount();
    int count=0;

    std::vector<dlib::rectangle> faces;

    while(true)
    {
        if(count==0)
          t=cv::getTickCount();

        // 获取图像
        cap >> im;
        cv::resize(im, imSnall, cv::Size(), 1.0/RESIZE_SCALE, 1.0/RESIZE_SCALE);

        // 将图片从opencv格式更改为dlib格式
        dlib::cv_image<bgr_pixel> cimSmall(imSnall);
        dlib::cv_image<bgr_pixel> cimg(im);

        if(count % SKIP_FRAMES ==0)
        {
            faces=detector(cimSmall);
        }

        // 获取每张脸的特征点
        std::vector<dlib::full_object_detection> shapes;
        for(int i=0;i<faces.size();i++)
        {
            dlib::rectangle r((long)(faces[i].left()*RESIZE_SCALE),(long)(faces[i].top()*RESIZE_SCALE),(long)(faces[i].right()*RESIZE_SCALE),(long)(faces[i].bottom()*RESIZE_SCALE));

            // 找出每张脸的shape
            dlib::full_object_detection shape=predictor(cimg,r);
            shapes.push_back(shape);

            // 绘制特征点
            renderFace(im, shape);
        }

        // 显示帧率
        cv::putText(im, cv::format("fps %.2f", fps), cv::Point(50, size.height-50),cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0,0,255),3);

        cv::imshow(winName, im);

        char key=cv::waitKey(1);
        if(key==27)
        {
            return EXIT_SUCCESS;
        }

        count ++;

        if(count ==100)
        {
            t=((double)cv::getTickCount() -t)/cv::getTickFrequency();
            fps=100.0/t;
            count=0;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    



    return EXIT_SUCCESS;
}

