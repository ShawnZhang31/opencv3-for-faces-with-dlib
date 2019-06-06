/**
 * @brief 绘制脸部关键点
 * 
 * @file drawLandmarks.hpp
 * @author 张晓民
 * @date 2018-08-20
 */
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>





/**
 * @brief 绘制脸部的关键特征点
 * 
 * @param img 需要绘制的图片
 * @param landmarks 关键特征点集合
 */
void drawLandmarks(cv::Mat &img, const dlib::full_object_detection &landmarks)
{
    for(int i=0;i<landmarks.num_parts();i++)
    {
        std::cout<<"检测到关键点数量为:"<<landmarks.num_parts()<<std::endl;
        
        cv::circle(img, cv::Point(landmarks.part(i).x(), landmarks.part(i).y()), 3, cv::Scalar(255,0, 255));
    }
}

