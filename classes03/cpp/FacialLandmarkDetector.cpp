/**
 * @brief 使用Dlib检测脸部关键特征点
 * 
 * @file facialLandmarkDetector.cpp
 * @author 张晓民
 * @date 2018-08-19
 */
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>

using namespace dlib;
using namespace std;
using namespace cv;

/**
 * @brief 绘制多边形
 * 
 * @param img 绘制的图片
 * @param landmark 关键特征点
 * @param start 开始点的索引
 * @param end 结束点的索引
 * @param isClosed 多边形是否闭合
 */
void drawPolyline(cv::Mat &img, const dlib::full_object_detection &landmark, const int start, const int end, bool isClosed=false)
{
    std::vector<cv::Point> points;
    for(int i=start;i<=end;i++)
    {
        points.push_back(cv::Point(landmark.part(i).x(), landmark.part(i).y()));
    }
    cv::polylines(img, points, isClosed,cv::Scalar(255,200,0),2,4);
}

/**
 * @brief 绘制脸部的关键特征点
 * 
 * @param img 需要绘制的图片
 * @param landmarks 关键特征点集合
 */
void renderFace(cv::Mat &img, const dlib::full_object_detection &landmarks)
{
    drawPolyline(img, landmarks, 0, 16);           // 下巴线
    drawPolyline(img, landmarks, 17, 21);          // 左眉毛
    drawPolyline(img, landmarks, 22, 26);          // 右眉毛
    drawPolyline(img, landmarks, 27, 30);          // 鼻梁
    drawPolyline(img, landmarks, 30, 35, true);    // 鼻尖
    drawPolyline(img, landmarks, 36, 41, true);    // 左眼圈
    drawPolyline(img, landmarks, 42, 47, true);    // 右眼圈
    drawPolyline(img, landmarks, 48, 59, true);    // 外嘴唇
    drawPolyline(img, landmarks, 60, 67, true);    // 内嘴唇
}

/**
 * @brief 绘制关键特征点
 * 
 * @param img 要绘制的图片
 * @param points 特征点集
 * @param color 颜色
 * @param radius 半径
 */
void renderFace(cv::Mat &img, const std::vector<cv::Point2f> &points, cv::Scalar color, int radius=3)
{
    for(int i=0;i<points.size();i++)
    {
        cv::circle(img, points[i], radius, color, -1);
    }
}


int main(int argc, char const *argv[])
{
    // 获取脸部检测器
    frontal_face_detector faceDetector = get_frontal_face_detector();

    // 关键特征点检测器
    shape_predictor landmarkDetector;

    // 加载面部关键特征点模型
    deserialize("../../common/resources/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

    // 读取图片
    cv::Mat im= cv::imread("../data/images/family.jpg");

    // 将OpenCV的图片格式转为Dlib的图片格式
    cv_image<bgr_pixel> dlibIm(im);

    // 检测图片中的脸
    std::vector<dlib::rectangle> faceRects=faceDetector(dlibIm);

    cout<<"检测到了 "<<faceRects.size()<<" 张脸"<<endl;

    // 对每个区域进行循环获取每张脸
    std::vector<dlib::full_object_detection> landmarksAll; /* 存储检测到的所有的关键点 */
    for(int i=0;i<faceRects.size();i++)
    {
        dlib::full_object_detection landmarks=landmarkDetector(dlibIm, faceRects[i]);
        std::cout<<"脸"<<i<<"检测到的特征点为:"<<landmarks.num_parts()<<std::endl;
        landmarksAll.push_back(landmarks);
    }

    // 绘制检测点
    for(int i=0;i<landmarksAll.size();i++)
    {
        renderFace(im, landmarksAll[i]);
    }

    cv::imshow("Detection", im);

    cv::waitKey(0);
    cv::imwrite("Detection.jpg", im);
    cv::destroyAllWindows();
    

    /* code */
    return 0;
}

