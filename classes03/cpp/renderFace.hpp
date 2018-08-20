/**
 * @brief 绘制检测到的人脸
 * 
 * @file renderFace.hpp
 * @author your name
 * @date 2018-08-20
 */
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>

/**
 * @brief 绘制关键特征点
 * 
 * @param img 要绘制的图片
 * @param points 特征点集
 * @param color 颜色
 * @param radius 半径
 */
void renderFacePoints(cv::Mat &img, const std::vector<cv::Point2f> &points, cv::Scalar color, int radius=3)
{
    for(int i=0;i<points.size();i++)
    {
        cv::circle(img, points[i], radius, color, -1);
    }
}

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
    std::vector<cv::Point2f> points;
    cv::Point point;
    for(int i=start;i<=end;i++)
    {
        point=cv::Point(landmark.part(i).x(), landmark.part(i).y());
        points.push_back(point);
    }
    cv::polylines(img, points, isClosed,cv::Scalar(255,200,0),2,4);
    renderFacePoints(img, points, cv::Scalar(255,0,255));
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

