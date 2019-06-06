#include <iostream>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include "faceBlendCommon.hpp"

using namespace std;
using namespace cv;
using namespace dlib;


int main(int argc, char** argv)
{
    // 初始化Dlib资源
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor predictor;
    string model_path = "../../common/resources/shape_predictor_68_face_landmarks.dat";
    deserialize(model_path) >> predictor;

    // 设置一个计时器用于统计代码的运行时间
    double t = (double)cv::getTickCount();

    // 要处理的图片
    string filename1 = "../data/images/ted_cruz.jpg";
    string filename2 = "../data/images/donald_trump.jpg";

    Mat img1 = imread(filename1);
    Mat img2 = imread(filename2);
    Mat img1Warped = img2.clone();

    // 关键点检测
    std::vector<Point2f> points1, points2;
    points1 = getLandmarks(detector, predictor, img1);
    points2 = getLandmarks(detector, predictor, img2);
    cout << "points1:"<<points1.size()<<endl;
    cout << "points2:"<<points2.size()<<endl;

    // 将图片转换为float类型
    img1.convertTo(img1, CV_32F);
    // img1 = img1/255.0f;
    img1Warped.convertTo(img1Warped, CV_32F);
    // img1Warped= img1Warped/255.0f;

    // 获取外轮回
    std::vector<Point2f> hull1;
    std::vector<Point2f> hull2;
    std::vector<int> hullIndex;

    convexHull(points2, hullIndex, false, false);
    for(int i =0; i<hullIndex.size(); i++)
    {
        hull1.push_back(points1[hullIndex[i]]);
        hull2.push_back(points2[hullIndex[i]]);
        cv::circle(img1Warped,points2[hullIndex[i]],1,cv::Scalar(0,255,0));
    }

    // 计算delaunay三角细分
    std::vector<std::vector<int>> dt;
    cv::Rect rect(0, 0, img1Warped.cols, img1Warped.rows);
    calculateDelaunayTriangles(rect, hull2, dt);
    cout<<"dt:"<<dt.size()<<endl;
    for(size_t i=0; i<dt.size(); i++)
    {
        cout<<i<<":"<<dt[i][0]<<"-"<<dt[i][1]<<"-"<<dt[i][2]<<endl;
    }
    // 应用仿射变换
    for(size_t i =0; i < dt.size(); i++)
    {
        std::vector<Point2f> t1, t2;
        for(size_t j=0; j<3; j++)
        {
            t1.push_back(hull1[dt[i][j]]);
            t2.push_back(hull2[dt[i][j]]);
        }
        warpTriangle(img1, img1Warped, t1, t2);
    }
    cout<<"时间开销:"<<((double)cv::getTickCount()-t)/cv::getTickFrequency()<<"秒"<<endl;

    double tClone = (double)cv::getTickCount();

    // 创建无缝拷贝的mask
    std::vector<Point> hull8u;
    for(int i =0; i < hull2.size(); i++)
    {
        Point pt(hull2[i].x, hull2[i].y);
        hull8u.push_back(pt);
    }
    Mat mask = Mat::zeros(img2.rows, img2.cols, img2.depth());
    cv::fillConvexPoly(mask, &hull8u[0], hull8u.size(), Scalar(255, 255, 255));
    // imshow("Mask", mask);
    // cv::waitKey(0);

    // 获取拷贝图片的中心
    Rect r = cv::boundingRect(hull2);
    Point center = (r.tl() + r.br())/2;

    Mat output;
    img1Warped.convertTo(img1Warped, CV_8UC3);
    cv::seamlessClone(img1Warped, img2, mask, center,output, NORMAL_CLONE);

    cout<<"无缝拷贝用时:"<<((double)cv::getTickCount()-tClone)/cv::getTickFrequency()<<"秒"<<endl;
    cout<<"总耗时:"<<((double)cv::getTickCount()-t)/cv::getTickFrequency()<<"秒"<<endl;

    imshow("Face Swapped before seamless cloning", img1Warped);
    imshow("Face Swapped after seamless cloning", output);
    waitKey(0);
    destroyAllWindows();

    return 0;
}

