#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;
using namespace cv;

int main(void)
{
    // 输入三角形1
    vector<Point2f> tri1;
    tri1.push_back(Point2f(50, 50));
    tri1.push_back(Point2f(180, 140));
    tri1.push_back(Point2f(150, 200));

    // 输出三角形1
    vector<Point2f> tri2;
    tri2.push_back(Point2f(72, 51));
    tri2.push_back(Point2f(246, 129));
    tri2.push_back(Point2f(222, 216));

    // 输出三角形2
    vector<Point2f> tri3;
    tri3.push_back(Point2f(77, 76));
    tri3.push_back(Point2f(260, 219));
    tri3.push_back(Point2f(242, 291));

    // 获取仿射变换的矩阵
    Mat warp=getAffineTransform(tri1,tri2);
    Mat warp2=getAffineTransform(tri1,tri3);

    // 显示矩阵
    cout<<"Warp Matrix 1:\n"<<warp<<endl;
    cout<<"Warp Matrix 2:\n"<<warp2<<endl;     

    return EXIT_SUCCESS;
}