#include <opencv2/opencv.hpp>
#include <stdlib.h>

using namespace cv;
using namespace std;

/**
 * @brief 
 * 
 * @param img1 
 * @param img2 
 * @param tri1 
 * @param tri2 
 */
void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> tri1, vector<Point2f> tri2);

int main(int argc, char** argv)
{
    // 读取图片
    Mat imgInput = imread("../data/images/kingfisher.jpg", IMREAD_COLOR);

    imgInput.convertTo(imgInput, CV_32FC3, 1/255.0);

    Mat imgOut = Mat::ones(imgInput.size(), imgInput.type());
    imgOut = Scalar(1.0,1.0,1.0);

    // 输入三角形
    vector<Point2f> triIn;
    triIn.push_back(Point2f(360,50));
    triIn.push_back(Point2f(60,10));
    triIn.push_back(Point2f(300,400));

    // 输出三角形
    vector<Point2f> triOut;
    triOut.push_back(Point2f(400,200));
    triOut.push_back(Point2f(160,270));
    triOut.push_back(Point2f(400,400));

    warpTriangle(imgInput, imgOut, triIn, triOut);

    // 绘制CV_AA抗锯齿仅支持CV_8U3图片
    imgInput.convertTo(imgInput, CV_8UC3, 255.0);
    imgOut.convertTo(imgOut, CV_8UC3, 255.0);

    // 绘制线条用的颜色
    Scalar color = Scalar(255, 150, 0);

    // cv::polylines需要Point而不是Point2f
    vector <Point> triInInt, triOutInt;
    for(int i=0; i<3; i++)
    {
        triInInt.push_back(Point(triIn[i].x, triIn[i].y));
        triOutInt.push_back(Point(triOut[i].x, triOut[i].y));
    }

    int lineWidth = 2;
    polylines(imgInput, triInInt, true, color, lineWidth, CV_AA);
    polylines(imgOut, triOutInt, true, color, lineWidth, CV_AA);

    imshow("Input", imgInput);
    imshow("Output", imgOut);

    waitKey(0);

    return EXIT_SUCCESS;
}

void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> tri1, vector<Point2f> tri2)
{
    // 获取包围盒
    Rect r1 = boundingRect(tri1);
    Rect r2 = boundingRect(tri2);

    // 截取图像
    Mat img1Cropped;
    img1(r1).copyTo(img1Cropped);

    imshow("img1Cropped", img1Cropped);

    vector<Point2f> tri1Cropped, tri2Cropped;
    vector<Point> tri2CroppedInt;
    for(int i=0; i<3; i++)
    {
        tri1Cropped.push_back(Point2f(tri1[i].x - r1.x, tri1[i].y - r1.y));
        tri2Cropped.push_back(Point2f(tri2[i].x - r2.x, tri2[i].y - r2.y));

        tri2CroppedInt.push_back(Point((int)(tri2[i].x - r2.x),(int)(tri2[i].y - r2.y)));
    }

    // 获取仿射变换矩阵
    Mat warpMat = getAffineTransform(tri1Cropped, tri2Cropped);

    // 应用仿射变换
    Mat img2Cropped = Mat::zeros(r2.height, r2.width, img1Cropped.type());
    warpAffine(img1Cropped, img2Cropped, warpMat, img2Cropped.size(), INTER_LINEAR, BORDER_REFLECT_101);

    imshow("warpAffine", img2Cropped);

    // 剔除掉三角形以外的像素
    Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
    fillConvexPoly(mask, tri2CroppedInt, Scalar(1.0, 1.0, 1.0), 16, 0);

    imshow("mask", mask);

    multiply(img2Cropped, mask, img2Cropped);
    imshow("mul1", img2Cropped);
    multiply(img2(r2), Scalar(1.0, 1.0, 1.0)-mask, img2(r2));
    imshow("mul2", img2);

    img2(r2) += img2Cropped;
}