#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(void)
{
    // 原始图像
    Mat im_src=imread("../../data/images/book2.jpg");
    vector<Point2f> pts_src;
    pts_src.push_back(Point2f(141, 131));
    pts_src.push_back(Point2f(480, 159));
    pts_src.push_back(Point2f(493, 630));
    pts_src.push_back(Point2f(64, 601));

    // 目标图像
    Mat im_dst=imread("../../data/images/book1.jpg");
    vector<Point2f> pts_dst;
    pts_dst.push_back(Point2f(318, 256));
    pts_dst.push_back(Point2f(534, 372));
    pts_dst.push_back(Point2f(316, 670));
    pts_dst.push_back(Point2f(73, 473));

    // 计算Homography矩阵
    Mat ho=findHomography(pts_src,pts_dst);

    // 输入图像
    Mat im_out;
    warpPerspective(im_src,im_out,ho,im_src.size());

    // 显示图片
    imshow("Source", im_src);
    imshow("Destination", im_dst);
    imshow("Warped Source Image", im_out);

    waitKey(0);
    destroyAllWindows();


}