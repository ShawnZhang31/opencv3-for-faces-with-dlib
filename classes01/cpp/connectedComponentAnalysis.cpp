#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/**
 * @brief 显示图像中的连通区域
 * 
 * @param im 要显示的图像
 */
void displayConnectedComponents(Mat &im);

int main(void)
{
    Mat image = imread("../../data/images/truth.png",IMREAD_GRAYSCALE);

    // 转化为阈值图像
    Mat imThresh;
    threshold(image,imThresh,127,255,THRESH_BINARY);

    // 检测图像中的连通区域
    Mat imLables;
    connectedComponents(imThresh,imLables);

    // 显示连通区域
    displayConnectedComponents(imLables);

    return EXIT_SUCCESS;
}

/**
 * @brief 显示图像中的连通区域
 * 
 * @param im 要显示的图像
 */
void displayConnectedComponents(Mat &im)
{
    Mat imLables = im.clone();

    // 1. 找出区域中的最大值和最小值
    Point minLoc, maxLoc;
    double min, max;

    minMaxLoc(imLables,&min,&max,&minLoc,&maxLoc);

    // 2、归一化：最小值为0，最大值为255
    imLables=255*(imLables-min)/(max-min);

    // 3.将图像转化为8-bits
    imLables.convertTo(imLables,CV_8U);

    // 4.使用颜色映射
    Mat imColorMap;
    applyColorMap(imLables,imColorMap,COLORMAP_JET);

    imshow("Lable",imColorMap);
    waitKey(0);
    destroyAllWindows();
}