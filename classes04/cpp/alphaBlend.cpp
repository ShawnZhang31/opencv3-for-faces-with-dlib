#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
 * @brief 使用mutiply方法和add方法进行alpha混合
 * 
 * @param alpha a值
 * @param foreground 前景图 
 * @param background 背景图
 * @param outImage 混合输出
 * @return Mat& 返回alpha混合之后的图片
 */
Mat& blend(Mat& alpha, Mat& foreground, Mat& background, Mat& outImage)
{
    Mat fore, back;
    multiply(alpha, foreground, fore);
    multiply(Scalar::all(1.0)-alpha, background, back);
    add(fore, back, outImage);

    return outImage;
}


/**
 * @brief 使用指针操作alpha混合
 * 
 * @param alpha 
 * @param foreground 
 * @param background 
 * @param outImage 
 * @return Mat& 
 */
Mat& alphaBlendDirectAccess(Mat& alpha, Mat& foreground, Mat& background, Mat& outImage)
{
    int numberOfPixels = foreground.rows * foreground.cols *foreground.channels();
    // reinterpret_cast是C++里的强制类型转换符
    float* fptr = reinterpret_cast<float*>(foreground.data);
    float* bptr = reinterpret_cast<float*>(background.data);
    float* aptr = reinterpret_cast<float*>(alpha.data);
    float* outImagePtr = reinterpret_cast<float*>(outImage.data);

    for(int k=0;k<numberOfPixels;k++)
    {
        *outImagePtr=(*fptr)*(*aptr)+(*bptr)*(1.0-*aptr);
        outImagePtr++;
        fptr++;
        bptr++;
        aptr++;
    }

    return outImage;
}


int main(int argc, char** argv)
{
    Mat foreGroundImage = imread("../data/images/foreGroundAssetLarge.png", IMREAD_UNCHANGED);
    Mat bgra[4];

    // 拆分通道
    split(foreGroundImage, bgra);

    // 将四个通道的数据分别保存
    vector<Mat> foregroundChannels;
    foregroundChannels.push_back(bgra[0]);
    foregroundChannels.push_back(bgra[1]);
    foregroundChannels.push_back(bgra[2]);

    imshow("B", foregroundChannels[0]);
    imshow("G", foregroundChannels[1]);
    imshow("R", foregroundChannels[2]);

    cout<<foreGroundImage.size()<<endl;
    Mat foreground = Mat::zeros(foreGroundImage.size(), CV_8UC3);
    merge(foregroundChannels, foreground);

    // 将alpha通道的数据单独保存
    vector<Mat> alphaChannels;
    alphaChannels.push_back(bgra[3]);
    alphaChannels.push_back(bgra[3]);
    alphaChannels.push_back(bgra[3]);
    Mat alpha = Mat::zeros(foreGroundImage.size(), CV_8UC3);
    merge(alphaChannels, alpha);

    // Mat copyWithMask = Mat::zeros(foreGroundImage.size(), CV_8UC3);
    // foreground.copyTo(copyWithMask, bgra[3]);
    // imshow("CopyWithMask", copyWithMask);
    // imshow("Alpha", bgra[3]>0);

    imshow("foreground", foreground);
    imshow("alpha", alpha);

    // 读取背景图片
    Mat background = imread("../data/images/backGroundLarge.jpg", IMREAD_COLOR);
    imshow("background", background);

    // 转换Mat的类型
    foreground.convertTo(foreground, CV_32FC3);
    background.convertTo(background, CV_32FC3);
    alpha.convertTo(alpha, CV_32FC3, 1.0/255);

    int numOfIterations = 1;

    Mat outImage = Mat::zeros(foreground.size(), foreground.type());

    // 使用普通的mutiply方法和add方法操作
    double t = (double)getTickCount();
    for(int i=0; i<numOfIterations; i++)
    {
        outImage = blend(alpha, foreground, background, outImage);
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout<<"使用普通方法执行alpha混合耗时:"<<t<<endl;
    imshow("normal-outImage", outImage);

    // 使用指针进行操作
    outImage = Mat::zeros(foreground.size(), foreground.type());
    t=(double)getTickCount();
    for(int i=0; i<numOfIterations; i++)
    {
        outImage = alphaBlendDirectAccess(alpha,foreground,background,outImage);
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout<<"使用指针操作执行alpha混合耗时:"<<t<<endl;
    imshow("pointer-outImage", outImage);

    waitKey(0);

    return EXIT_SUCCESS;
}