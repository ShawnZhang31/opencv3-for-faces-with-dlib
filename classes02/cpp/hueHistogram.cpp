#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(void)
{
    // 读取一张图像
    Mat img=imread("../data/images/capsicum.jpg");

    // 将图像转化为HSV颜色空间
    Mat hsvImage;
    cvtColor(img,hsvImage,COLOR_BGR2HSV);

    // 拆分图像的通道
    vector<Mat> channels(3);
    split(hsvImage, channels);

    imshow("Image", img);

    // 初始化参数
    int histSize=180;
    float range[]={0,179};
    const float *ranges[]={range};

    // 计算直方图
    MatND hist;
    calcHist(&channels[0],1,0,Mat(),hist,1,&histSize,ranges);
    cout<<hist<<endl;
    // 绘制直方图
    int hist_w=histSize*15; //直方图的宽度
    int hist_h=400;  //直方图的高度
    int bin_w=cvRound((double)hist_w/histSize);

    // 将直方图构建为一个图片
    Mat histImage(hist_h,hist_w,CV_8UC3,Scalar(255,255,255));
    normalize(hist, hist,0,histImage.rows,NORM_MINMAX,-1,Mat());

    // 绘制x轴
    line(histImage,Point(0,hist_h-35), Point(0,hist_h-25), Scalar(0,0,0), 2, 8, 0);
    putText(histImage, "0", Point(0,hist_h-5), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0,0,0), 1, LINE_AA);

    for(int i=1; i<histSize;i++)
    {
        line(histImage, Point(bin_w*(i-1), hist_h-30-cvRound(hist.at<float>(i-1))),
            Point( bin_w*(i), hist_h - 30 - cvRound(hist.at<float>(i))),
            Scalar(0,0,255),2,8,0);
        
        if(i%2==0)
        {
            char buffer[5];
            sprintf(buffer,"%d",i);
            line(histImage, Point(i*bin_w, hist_h - 35), Point(i*bin_w, hist_h - 25), Scalar(0, 0, 0), 2, 8, 0);
            putText(histImage, buffer, Point(i*bin_w, hist_h-5), FONT_HERSHEY_COMPLEX, .5, Scalar(0,0,0), 1, LINE_AA);
        }
    }


    imshow("Histogram", histImage);

    waitKey(0);
    destroyAllWindows();

    return EXIT_SUCCESS;
}