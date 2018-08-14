#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;
 
int main(void)
{
    // 打开摄像头
    VideoCapture cap(0);

    if(!cap.isOpened())
    {
        cout<<"摄像头打开失败"<<endl;
        return -1;
    }

    // 获取图像的宽度和高度
    int frame_width=cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height=cap.get(CV_CAP_PROP_FRAME_HEIGHT);

    VideoWriter video("output.avi",CV_FOURCC('M','J','P','G'),30,Size(frame_width,frame_height));

    while(cap.isOpened())
    {
        Mat frame;
        cap >> frame;

        if(frame.empty())
            break;

        video.write(frame);

        imshow("Frame",frame);

        // 按下ESC键退出
        char c=(char)waitKey(25);
        if(c==27)
        break;
    }

    cap.release();
    video.release();

    return EXIT_SUCCESS;
}