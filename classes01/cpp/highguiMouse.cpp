#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;

Point center, circumFerence;
Mat source;

/**
 * @brief 绘制圆
 * 
 * @param action 动作类型 
 * @param x 鼠标指针的x坐标
 * @param y 鼠标指针的y坐标
 * @param flags 事件标志
 * @param userdata 事件数据
 */
void drawCircle(int action, int x, int y, int flags, void *userdata)
{
    // 当鼠标左键按下
    if(action == EVENT_LBUTTONDOWN)
    {
        center = Point(x,y);
        circle(source,center,1,Scalar(255.255,0),2,CV_AA);
    }
    else if(action == EVENT_LBUTTONUP) //左键放开
    {
        circumFerence=Point(x,y);
        float radius = sqrt(pow(center.x-circumFerence.x,2)+pow(center.y-circumFerence.y,2));

        circle(source,center,radius,Scalar(0,255,0),2,CV_AA);
        imshow("Window",source);
    }
    else if(action ==EVENT_MOUSEMOVE)
    {
        circumFerence=Point(x,y);
        float radius = sqrt(pow(center.x-circumFerence.x,2)+pow(center.y-circumFerence.y,2));

        circle(source,center,radius,Scalar(0,255,0),2,CV_AA);
        imshow("Window",source);
    }
}

int main(void)
{
    source = imread("../../data/images/sample.jpg");

    Mat dummy = source.clone();
    namedWindow("Window");

    setMouseCallback("Window",drawCircle);

    int k =0;
    while(k!=27)
    {
        imshow("Window", source);
        putText(source,"选择中心，按下左键并拖拉花园，按下ESC建退出，按下C键清除",Point(10,30),FONT_HERSHEY_SIMPLEX,0.7,Scalar(255,255,255),2);

        k=waitKey(20) & 0xFF;
        if(k==99)
          dummy.copyTo(source);
    }

    return 0;
}