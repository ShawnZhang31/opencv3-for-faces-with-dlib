#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(void)
{
    Mat image=imread("../../data/images/mark.jpg");

    // 绘制直线
    Mat imageLine = image.clone();
    line(imageLine,Point(322,179),Point(400,183),Scalar(0,255.0),1,CV_AA);
    imshow("Line",imageLine);

    // 绘制○
    Mat imageCirle = image.clone();
    circle(imageCirle,Point(350,200),150,Scalar(0,255,0),1,CV_AA);
    imshow("Circle",imageCirle);

    // 绘制椭圆
    Mat imageEllipse=image.clone();
    ellipse(imageEllipse,Point(360,200),Size(100,170),45,0,360,Scalar(255,0,0),1,8);
    imshow("Ellipse",imageEllipse);

    // 绘制矩形
    Mat imageRectangle=image.clone();
    rectangle(imageRectangle,Point(208,55),Point(450,355),Scalar(0,255,0),1,8);
    imshow("Rectangle",imageRectangle);

    // 书写文字
    Mat imageText=image.clone();
    putText(imageText,"Mark Zuckerberg", Point(205,50),FONT_HERSHEY_COMPLEX,1,Scalar(0,255,0),1,8);
    circle(imageText,Point(205,50),1,Scalar(0,0,255),1,8);
    imshow("Text",imageText);

    


    

    waitKey(0);

    return EXIT_SUCCESS;
}