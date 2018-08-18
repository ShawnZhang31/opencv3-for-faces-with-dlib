#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct userdata
{
    Mat im;
    vector<Point2f> points;
};

void mouseHandle(int event, int x, int y, int flags, void* data_ptr)
{
    if(event == EVENT_LBUTTONDOWN)
    {
        userdata *data=((userdata *) data_ptr);
        circle(data->im,Point(x,y),3,Scalar(0,255,255),5,CV_AA);
        imshow("Destionation",data->im);

        if(data->points.size()<4)
        {
            data->points.push_back(Point2f(x,y));
        }
    }
}

int main(void)
{
    // 读入原始图像
    Mat im_src=imread("../../data/images/first-image.jpg");
    imshow("SRC", im_src);

    // 获取原始图像的4个相关的点
    vector<Point2f> pts_src;
    pts_src.push_back(Point2f(0,0));
    pts_src.push_back(Point2f(im_src.cols-1,0));
    pts_src.push_back(Point2f(im_src.cols-1,im_src.rows-1));
    pts_src.push_back(Point2f(0,im_src.rows-1));

    // 读入目标图像
    Mat im_dst=imread("../../data/images/times-square.jpg");
    Mat dst_temp=im_dst.clone();
    imshow("Destionation", dst_temp);
    userdata data;
    data.im=dst_temp;
    setMouseCallback("Destionation",mouseHandle,&data);
    waitKey(0);

    Mat h=findHomography(pts_src,data.points);
    
    warpPerspective(im_src,dst_temp,h,dst_temp.size());

    imshow("Wraped",dst_temp);

    // 将两幅图像进行叠加：C=A+B即可,但是需要将目标图像中要合并区域的像素值改为0
    Point pts_dst[4];
    for(int i =0;i<4;i++)
    {
        pts_dst[i]=data.points[i];
    }
    fillConvexPoly(im_dst,pts_dst,4,Scalar(0),CV_AA);

    imshow("Filled", im_dst);

    im_dst += dst_temp;

    // 显示最终结果
    imshow("Finally", im_dst);
    waitKey(0);
    destroyAllWindows();



    return EXIT_SUCCESS;
}