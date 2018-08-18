#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct userdata
{
    Mat im;
    vector<Point2f> points;
};

void mouseHandler(int event, int x, int y, int flags, void* data_ptr)
{
    if(event == EVENT_LBUTTONDOWN)
    {
        userdata *data=((userdata*) data_ptr);
        circle(data->im, Point(x,y), 3, Scalar(0,0,255),5,CV_AA);
        imshow("Image", data->im);
        if(data->points.size()<4)
        {
            data->points.push_back(Point2f(x,y));
        }
    }
}

int main(void)
{
    Mat im_src=imread("../../data/images/book1.jpg");

    Size size(300,400);
    Mat im_dst=Mat::zeros(size,CV_8UC3);

    vector<Point2f> pts_dst;

    pts_dst.push_back(Point2f(0,0));
    pts_dst.push_back(Point2f(size.width-1,0));
    pts_dst.push_back(Point2f(size.width-1,size.height-1));
    pts_dst.push_back(Point2f(0,size.height-1));

    Mat im_temp = im_src.clone();
    userdata data;
    data.im=im_temp;

    imshow("Image",im_temp);
    setMouseCallback("Image",mouseHandler,&data);
    waitKey(0);

    Mat h= findHomography(data.points,pts_dst);
    warpPerspective(im_src,im_dst,h,size);

    imshow("Image", im_dst);
    waitKey(0);
    destroyAllWindows();


    return EXIT_SUCCESS;
}