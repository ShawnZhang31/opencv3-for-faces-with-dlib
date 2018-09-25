#include <iostream>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace dlib;

#define RESIZE_HEIGHT 360
#define FACE_DOWNSAMPLE_RATIO_DLIB 1.5

#ifndef M_PI
    #define M_PI 3.14159
#endif

Mat barrel(Mat &src, float k)
{
    int w = src.cols;
    int h = src.rows;

    Mat Xd = Mat::zeros(src.size(), CV_32F);
    Mat Yd = Mat::zeros(src.size(), CV_32F);

    float Xu, Yu;
    for (int y=0;y<h;y++)
    {
        for(int x=0;x<w;x++)
        {
            Xu = ((float)x/w)-0.5;
            Yu = ((float)y/h)-0.5;

            float r= sqrt(Xu*Xu+Yu*Yu);

            // 针孔变形
            float rn = std::min((double)r, r+(pow(r,k)-r)*cos(M_PI*r));

            // 对网格进行变形处理
            Xd.at<float>(y,x) = w*(rn*Xu/r + 0.5);
            Yd.at<float>(y,x) = h*(rn*Yu/r + 0.5);
        }
    }
    
    imshow("xd", Xd);
    imshow("yd", Yd);

    Mat dst;
    imshow("src", src);
    cv::remap(src, dst, Xd, Yd, CV_INTER_CUBIC, BORDER_CONSTANT, Scalar(0, 0, 0));
    imshow("dst", dst);
    return dst;
}

int main(int argc, char const *argv[])
{
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor poseModel;
    string modelPath = "../../common/resources/zxm_shape_predictor_70_face_landmarks.dat";
    deserialize(modelPath) >> poseModel;

    float bulgeAmount = 2;
    int radius = 20;
    cout<<"说明:"<<endl<<"./build/bugeyeVideo < bugleAmount default: 2> <radius around eye default : 30>"<<endl;
    if(argc == 2)
    {
        bulgeAmount = atof(argv[1]);
    }
    else if (argc == 3)
    {
        bulgeAmount = atof(argv[1]);
        radius = atoi(argv[2]);
    }

    cv::VideoCapture capture(0);
    if(!capture.isOpened())
    {
        cout<<"摄像头打开失败"<<endl;
        return EXIT_FAILURE;
    }

    Mat src, eyeRegion, output;
    std::vector<dlib::rectangle> faces;
    while(1)
    {
        double timeTotal = (double)cv::getTickCount();
        capture >> src;

        

        int height = src.rows;
        float IMAGE_RESIZE = (float)height/RESIZE_HEIGHT;
        cv::resize(src, src, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);
        cv::Size size = src.size();

        cv::Mat srcSmall;
        cv::resize(src, srcSmall, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO_DLIB, 1.0/FACE_DOWNSAMPLE_RATIO_DLIB);

        cv_image<bgr_pixel> cimg(src);
        cv_image<bgr_pixel> cimgSmall(srcSmall);

        faces = detector(cimgSmall);
        cout<<"<FACE_DOWNSAMPLE_RATIO_DLIB:"<<FACE_DOWNSAMPLE_RATIO_DLIB<<">-检测了脸部区域耗时:"<<((double)cv::getTickCount() - timeTotal)/cv::getTickFrequency()<<endl;
        
        output = src.clone();

        if(!faces.size())
        {
            cout<<"未能检测到脸"<<endl;
            if((cv::waitKey(1) & 0xFF) == 27)
                return EXIT_SUCCESS;
            continue;
        }

        for(int i=0; i<faces.size(); i++)
        {
            dlib::rectangle r(
                                (long)(faces[i].left()*FACE_DOWNSAMPLE_RATIO_DLIB),
                                (long)(faces[i].top()*FACE_DOWNSAMPLE_RATIO_DLIB),
                                (long)(faces[i].right()*FACE_DOWNSAMPLE_RATIO_DLIB),
                                (long)(faces[i].bottom()*FACE_DOWNSAMPLE_RATIO_DLIB)
                             );
            
            full_object_detection landmarks;
            landmarks = poseModel(cimg, r);

            Rect roiEyeRight( (landmarks.part(43).x()-radius),
                              (landmarks.part(43).y()-radius),
                              (landmarks.part(46).x() - landmarks.part(43).x() + 2*radius),
                              (landmarks.part(47).y() - landmarks.part(43).y() + 2*radius));
            Rect roiEyeLeft( (landmarks.part(37).x()-radius),
                              (landmarks.part(37).y()-radius),
                              (landmarks.part(40).x() - landmarks.part(37).x() + 2*radius),
                              (landmarks.part(41).y() - landmarks.part(37).y() + 2*radius));
            
            
            src(roiEyeRight).copyTo(eyeRegion);
            eyeRegion=barrel(eyeRegion, bulgeAmount);
            eyeRegion.copyTo(output(roiEyeRight));

            src(roiEyeLeft).copyTo(eyeRegion);
            eyeRegion=barrel(eyeRegion, bulgeAmount);
            eyeRegion.copyTo(output(roiEyeLeft));
        }

       

        imshow("Bug Eye Demo-BGR", output);

        int k = cv::waitKey(1);
        if(k==27)
            break;
    }

    capture.release();
    cv::destroyAllWindows();

    return 0;
}


