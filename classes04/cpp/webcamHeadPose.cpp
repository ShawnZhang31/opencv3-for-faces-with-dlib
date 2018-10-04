#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include "../../classes03/cpp/renderFace.hpp"

using namespace dlib;
using namespace std;
using namespace cv;

#define FACE_DOWNSAMPLE_RATIO 2
#define SKIP_FRAMES 10
#define OPENCV_FACE_RENDER

/**
 * @brief 获取参考点的3D坐标(U,V,W)
 * 
 * @return std::vector<cv::Point3d> 
 */
std::vector<cv::Point3d> get3dModelPoints()
{
    std::vector<cv::Point3d> modelPoints;
    modelPoints.push_back(cv::Point3d(0, 0, 0));   //使用OpenCV的POSIT时，第一个点必须是(0,0,0)
    modelPoints.push_back(cv::Point3d(0, -330, -65));
    modelPoints.push_back(cv::Point3d(-225, 170, -135));
    modelPoints.push_back(cv::Point3d(225, 170, -135));
    modelPoints.push_back(cv::Point3d(-150, -150, -125));
    modelPoints.push_back(cv::Point3d(150, -150, -125));

    return modelPoints;
}

/**
 * @brief 获取图片上的2D参照点
 * 
 * @param d full_object_detection 对象检测
 * @return std::vector<cv::Point2d> 
 */
std::vector<cv::Point2d> get2dImagePoints(dlib::full_object_detection &d)
{
    std::vector<cv::Point2d> imagePoints;
    imagePoints.push_back( cv::Point2d(d.part(30).x(), d.part(30).y() ));        // 鼻尖点
    imagePoints.push_back( cv::Point2d(d.part(8).x(), d.part(8).y() ));          // 下巴
    imagePoints.push_back( cv::Point2d(d.part(36).x(), d.part(36).y() ));        // 左眼角点
    imagePoints.push_back( cv::Point2d(d.part(45).x(), d.part(45).y() ));        // 右眼角点
    imagePoints.push_back( cv::Point2d(d.part(48).x(), d.part(48).y() ));        // 左嘴角点
    imagePoints.push_back( cv::Point2d(d.part(54).x(), d.part(54).y() ));        // 右嘴角点

    return imagePoints;
}

/**
 * @brief 获取摄像头的矩阵
 * 
 * @param focal_length 矩阵
 * @param center 图片中心点
 * @return cv::Mat 返回的3x3的矩阵
 */
cv::Mat getCameraMatrix(float focal_length, cv::Point2d center)
{
    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << focal_length, 0 , center.x, 0, focal_length, center.y, 0, 0, 1);
    return cameraMatrix;
}

int main(int argc, char const *argv[])
{
    try
    {
        cv::VideoCapture capture(0);
        if(!capture.isOpened())
        {
            cerr<< "不能打开摄像头"<<endl;
            return EXIT_FAILURE;
        }

        double fps = 30.0f;
        cv::Mat im;

        capture >> im;
        cv::Mat imSmall, imDisplay;

        cv::resize(im, imSmall, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);
        cv::resize(im, imDisplay, cv::Size(), 0.5, 0.5);

        cv::Size size = im.size();

        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        dlib::shape_predictor predictor;

        dlib::deserialize("../../common/resources/zxm_shape_predictor_70_face_landmarks.dat") >> predictor;

        int count = 0;
        double t = (double)cv::getTickCount();

        std::vector<dlib::rectangle> faces;

        while(1)
        {

            if(count == 0)
                t = cv::getTickCount();
            
            capture >> im;

            cv::resize(im, imSmall, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);

            cv_image<bgr_pixel> cimgSmall(imSmall);
            cv_image<bgr_pixel> cimg(im);

            if(count % SKIP_FRAMES == 0)
            {
                faces = detector(cimgSmall);
            }

            std::vector<cv::Point3d> modelPoints = get3dModelPoints();
            std::vector<dlib::full_object_detection> shapes;

            for(unsigned long i = 0; i<faces.size(); i++)
            {
                dlib::rectangle r( (long)(faces[i].left()*FACE_DOWNSAMPLE_RATIO),
                                   (long)(faces[i].top()*FACE_DOWNSAMPLE_RATIO),
                                   (long)(faces[i].right()*FACE_DOWNSAMPLE_RATIO),
                                   (long)(faces[i].bottom()*FACE_DOWNSAMPLE_RATIO)
                                 );
                
                dlib::full_object_detection shape = predictor(cimg, r);
                shapes.push_back(shape);

                renderFace(im, shape);

                std::vector<cv::Point2d> imagePoints  = get2dImagePoints(shape);

                // 摄像头参数
                double focal_length = im.cols;
                cv::Mat cameraMatrix = getCameraMatrix(focal_length, cv::Point2d(im.cols/2, im.rows/2));

                // 假设摄像头没有径向畸变
                cv::Mat distCoeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);

                // 使用solvePnP计算rotation和translation
                cv::Mat rotationVector;
                cv::Mat translationVector;
                cv::solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs, rotationVector, translationVector);

                std::vector<cv::Point3d> noseEndPoint3D;
                std::vector<cv::Point2d> noseEndPoint2D;
                noseEndPoint3D.push_back(cv::Point3d(0,0,1000));
                cv::projectPoints(noseEndPoint3D, rotationVector, translationVector, cameraMatrix, distCoeffs, noseEndPoint2D);

                cv::line(im, imagePoints[0], noseEndPoint2D[0], cv::Scalar(255,0,255), 2);
            }
            
            

            cv::putText(im, cv::format("fps %.2f", fps), cv::Point(50, size.height-50), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 255));

            imDisplay = im;
            cv::resize(im, imDisplay, cv::Size(), 0.5, 0.5);
            cv::imshow("Webcam Head Pose", imDisplay);

            if(count % 15 == 0)
            {
                int k = cv::waitKey(1);
                if(k == 'q' || k==27)
                {
                    break;
                }
            }

            count++;
            if(count == 100)
            {
                t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
                fps = 100.0/t;
                count = 0;
            }
        }
    }
    catch(serialization_error& e)
    {
        cout << "识别模型没有找到"<<endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() <<endl;
    }
    return 0;
}

