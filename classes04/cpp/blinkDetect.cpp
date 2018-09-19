#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/plot.hpp>
#include <opencv2/opencv.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>


using namespace std;
using namespace cv;
using namespace dlib;

// 全局参数
int state =0;
double thresh = 0.43; // 判断闭眼的阈值
float normallizeCount =0;
int eyeRegionCount = 0;
string modelPath = "../../common/resources/zxm_shape_predictor_70_face_landmarks.dat"; //形状检测模型
shape_predictor poseModel; //形状检测器
frontal_face_detector detector=dlib::get_frontal_face_detector(); //脸部检测器

Mat frame, eye;
#define RESIZE_HEIGHT 480;
#define FACE_DOWNSAMPLE_RATIO_DLIB 2.5

float blinkTime = 0.2;  // 眨眼的事件限制是200ms
float drowsyTime = 1.0; // 疲劳的时间限制是1000ms
int drowsyLimit = 0;
int falseBlinkLimit = 0;

static int lefteye[]={36,37,38,39,40,41};
std::vector<int> lefteye_index(lefteye, lefteye + sizeof(lefteye)/sizeof(lefteye[0]));
static int righteye[]={42,43,44,45,46,47};
std::vector<int> righteye_index(righteye, righteye + sizeof(righteye)/sizeof(righteye[0]));

/**
 * @brief 检查眼睛的状态是睁开的还是闭起来的
 * 
 * @param landmarks 关键点
 * @return int 0:闭，1:开
 */
int checkEyeStatus(full_object_detection& landmarks)
{
    // 创建一个黑色的图片作为眼睛的遮罩
    Mat mask = Mat::zeros(frame.rows, frame.cols, frame.depth());
    std::vector<Point> hullLeftEye;
    for(int i =0; i < lefteye_index.size(); i++)
    {
        Point pt(landmarks.part(lefteye_index[i]).x(), landmarks.part(lefteye_index[i]).y());
        hullLeftEye.push_back(pt);
    }
    fillConvexPoly(mask, hullLeftEye, cv::Scalar(255, 255, 255));

    std::vector<Point> hullRightEye;
    for(int i =0; i < righteye_index.size(); i++)
    {
        Point pt(landmarks.part(righteye_index[i]).x(), landmarks.part(righteye_index[i]).y());
        hullRightEye.push_back(pt);
    }
    fillConvexPoly(mask, hullRightEye, cv::Scalar(255, 255, 255));

    // 眼睛的长度
    int lenLeftEyeX = landmarks.part(lefteye_index[3]).x() - landmarks.part(lefteye_index[0]).x();
    int lenLeftEyeY = landmarks.part(lefteye_index[3]).y() - landmarks.part(lefteye_index[0]).y();
    float lenLeftEyeSquart = lenLeftEyeX*lenLeftEyeX + lenLeftEyeY*lenLeftEyeY;

    eyeRegionCount = cv::countNonZero(mask == 255);
    
    // 使用眼睛的长度对数值进行归一化
    normallizeCount = (float)eyeRegionCount/lenLeftEyeSquart;

    eye = Mat::zeros(frame.rows, frame.cols, frame.depth());
    frame.copyTo(eye, mask);

    int eyeStatus = 1;

    if(normallizeCount < thresh)
        eyeStatus = 0;

    return eyeStatus;
}

/**
 * @brief 使用FSM追踪用户的眨眼状态，推算用户的行为
 * 
 * @param eyeStatus 眼睛的状态
 * @param blinkCount 眨眼的次数
 * @param drowsy 疲劳
 * @return int 
 */
int checkBlinkStatus(int eyeStatus, int& blinkCount, int& drowsy)
{
    if(state >=0 && state <= falseBlinkLimit)
    {
        // 如果在当前帧眼睛是睁开的，那就继续保持;判断是否是无效的眨眼
        if(eyeStatus)
        {
            state = 0;
        }
        else
        {
            state++;
        }
    }
    else if(state > falseBlinkLimit && state <= drowsyLimit)
    {
        if(eyeStatus)
        {
            state = 0;
            blinkCount++;
            return 1;
        }
        else
        {
            state++;
        }
    }
    else
    {
        if(eyeStatus)
        {
            state = 0;
            blinkCount++;
            drowsy = 0;
            return 1;
        }
        else
        {
            drowsy =1;
        }
    }
    return 0;
}

int main(int argc, char const *argv[])
{
    cout<<"注意: ./blinkDetect <默认的阈值为："<< thresh <<endl;
    
    if(argc == 2) //自定义阈值
    {
        thresh = atof(argv[1]);
    }

    VideoCapture capture;
    deserialize(modelPath) >> poseModel;
    std::vector<dlib::rectangle> faces;

    capture.open(0);

    // 计算帧率
    int blinkCount = 0;
    int drowsy =0;
    double t=0;

    // 首先空跑几帧等待摄像头状态稳定
    for(int i=0; i<50; i++)
        capture.read(frame);
    
    float totalTime = 0.0;
    int validFrames = 0;
    int dummyFrames = 50;
    float spf = 0;
    double timeLandmarks = 0.0;

    while(validFrames < dummyFrames)
    {
        t = (double)cv::getTickCount();

        capture.read(frame);
        validFrames++;
        int height = frame.rows;
        float IMAGE_SIZE =(float)height/RESIZE_HEIGHT;

        cv::resize(frame, frame, cv::Size(), 1.0/IMAGE_SIZE, 1.0/IMAGE_SIZE);
        cv::Size size = frame.size();
        cv::Mat frame_small;
        cv::resize(frame, frame_small, cv::Size(),1.0/FACE_DOWNSAMPLE_RATIO_DLIB, 1.0/FACE_DOWNSAMPLE_RATIO_DLIB);

        cv_image<bgr_pixel> cimg(frame);
        cv_image<bgr_pixel> cimg_small(frame_small);

        // 检测脸部
        faces = detector(cimg_small);

        timeLandmarks = ((double)cv::getTickCount() - t)/cv::getTickFrequency();

        // 检查帧是否有效
        if(!faces.size())
        {
            validFrames--;
            putText(frame,"不能检测到面部，请检查!", cv::Point(10,50),FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,0,255));
            putText(frame,"或者采样的比例设置的太小了!", cv::Point(10,50),FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,0,255));
            imshow("Blink Detection Demo", frame);
            if((waitKey(1)&0xFF) == 27)
            {
                return 0;
            }
        }
        else
            totalTime += timeLandmarks;
    }

    spf = totalTime/dummyFrames;
    cout<<"SPF(每帧用时):"<<spf<<"秒"<<endl;

    /**
     * 使用spf计算眨眼的帧数显示和疲劳的帧数限制，这些参会用于FSM状态检测
     */
    drowsyLimit =int(drowsyTime/spf);
    falseBlinkLimit = int(blinkTime/spf);
    cout<<"疲劳帧数限制:"<<drowsyLimit<<"帧("<<drowsyLimit*spf*1000<<" ms)"<<endl;
    cout<<"假眨眼的帧数限制:"<<falseBlinkLimit<<"帧("<<falseBlinkLimit*spf*1000<<" ms)"<<endl;

    if(!capture.isOpened())
    {
        cout<<"错误：摄像头打开失败!"<<endl;
        return -1;
    }

    while(capture.read(frame))
    {
        if(frame.empty())
        {
            cout<<"警告:未获取到有效的像素"<<endl;
            break;
        }

        int height = frame.rows;
        float IMAGE_RESIZE = (float)height/RESIZE_HEIGHT;
        cv::resize(frame, frame, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);
        cv::Size size = frame.size();

        cv::Mat frame_small;
        cv::resize(frame, frame_small, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO_DLIB, 1.0/FACE_DOWNSAMPLE_RATIO_DLIB);
        cv_image<bgr_pixel> cimg(frame);
        cv_image<bgr_pixel> cimg_small(frame_small);

        faces = detector(cimg_small);
        if(!faces.size())
        {
            putText(frame,"不能检测到面部，请检查!", cv::Point(10,50),FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,0,255));
            putText(frame,"或者采样的比例设置的太小了!", cv::Point(10,50),FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,0,255));
            imshow("Blink Detection Demo", frame);
            if((waitKey(1)&0xFF) == 27)
            {
                return 0;
            }
            continue;
        }
        dlib::rectangle r(
                            (long)(faces[0].left()*FACE_DOWNSAMPLE_RATIO_DLIB),
                            (long)(faces[0].top()*FACE_DOWNSAMPLE_RATIO_DLIB),
                            (long)(faces[0].right()*FACE_DOWNSAMPLE_RATIO_DLIB),
                            (long)(faces[0].bottom()*FACE_DOWNSAMPLE_RATIO_DLIB)
                         );

        full_object_detection landmarks;
        landmarks = poseModel(cimg, r);

        // 检查用户的眼睛是睁开的还是闭起来的
        int eyeStatus = checkEyeStatus(landmarks);

        // 将眼睛的状态传入FSM来确定眨眼的状态
        int blinkStatus = checkBlinkStatus(eyeStatus, blinkCount, drowsy);

        for(int i=0;i<righteye_index.size();i++)
        {
            cv::Point pt(landmarks.part(righteye_index[i]).x(), landmarks.part(righteye_index[i]).y() );
            circle(frame, pt, 1, Scalar(255, 0, 255), 1, 8);
        }
        for(int i=0;i<lefteye_index.size();i++)
        {
            cv::Point pt(landmarks.part(lefteye_index[i]).x(), landmarks.part(lefteye_index[i]).y() );
            circle(frame, pt, 1, Scalar(255, 0, 255), 1, 8);
        }

        if(drowsy)
        {
            putText(eye, cv::format("state:%d , blinks:%d", state, blinkCount), cv::Point(10, 50), cv::FONT_HERSHEY_COMPLEX, 0.9, cv::Scalar(0, 255, 255), 1);
            putText(frame, cv::format("!!! DROWSY !!!"), cv::Point(50,100), cv::FONT_HERSHEY_COMPLEX,1.2, cv::Scalar(0, 0, 255),1);
        }
        else
        {
            putText(eye, cv::format("state:%d , blinks:%d", state, blinkCount), cv::Point(10, 50), cv::FONT_HERSHEY_COMPLEX, 0.9, cv::Scalar(0, 255, 255), 1);
            putText(frame, cv::format("blinks: %d", blinkCount), cv::Point(50,100), cv::FONT_HERSHEY_COMPLEX,0.9, cv::Scalar(0, 255, 255),1);
        }

        imshow("Blink Detection Demo", frame);
        imshow("Eye", eye);
        char c= (char)waitKey(1);
        if(c==27)
        {
            break;
        }

    }

    capture.release();
    destroyAllWindows();


    return 0;
}
