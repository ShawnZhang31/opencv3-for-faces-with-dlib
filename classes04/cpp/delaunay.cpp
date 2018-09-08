#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

/**
 * @brief 获取顶点的索引
 * 
 * @param points 顶点集合
 * @param point 点
 * @return int 返回索引值
 */
static int findIndex(vector<Point2f>& points, Point2f &point)
{
    int minIndex = 0;
    double minDistance = norm(points[0] - point);
    for(int i=0; i<points.size();i++)
    {
        double distance = norm(points[i]-point);
        cout<<i<<":"<<"minDistance="<<minDistance<<";distance="<<distance<<endl;
        if(distance < minDistance)
        {
            minIndex = i;
            minDistance = distance;
        }
    }
    return minIndex;
}

/**
 * @brief 保存Delaunary的数据
 * 
 * @param subdiv 
 * @param points 
 * @param filename 
 */
static void writeDelaunay(Subdiv2D& subdiv, vector<Point2f>& points, const string &filename)
{
    //输出文件流
    ofstream ofs;
    ofs.open(filename);

    // 每个三角形有三个点组成，共六个浮点型数值
    vector<Vec6f> triangeleList;
    subdiv.getTriangleList(triangeleList);
    
    // 我们关心的实际上是构成三角形的索引，而不是具体数值
    vector<Point2f> vectices(3);
    for(size_t i=0; i<triangeleList.size();i++)
    {
        Vec6f t=triangeleList[i];
        vectices[0]=Point2f(t[0],t[1]);
        vectices[1]=Point2f(t[2],t[3]);
        vectices[2]=Point2f(t[4],t[5]);

        cout<<i<<":"<<t[0]<<" "<<t[1]<<" "<<t[2]<<" "<<t[3]<<" "<<t[4]<<" "<<t[5]<<" "<<endl;

        ofs << findIndex(points, vectices[0])<<" "
        << findIndex(points, vectices[1])<<" "
        << findIndex(points, vectices[2])<<endl;
    }

    ofs.close();
}

int main(int argc, char const *argv[])
{
    /* 读取点集数据
     */
    vector<Point2f> points;
    string pointsFilename("../data/images/smiling-man-delaunay.txt");
    ifstream ifs(pointsFilename);
    cout<<"Reading file"<<pointsFilename<<endl;
    int x, y;
    while(ifs >> x >> y)
    {
        points.push_back(Point2f(x,y));
    }
    cout<<"Reading completed!"<<endl;

    // subdiv2d 类需要使用包含点集的矩形来初始化
    Rect rect = boundingRect(points);
    Subdiv2D subdiv(rect);

    // 将点传入划分器
    for(vector<Point2f>::iterator it = points.begin(); it != points.end();it++)
    {
        subdiv.insert(*it);
    }

    // 将划分后的三角形数据保存下来
    string outputFileName("results/smiling-man-delaunay.tri");

    cout<<"将三角形数据写入文件..."<<endl;
    writeDelaunay(subdiv, points, outputFileName);
    cout<<"写入完成！"<<endl;

    return 0;
}
