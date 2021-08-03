#include <iostream>
#include <string>
#include <cstdlib>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include "cv.h"
#include "highgui.h"
#include "math.h"
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

#define C CV_PI / 3

cv::Mat HSV;
cv::Mat threshold_ori;

int Otsu(IplImage* src)
{
    int height = src->height;
    int width = src->width;
    //histogram    
    float histogram[256] = { 0 };
    for (int i = 0; i < height; i++)
    {
        unsigned char* p = (unsigned char*)src->imageData + src->widthStep * i;
        for (int j = 0; j < width; j++)
        {
            histogram[*p++]++;
        }
    }
    //normalize histogram    
    int size = height * width;
    for (int i = 0; i < 256; i++)
    {
        histogram[i] = histogram[i] / size;
    }

    //average pixel value    
    float avgValue = 0;
    for (int i = 0; i < 256; i++)
    {
        avgValue += i * histogram[i];  //整幅图像的平均灰度  
    }

    int threshold;
    float maxVariance = 0;
    float w = 0, u = 0;
    for (int i = 0; i < 256; i++)
    {
        w += histogram[i];  //假设当前灰度i为阈值, 0~i 灰度的像素(假设像素值在此范围的像素叫做前景像素) 所占整幅图像的比例  
        u += i * histogram[i];  // 灰度i 之前的像素(0~i)的平均灰度值： 前景像素的平均灰度值  

        float t = avgValue * w - u;
        float variance = t * t / (w * (1 - w));
        if (variance > maxVariance)
        {
            maxVariance = variance;
            threshold = i;
        }
    }

    return threshold;
}

void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    cv::Mat cameraFeed;
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8); // Caution the type here.
    }
    catch (cv_bridge::Exception& ex)
    {
        ROS_ERROR("cv_bridge exception in rgbcallback: %s", ex.what());
        exit(-1);
    }
 
    cameraFeed = cv_ptr->image.clone();
    IplImage temp = IplImage(cameraFeed);
    IplImage *ImgGray = &temp;
    IplImage *img = cvCreateImage(cvGetSize(ImgGray), IPL_DEPTH_8U, 1);
    cvCvtColor(ImgGray, img, CV_BGR2GRAY); //cvCvtColor(src,des,CV_BGR2GRAY)
    // cvNamedWindow("img", 1);
    // cvShowImage("img", img);

    IplImage* dst = cvCreateImage(cvGetSize(img), 8, 1);
    int threshold = Otsu(img);//最大类间方差阈值分割
    printf("threshold = %d\n", threshold);
    cvThreshold(img, dst, threshold, 255, CV_THRESH_BINARY);
    cvNamedWindow("dst", 1);
    cvShowImage("dst", dst);

    CvRect roi = cvRect(0, 0, (msg->width)/2, ((msg->height) * 0.8)); //去除复杂背景
    IplImage *img1 = cvCreateImage(cvGetSize(dst), dst->depth, dst->nChannels); //设置白板
    for (int y = 0; y < img1->height; y++)
    {
        for (int x = 0; x < img1->width; x++)
        {
            CvScalar cs = (255);
            cvSet2D(img1, y, x, cs);
        }
    }
    CvRect roi1 = cvRect(0, 0, (msg->width) / 2, ((msg->height) * 0.8));

    cvSetImageROI(dst, roi);
    cvSetImageROI(img1, roi1);
    cvCopy(dst, img1);
    cvResetImageROI(dst);
    cvResetImageROI(img1);
    cvNamedWindow("result", 1);
    cvShowImage("result", img1);

    IplImage*edge = cvCreateImage(cvGetSize(img1), 8, 1);//canny边缘检测
    int edgeThresh = 1;
    cvCanny(img1, edge, edgeThresh, edgeThresh * 3, 3);
    cvNamedWindow("canny", 1);
    cvShowImage("canny", edge);
    int count = 0;
    for (int yy = 0; yy < edge->height; yy++)//统计边缘图像中共有多少个黑色像素点
    {
        for (int xx = 0; xx < edge->width; xx++)
        {
            //CvScalar ss = (255);
            double ds = cvGet2D(edge, yy, xx).val[0];
            if (ds == 0)
                count++;
        }
    }
    // std::cout << "num of black points:" << count << std::endl;
    // std::cout << "num of white points:" << ((edge->width) * (edge->height)) << std::endl;
    // std::cout << "num of - points:" << (((edge->width) * (edge->height)) - count) << std::endl;

    int dianshu_threshold = (((edge->width) * (edge->height)) - count) / 4; //将白色像素点数的四分之一作为hough变换的阈值
    IplImage* houghtu = cvCreateImage(cvGetSize(edge), IPL_DEPTH_8U, 1);//hough直线变换
    CvMemStorage *storage = cvCreateMemStorage();//内存存储器是一个可用来存储诸如序列，轮廓，图形,子划分等动态增长数据结构的底层结构。
    CvSeq*lines = 0;
    int i,j,k,m,n;
    while (true)//循环找出合适的阈值，使检测到的直线的数量在8-12之间
    {
        lines = cvHoughLines2(edge, storage, CV_HOUGH_STANDARD, 1, CV_PI / 180, dianshu_threshold, 0, 0);
        int line_number = lines->total;
        if (line_number <8)
        {
            dianshu_threshold = dianshu_threshold - 2;
        }
        else if (line_number > 12)
        {
            dianshu_threshold = dianshu_threshold +1;
        }
        else
        {
            printf("line_number=%d\n", line_number);
            break;
        }
    }

    // std::cout << "num of dianshu_threshold:" << dianshu_threshold << std::endl;
    int A = 10;
    double B = CV_PI / 10;

    while (1)
    {
        for (i = 0; i <lines->total; i++)//将多条非常相像的直线剔除
        {
            for (j = 0; j < lines->total; j++)
            {
                if (j != i)
                {
                    float*line1 = (float*)cvGetSeqElem(lines, i);
                    float*line2 = (float*)cvGetSeqElem(lines, j);
                    float rho1 = line1[0];
                    float threta1 = line1[1];
                    float rho2 = line2[0];
                    float threta2 = line2[1];
                    if (abs(rho1 - rho2) < A && abs(threta1 - threta2) < B)
                        cvSeqRemove(lines, j);
                }
            }
        }
        if (lines->total > 4)//剔除一圈后如何直线的数量大于4，则改变A和B，继续删除相似的直线
        {
            A = A + 1;
            B = B + CV_PI / 180;
        }
        else
        {
            printf("lines->total=%d\n", lines->total);
            break;
        }
    }

    for (k= 0; k < lines->total; k++)//画出直线
    {
        float*line = (float*)cvGetSeqElem(lines, k);
        float rho = line[0];//r=line[0]
        float threta = line[1];//threta=line[1]
        CvPoint pt1, pt2;
        double a = cos(threta), b = sin(threta);
        double x0 = a*rho;
        double y0 = b*rho;
        pt1.x = cvRound(x0 + 100 * (-b));//定义直线的终点和起点，直线上每一个点应该满足直线方程r=xcos(threta)+ysin(threta);
        pt1.y = cvRound(y0 + 100 * (a));
        pt2.x = cvRound(x0 - 1200 * (-b));
        pt2.y = cvRound(y0 - 1200 * (a));
        cvLine(houghtu, pt1, pt2, CV_RGB(0, 255, 255), 1, 8);
    }
    int num = 0;
    CvPoint arr[8] = { { 0, 0 } };
    for (m = 0; m < lines->total; m++)//画出直线的交点
    {
        for (n = 0; n < lines->total; n++)
        {
            if (n!= m)
            {
                float*Line1 = (float*)cvGetSeqElem(lines,m);
                float*Line2 = (float*)cvGetSeqElem(lines,n);
                float Rho1 = Line1[0];
                float Threta1 = Line1[1];
                float Rho2 =Line2[0];
                float Threta2 = Line2[1];
                if (abs(Threta1 - Threta2) > C)
                {
                    double a1 = cos(Threta1), b1 = sin(Threta1);
                    double a2 = cos(Threta2), b2 = sin(Threta2);
                    CvPoint pt;
                    pt.x = (Rho2*b1 - Rho1*b2) / (a2*b1 - a1*b2);//直线交点公式
                    pt.y = (Rho1 - a1*pt.x) / b1;
                    cvCircle(houghtu, pt, 3, CV_RGB(255, 255, 0));
                    arr[num++] = pt;//将点的坐标保存在一个数组中
                }
            }

        }
    }
    // printf("num=%d\n", num);
    // printf("arr[0].x=%d\n", arr[0].x);
    // printf("arr[0].y=%d\n", arr[0].y);
    // printf("arr[1].x=%d\n", arr[1].x);
    // printf("arr[1].y=%d\n", arr[1].y);
    // printf("arr[2].x=%d\n", arr[2].x);
    // printf("arr[2].y=%d\n", arr[2].y);
    // printf("arr[3].x=%d\n", arr[3].x);
    // printf("arr[3].y=%d\n", arr[3].y);
    // printf("arr[4].x=%d\n", arr[4].x);
    // printf("arr[4].y=%d\n", arr[4].y);
    // printf("arr[5].x=%d\n", arr[5].x);
    // printf("arr[5].y=%d\n", arr[5].y);
    // printf("arr[6].x=%d\n", arr[6].x);
    // printf("arr[6].y=%d\n", arr[6].y);
    // printf("arr[7].x=%d\n", arr[7].x);
    // printf("arr[7].y=%d\n", arr[7].y);

    CvPoint arr1[8] = { { 0, 0 } };//将重复的角点剔除
    int num1 = 0;
    for (int r = 0; r < 8; r++)
    {
        int s = 0;
        for (; s < num1; s++)
        {
            if (abs(arr[r].x - arr1[s].x) <= 2 && abs(arr[r].y - arr1[s].y) <= 2)
                break;
        }
        if (s == num1)
        {
            arr1[num1] = arr[r];
            num1++;
        }
    }

    printf("num1=%d\n", num1);
    printf("arr1[0].x=%d\n", arr1[0].x);
    printf("arr1[0].y=%d\n", arr1[0].y);
    printf("arr1[1].x=%d\n", arr1[1].x);
    printf("arr1[1].y=%d\n", arr1[1].y);
    printf("arr1[2].x=%d\n", arr1[2].x);
    printf("arr1[2].y=%d\n", arr1[2].y);
    printf("arr1[3].x=%d\n", arr1[3].x);
    printf("arr1[3].y=%d\n", arr1[3].y);
    printf("arr1[4].x=%d\n", arr1[4].x);
    printf("arr1[4].y=%d\n", arr1[4].y);
    printf("arr1[5].x=%d\n", arr1[5].x);
    printf("arr1[5].y=%d\n", arr1[5].y);
    printf("arr1[6].x=%d\n", arr1[6].x);
    printf("arr1[6].y=%d\n", arr1[6].y);
    printf("arr1[7].x=%d\n", arr1[7].x);
    printf("arr1[7].y=%d\n", arr1[7].y);

    for (int w = 0; w < 4; w++)
    {
        CvPoint ps;
        ps = arr1[w];
        cvCircle(img, ps, 3, CV_RGB(255,0,0));
    }
    cvNamedWindow("img", 1);
    cvShowImage("img", img);
    cvNamedWindow("houghtu", 1);
    cvShowImage("houghtu", houghtu);

    cvWaitKey(10000);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "lidar_camera_extrinsic_calibration_node");
    ros::NodeHandle node;
    ros::NodeHandle n("~");
    ros::Subscriber sub = n.subscribe("/camera/image_color", 1000, imageCallback);
    ros::spin();
    return 0;
}
