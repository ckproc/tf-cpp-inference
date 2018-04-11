//#include "tensorflow/core/public/session.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <glob.h>
#include <vector>
#include <math.h>
#include "time.h"

//using namespace std;
//using namespace cv;
/*
struct BoundingBox{
        //rect two points
        float x1, y1;
        float x2, y2;
        //regression
        float dx1, dy1;
        float dx2, dy2;
        //cls
        float score;
        //inner points
        float points_x[5];
        float points_y[5];
    };

void load_face_detector(std::string model);

void load_face_embedding(std::string model);

int face_detect(cv::Mat frame  , std::vector<BoundingBox> &faces);

void face_embedding(std::vector<cv::Mat> &aligned_img, std::vector<cv::Mat> &embed);

void single_face_embedding(cv::Mat &face_img, cv::Mat &float_feature);

*/
void load_filter_face(std::string model);

int face_filter(cv::Mat img);














int align_mtcnn(cv::Mat &image, std::unique_ptr<tensorflow::Session> &session1, std::unique_ptr<tensorflow::Session> &session2,
   std::unique_ptr<tensorflow::Session> &session3, std::vector<BoundingBox> &faces);
   
int Load_model(std::string model,std::unique_ptr<tensorflow::Session> &session1, std::unique_ptr<tensorflow::Session> &session2,
   std::unique_ptr<tensorflow::Session> &session3);
};*/
