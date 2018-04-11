// tensorflow/cc/example/example.cc
#include "detect_embed.hpp"
#include<iostream>
#include <vector>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"




using tensorflow::Tensor;
using tensorflow::Status;
//using tensorflow::string;
using tensorflow::int32;
using namespace tensorflow;
using namespace tensorflow::ops;
using namespace std;
using namespace cv;
static String sEndTime ="2018-2-31";
bool IsValidTime(const time_t& AEndTime, const time_t& ANowTime );
time_t str_to_time_t(const string& ATime, const string& AFormat="%d-%d-%d");
time_t NowTime();
bool isAvailability() {
    string sTemp;
    time_t t_Now = NowTime();
    time_t t_End = str_to_time_t(sEndTime);
    return IsValidTime(t_End, t_Now);
}

time_t str_to_time_t(const string& ATime, const string& AFormat)  {  
    struct tm tm_Temp;  
    time_t time_Ret;  
    try 
    {
        int i = sscanf(ATime.c_str(), AFormat.c_str(),// "%d/%d/%d %d:%d:%d" ,       
            &(tm_Temp.tm_year),   
            &(tm_Temp.tm_mon),   
            &(tm_Temp.tm_mday),  
            &(tm_Temp.tm_hour),  
            &(tm_Temp.tm_min),  
            &(tm_Temp.tm_sec),  
            &(tm_Temp.tm_wday),  
            &(tm_Temp.tm_yday));  
        tm_Temp.tm_year -= 1900;  
        tm_Temp.tm_mon --;  
        tm_Temp.tm_hour=0;  
        tm_Temp.tm_min=0;  
        tm_Temp.tm_sec=0;  
        tm_Temp.tm_isdst = 0;
        time_Ret = mktime(&tm_Temp);  
        return time_Ret;  
    } catch(...) {
        return 0;
    }
}
time_t NowTime(){
    time_t t_Now = time(0);
    struct tm* tm_Now = localtime(&t_Now);
    tm_Now->tm_hour =0;
    tm_Now->tm_min = 0;
    tm_Now->tm_sec = 0;
    return  mktime(tm_Now);  
}
bool IsValidTime(const time_t& AEndTime, const time_t& ANowTime ){
    return (AEndTime >= ANowTime);
}

unique_ptr<tensorflow::Session> session0;
//unique_ptr<tensorflow::Session> session1;
//unique_ptr<tensorflow::Session> session2;
//unique_ptr<tensorflow::Session> session3;
unique_ptr<tensorflow::Session> session4;
unique_ptr<tensorflow::Session> session5;

//tensorflow::GraphDef graph_def;

enum NMS_TYPE{
        MIN,
        UNION,
    };
	
struct CmpBoundingBox{
        bool operator() (const BoundingBox& b1, const BoundingBox& b2)
        {
            return b1.score > b2.score;
        }
    };
inline std::vector<std::string> glob(const std::string& pat){
    
    glob_t glob_result;
    glob(pat.c_str(),GLOB_TILDE,NULL,&glob_result);
    vector<string> ret;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        ret.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return ret;
}

Tensor convert_tensor(vector<cv::Mat> &image) //Tensor &inputImg)
{
  int b_size=image.size();
  int inputheight=image[0].rows;
  int inputwidth=image[0].cols;
  Tensor inputImg(DT_FLOAT, TensorShape({b_size,inputheight,inputwidth,3}));
  auto inputImageMapped = inputImg.tensor<float, 4>();
  for (int b=0;b<b_size;++b){
   for (int y = 0; y < inputheight; ++y) {
    const float* source_row = ((float*)image[b].data) + (y * inputwidth * 3);
    for (int x = 0; x < inputwidth; ++x) {
        const float* source_pixel = source_row + (x * 3);
        inputImageMapped(b, y, x, 0) = source_pixel[2];
        inputImageMapped(b, y, x, 1) = source_pixel[1];
        inputImageMapped(b, y, x, 2) = source_pixel[0];
		//cout<<inputImageMapped(b, y, x, 0)<<" ";
    }
   }	
  }
   return inputImg;
}

Tensor filter_convert_tensor(vector<cv::Mat> &image) //Tensor &inputImg)
{
  int b_size=image.size();
  //cout<<b_size<<endl;
  int inputheight=image[0].rows;
  int inputwidth=image[0].cols;
  Tensor inputImg(DT_FLOAT, TensorShape({b_size,inputheight,inputwidth,3}));
  auto inputImageMapped = inputImg.tensor<float, 4>();
  for (int b=0;b<b_size;++b){
   for (int y = 0; y < inputheight; ++y) {
    const float* source_row = ((float*)image[b].data) + (y * inputwidth * 3);
    for (int x = 0; x < inputwidth; ++x) {
        const float* source_pixel = source_row + (x * 3);
        inputImageMapped(b, y, x, 0) = source_pixel[2]-123.68;
        //inputImageMapped(b, y, x, 0) = source_pixel[2]-127.5;
        //inputImageMapped(b, y, x, 1) = source_pixel[1]-127.5;
        inputImageMapped(b, y, x, 1) = source_pixel[1]-116.78;
        //inputImageMapped(b, y, x, 2) = source_pixel[0]-127.5;
        inputImageMapped(b, y, x, 2) = source_pixel[0]-103.94;
    }
   }	
  }
   return inputImg;
}

/*
void nms_cpu(vector<BoundingBox>& boxes, float threshold, NMS_TYPE type, vector<BoundingBox>& filterOutBoxes)
{
    filterOutBoxes.clear();
    if(boxes.size() == 0)
        return;
    //descending sort
    sort(boxes.begin(), boxes.end(), CmpBoundingBox() );
    vector<size_t> idx(boxes.size());
    //std::iota(idx.begin(), idx.end(), 0);//create index
    for(int i = 0; i < idx.size(); ++i)
    { 
        idx[i] = i; 
    }
    while(idx.size() > 0)
    {
        int good_idx = idx[0];
        filterOutBoxes.push_back(boxes[good_idx]);
        //hypothesis : the closer the scores are similar
        vector<size_t> tmp = idx;
        idx.clear();
        for(int i = 1; i < tmp.size(); ++i)
        {
            int tmp_i = tmp[i];
            float inter_x1 = std::max( boxes[good_idx].x1, boxes[tmp_i].x1 );
            float inter_y1 = std::max( boxes[good_idx].y1, boxes[tmp_i].y1 );
            float inter_x2 = std::min( boxes[good_idx].x2, boxes[tmp_i].x2 );
            float inter_y2 = std::min( boxes[good_idx].y2, boxes[tmp_i].y2 );
             
            float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
            float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);
            
            float inter_area = w * h;
            float area_1 = (boxes[good_idx].x2 - boxes[good_idx].x1 + 1) * (boxes[good_idx].y2 - boxes[good_idx].y1 + 1);
            float area_2 = (boxes[i].x2 - boxes[i].x1 + 1) * (boxes[i].y2 - boxes[i].y1 + 1);
            float o = ( type == UNION ? (inter_area / (area_1 + area_2 - inter_area)) : (inter_area / std::min(area_1 , area_2)) );           
            if( o <= threshold )
                idx.push_back(tmp_i);
        }
    }
}
void generateBoundingBox(Tensor & boxRegs, Tensor& cls,float scale, float threshold, vector<BoundingBox>& filterOutBoxes)
{
	filterOutBoxes.clear();
    int stride = 2;
    int cellsize = 12;
	int w = boxRegs.dim_size(2);
    int h = boxRegs.dim_size(1);
	//boxRegs.tensor<float, 4>()(n,h,w,c)
	for(int y = 0; y < h; ++y)
    {
        for(int x = 0; x < w; ++x)
        {
			float score = cls.tensor<float,4>()(0,y,x,1);
            if ( score >= threshold)
            {
                BoundingBox box;
                box.dx1 = boxRegs.tensor<float,4>()(0,y,x,0);
                box.dy1 = boxRegs.tensor<float,4>()(0,y,x,1);
                box.dx2 = boxRegs.tensor<float,4>()(0,y,x,2);
                box.dy2 = boxRegs.tensor<float,4>()(0,y,x,3);
                
                box.x1 = std::floor( (stride * x + 1) / scale );
                box.y1 = std::floor( (stride * y + 1) / scale );
                box.x2 = std::floor( (stride * x + cellsize) / scale );  
                box.y2 = std::floor( (stride * y + cellsize) / scale );
                box.score = score;
                //add elements
                filterOutBoxes.push_back(box);
            }
        }
    }	
}

void filteroutBoundingBox(const vector<BoundingBox >& boxes, Tensor& boxRegs, Tensor& cls, 
                  Tensor& points,float threshold, vector<BoundingBox >& filterOutBoxes)
{
    filterOutBoxes.clear();
    //assert(box_shape.size() == cls_shape.size());
    //assert(box_shape[0] == boxes.size() && cls_shape[0] == boxes.size());
   // assert(box_shape[1] == 4 && cls_shape[1] == 2);
    //if(points.size() > 0)
    //{
    //    assert(points_shape[0] == boxes.size() && points_shape[1] == 10);
    //}

    //int w = box_shape[3];
    //int h = box_shape[2];
    for(int i = 0; i < boxes.size(); ++i)
    {
        float score = cls.tensor<float,2>()(i,1);//[i * 2 + 1];
        if ( score > threshold )
        {
            BoundingBox box = boxes[i];
            float w = boxes[i].y2 - boxes[i].y1 + 1;
            float h = boxes[i].x2 - boxes[i].x1 + 1;
            if( points.dims() > 1)
            {
                for(int p = 0; p < 5; ++p)
                {
                    box.points_x[p] = points.tensor<float,2>()(i,5+p) * w + boxes[i].x1 - 1;//[i * 10 + 5 + p] * w + boxes[i].x1 - 1;
                    box.points_y[p] = points.tensor<float,2>()(i,p) * h + boxes[i].y1-1;//[i * 10 + p] * h + boxes[i].y1 - 1;
                }
            }
            box.dx1 = boxRegs.tensor<float,2>()(i,0);//[i * 4 + 0];
            box.dy1 = boxRegs.tensor<float,2>()(i,1);
            box.dx2 = boxRegs.tensor<float,2>()(i,2);
            box.dy2 = boxRegs.tensor<float,2>()(i,3);            
            
            box.x1 = boxes[i].x1 + box.dy1 * w;
            box.y1 = boxes[i].y1 + box.dx1 * h;
            box.x2 = boxes[i].x2 + box.dy2 * w;
            box.y2 = boxes[i].y2 + box.dx2 * h;
            
            //rerec
            w = box.x2 - box.x1;
            h = box.y2 - box.y1;
            float l = std::max(w, h);
            box.x1 += (w - l) * 0.5;
            box.y1 += (h - l) * 0.5;
            box.x2 = box.x1 + l;
            box.y2 = box.y1 + l;
            box.score = score;
            
            filterOutBoxes.push_back(box);
        }
    }
}

void filteroutBoundingBox1(const vector<BoundingBox >& boxes, Tensor& boxRegs, Tensor& cls, 
                  Tensor& points,float threshold, vector<BoundingBox >& filterOutBoxes)
{
    filterOutBoxes.clear();
    //assert(box_shape.size() == cls_shape.size());
    //assert(box_shape[0] == boxes.size() && cls_shape[0] == boxes.size());
   // assert(box_shape[1] == 4 && cls_shape[1] == 2);
    //if(points.size() > 0)
    //{
    //    assert(points_shape[0] == boxes.size() && points_shape[1] == 10);
    //}

    //int w = box_shape[3];
    //int h = box_shape[2];
    for(int i = 0; i < boxes.size(); ++i)
    {
        float score = cls.tensor<float,2>()(i,1);//[i * 2 + 1];
        if ( score > threshold )
        {
            BoundingBox box = boxes[i];
            float w = boxes[i].y2 - boxes[i].y1 + 1;
            float h = boxes[i].x2 - boxes[i].x1 + 1;
            if( points.dims() > 1)
            {
                for(int p = 0; p < 5; ++p)
                {
                    box.points_x[p] = points.tensor<float,2>()(i,5+p) * w + boxes[i].x1 - 1;//[i * 10 + 5 + p] * w + boxes[i].x1 - 1;
                    box.points_y[p] = points.tensor<float,2>()(i,p) * h + boxes[i].y1-1;//[i * 10 + p] * h + boxes[i].y1 - 1;
                }
            }
            box.dx1 = boxRegs.tensor<float,2>()(i,0);//[i * 4 + 0];
            box.dy1 = boxRegs.tensor<float,2>()(i,1);
            box.dx2 = boxRegs.tensor<float,2>()(i,2);
            box.dy2 = boxRegs.tensor<float,2>()(i,3);            
            
            box.x1 = boxes[i].x1 + box.dy1 * w;
            box.y1 = boxes[i].y1 + box.dx1 * h;
            box.x2 = boxes[i].x2 + box.dy2 * w;
            box.y2 = boxes[i].y2 + box.dx2 * h;
            
            //rerec
            //w = box.x2 - box.x1;
            //h = box.y2 - box.y1;
            //float l = std::max(w, h);
            //box.x1 += (w - l) * 0.5;
            //box.y1 += (h - l) * 0.5;
            //box.x2 = box.x1 + l;
            //box.y2 = box.y1 + l;
            box.score = score;
            
            filterOutBoxes.push_back(box);
        }
    }
}

void buildInput(vector<cv::Mat> &candidates,cv::Mat &image, vector<BoundingBox> &totalBoxes, const cv::Size& ta_size, const int img_H, const int img_W)
{
	cv::Rect img_rect(0, 0, img_W, img_H);
		for(int n = 0; n < totalBoxes.size(); ++n)
        {
			cv::Rect rect;
			rect.x = totalBoxes[n].x1;
            rect.y = totalBoxes[n].y1;
            rect.width = totalBoxes[n].x2 - totalBoxes[n].x1 + 1;
            rect.height = totalBoxes[n].y2 - totalBoxes[n].y1 + 1;
			cv::Rect cuted_rect = rect & img_rect;
            cv::Rect inner_rect(cuted_rect.x - rect.x, cuted_rect.y - rect.y, cuted_rect.width, cuted_rect.height);
			cv::Mat tmp(rect.height, rect.width, CV_32FC3, cv::Scalar(0.0));
            image(cuted_rect).copyTo(tmp(inner_rect));
            cv::resize(tmp, tmp, ta_size); 
			candidates.push_back(tmp);
		}
		
} */

void prewhiten(cv::Mat &crop,cv::Mat &new_crop){
	using namespace cv;
	cv::Scalar tempVal = mean(crop);
	float mean1 = tempVal.val[0];
	float mean2 = tempVal.val[1];
	float mean3 = tempVal.val[2];
	float mean=(mean1+mean2+mean3)/3.0;
	float size=crop.rows*crop.cols*crop.channels();
	uint8_t* pixelPtr = (uint8_t*)crop.data;
	float temp=0;
	for(int i=0;i<size;++i){
		float value=float(pixelPtr[i]);
		temp+=pow(value-mean,2.0);
	}
	temp=sqrt(temp/size);
	float std_adj=std::max(temp,float(1.0/sqrt(size)));
	//elenment-wise operation
	for(int j=0;j<new_crop.rows;++j){
		for(int k=0;k<new_crop.cols;++k){
			new_crop.at<Vec3f>(j,k)[0]=(float(crop.at<Vec3b>(j,k)[0])-mean)/std_adj;
			new_crop.at<Vec3f>(j,k)[1]=(float(crop.at<Vec3b>(j,k)[1])-mean)/std_adj;
			new_crop.at<Vec3f>(j,k)[2]=(float(crop.at<Vec3b>(j,k)[2])-mean)/std_adj;
		}
	}
	
}
/*
int align_mtcnn(cv::Mat &image, vector<BoundingBox> &faces)//cv::Mat &aligned_img)
{
  string pnet_input_layer = "pnet/input:0";
  string pnet_output_layer1 = "pnet/conv4-2/BiasAdd:0";
  string pnet_output_layer2 ="pnet/prob1:0"; 
  string rnet_input_layer = "rnet/input:0";
  string rnet_output_layer1 ="rnet/conv5-2/conv5-2:0";
  string rnet_output_layer2 ="rnet/prob1:0";  
  string onet_input_layer="onet/input:0";
  string onet_output_layer1="onet/conv6-2/conv6-2:0";
  string onet_output_layer3="onet/conv6-3/conv6-3:0";
  string onet_output_layer2="onet/prob1:0"; 
  std::vector<Tensor> pnet_outputs;
  std::vector<Tensor> rnet_outputs;
  std::vector<Tensor> onet_outputs;
  int minsize =16;
  float P_thres = 0.5;
  float R_thres = 0.6;
  float O_thres =0.7;
  float factor = 0.8;
  //float factor = 0.50;
  
  float input_mean = 127.5;
  float input_std=0.0078125;
    //cv::Mat c_image;
    //image.convertTo( c_image, CV_32FC3, input_std, -input_mean * input_std);
    image.convertTo( image, CV_32FC3, input_std, -input_mean * input_std);
	//image.convertTo(image,CV_8UC3);
	//image=image.t();
    //c_image=c_image.t(); 
    image=image.t(); 
    int img_H = image.rows;
    int img_W = image.cols;
    int minl  = cv::min(img_H, img_W);
	float m = 12.0 / minsize;
    minl *= m;
    vector<float> all_scales;
    float cur_scale = 1.0;
    while( minl >= 12.0 )
    {
        all_scales.push_back( m * cur_scale);
        cur_scale *= factor;
        minl *= factor;
    }	
	//stage 1:pnet :rec and regression
	vector<BoundingBox> totalBoxes;
	vector<cv::Mat> cur_images;
	cv::Mat cur_image;
	
	double t1=(double)getTickCount();
	for(int i = 0; i < all_scales.size(); ++i)
    {
		cur_images.clear();
		cur_scale=all_scales[i];
		int hs = cvCeil(img_H * cur_scale);
        int ws = cvCeil(img_W * cur_scale);		
		cv::resize(image, cur_image, cv::Size(ws, hs));
		cur_images.push_back(cur_image);
		Tensor inputImg = convert_tensor(cur_images);
		//graph::SetDefaultDevice("/gpu:0", &graph_def);//cout<<"ok1"<<endl;
		Status run_status = session1->Run({{pnet_input_layer, inputImg}},{pnet_output_layer1,pnet_output_layer2}, {}, &pnet_outputs);
        if (!run_status.ok())
			{      //cout<<"sess1"<<endl;
		           LOG(ERROR) << "Running model failed: " << run_status;
		           return -1;
				   }
				   //cout<<"ok2"<<endl;
		vector<BoundingBox> filterOutBoxes;
        vector<BoundingBox> nmsOutBoxes;
		generateBoundingBox(pnet_outputs[0], pnet_outputs[1], cur_scale, P_thres, filterOutBoxes);
		nms_cpu(filterOutBoxes, 0.5, UNION, nmsOutBoxes);
        if(nmsOutBoxes.size() > 0)
            totalBoxes.insert(totalBoxes.end(), nmsOutBoxes.begin(), nmsOutBoxes.end());
	 }
	 //cout<<"p4"<<endl;
	//cout<<totalBoxes.size()<<endl;
	
	  if (totalBoxes.size() > 0)
      {
        vector<BoundingBox> globalFilterBoxes;
        nms_cpu(totalBoxes, 0.7, UNION, globalFilterBoxes);
        totalBoxes.clear();
        for(int i = 0; i < globalFilterBoxes.size(); ++i)
        {
            float regw = globalFilterBoxes[i].y2 - globalFilterBoxes[i].y1 ;
            float regh = globalFilterBoxes[i].x2 - globalFilterBoxes[i].x1;
            BoundingBox box;
            float x1 = globalFilterBoxes[i].x1 + globalFilterBoxes[i].dy1 * regw;
            float y1 = globalFilterBoxes[i].y1 + globalFilterBoxes[i].dx1 * regh; 
            float x2 = globalFilterBoxes[i].x2 + globalFilterBoxes[i].dy2 * regw;
            float y2 = globalFilterBoxes[i].y2 + globalFilterBoxes[i].dx2 * regh;
            float score = globalFilterBoxes[i].score;
            float h = y2 - y1;
            float w = x2 - x1;
            float l = std::max(h, w);
            x1 += (w - l) * 0.5;
            y1 += (h - l) * 0.5;
            x2 = x1 + l;
            y2 = y1 + l;
            box.x1 = x1, box.x2 = x2, box.y1 = y1, box.y2 = y2, box.score=score;
            totalBoxes.push_back(box);
        }
      }
      	//double t2=(double)getTickCount();  
		//double a=(t2-t1)/cv::getTickFrequency();
		//cout<<"pnet"<<a<<"	";
	if(totalBoxes.size() > 0)
    {
		vector<cv::Mat> r_candidates;
		buildInput(r_candidates, image, totalBoxes, cv::Size(24,24), img_H, img_W);
		Tensor rnet_input = convert_tensor(r_candidates);
		
		Status run_status = session2->Run({{rnet_input_layer,rnet_input }},{rnet_output_layer1,rnet_output_layer2}, {}, &rnet_outputs);
        if (!run_status.ok())
			{ 
				LOG(ERROR) << "Running model failed: " << run_status;
		       //cout<<"sess2"<<endl;
		          return -1;   }
		
		//cout<<rnet_outputs[0].dim_size(0)<<","<<rnet_outputs[0].dim_size(1)<<endl;
		vector<BoundingBox> filterOutBoxes;
		Tensor empty;
        filteroutBoundingBox(totalBoxes, rnet_outputs[0], rnet_outputs[1], empty, R_thres, filterOutBoxes);	
        nms_cpu(filterOutBoxes, 0.7, UNION, totalBoxes);
		
	}
		//double t3=(double)getTickCount();  
		//double b=(t3-t2)/cv::getTickFrequency();
		//cout<<"rnet"<<b<<"	";
	if(totalBoxes.size()>0)
	{
		vector<cv::Mat> o_candidates;
		
		buildInput(o_candidates, image, totalBoxes, cv::Size(48,48), img_H, img_W);
		
		Tensor onet_input = convert_tensor(o_candidates);
	
		Status run_status = session3->Run({{onet_input_layer, onet_input}},{onet_output_layer1,onet_output_layer2,onet_output_layer3}, {}, &onet_outputs);
        if (!run_status.ok()) 
		{ LOG(ERROR) << "Running model failed: " << run_status;
	              return -1;
				  }
		vector<BoundingBox> filterOutBoxes;

        filteroutBoundingBox1(totalBoxes, onet_outputs[0], onet_outputs[1], onet_outputs[2], O_thres, filterOutBoxes);
		
        nms_cpu(filterOutBoxes, 0.7, MIN, totalBoxes);
	}
		//double t4=(double)getTickCount();  
		//double c=(t4-t3)/cv::getTickFrequency();
		//cout<<"onet"<<c<<"	";

	if(totalBoxes.size()==0){ 
        cout<<"size is zero"<<endl;
		return -1;
	}
	faces = totalBoxes;

    return 0;
	
	

   
} */


Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  SessionOptions opts;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  //opts.config.set_intra_op_parallelism_threads(64);
  //opts.config.set_inter_op_parallelism_threads(64);
 // opts.config.add_session_inter_op_thread_pool()->set_num_threads(64);
  session->reset(tensorflow::NewSession(opts));
  //session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));

  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}
/*
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  SessionOptions opts;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  opts.config.mutable_gpu_options()->set_visible_device_list("1");
  //opts.config.mutable_cpu_options()->set_visible_device_list("0");
  //opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(portion);
  opts.config.mutable_gpu_options()->set_allow_growth(true);
  session->reset(tensorflow::NewSession(opts));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  
  return Status::OK();
} 
*/
void load_filter_face(string model){
	string graph_face_filter=model+"/"+"filter.pb";
	Status load_graph_status = LoadGraph(graph_face_filter, &session0);
}


int face_filter(cv::Mat img){
	
	string input="input/image";
	string input2="Placeholder_1";
	string output="MobilenetV1/Logits/SpatialSqueeze";
	resize(img,img,cv::Size(224,224));
	img.convertTo( img, CV_32FC3);
	std::vector<Tensor> facenet_outputs;
	  vector<cv::Mat> input_facenet;
	  input_facenet.push_back(img);
	  //cout<<"p2"<<endl;
	  Tensor in_facenet = filter_convert_tensor(input_facenet);
	  //cout<<"p3"<<endl;
	  Tensor phase(DT_BOOL, TensorShape());
	  phase.scalar<bool>()()=false;
      Status run_status = session0->Run({{input, in_facenet},{input2, phase}},
                                   {output}, {}, &facenet_outputs);
        if (!run_status.ok()) {
             LOG(ERROR) << "Running model failed: " << run_status;
             return 0;
        }
		
		if(facenet_outputs[0].tensor<float,2>()(0,0)>=facenet_outputs[0].tensor<float,2>()(0,1)){
			return 1;
		}
		else{
			return 0;
		}
}
/*
void load_face_detector(string model){
	if (!isAvailability()) {
			cerr << "The library out of data. Please contact the author\n";  
			exit(1);
	}
	
  string graph_pnet=model+"/"+"pnet.pb";
  string graph_rnet=model+"/"+"rnet.pb";
  string graph_onet=model+"/"+"onet.pb";
  Status load_graph_status = LoadGraph(graph_pnet, &session1);
  load_graph_status = LoadGraph(graph_rnet, &session2);
  load_graph_status = LoadGraph(graph_onet, &session3);
	
}*/
Tensor convert_detect_tensor(vector<cv::Mat> &image) //Tensor &inputImg)
{
  int b_size=image.size();
  
  int inputheight=image[0].rows;
  int inputwidth=image[0].cols;
  //cout<<b_size<<" "<<inputheight<<" "<<inputwidth<<endl;
  Tensor inputImg(DT_UINT8, TensorShape({b_size,inputheight,inputwidth,3}));
  auto inputImageMapped = inputImg.tensor<uint8, 4>();
  //cout<<"a"<<endl;
  for (int b=0;b<b_size;++b){
   for (int y = 0; y < inputheight; ++y) {
    //cout<<"b"<<endl;
	const uint8* source_row = ((uint8*)image[b].data) + (y * inputwidth * 3);
    for (int x = 0; x < inputwidth; ++x) {
		//cout<<"c"<<endl;
        const uint8* source_pixel = source_row + (x * 3);
        inputImageMapped(b, y, x, 0) = source_pixel[2];
        inputImageMapped(b, y, x, 1) = source_pixel[1];
        inputImageMapped(b, y, x, 2) = source_pixel[0];
        //cout<<"d"<<endl;
	}
   }	
  }
  //#cout<<"w?"<<endl;
   return inputImg;
}

void load_face_detector(string model){
	if (!isAvailability()) {
			cerr << "The library out of data. Please contact the author\n";  
			exit(1);
	}
	
  string graph_ssd=model+"/"+"frozen_inference_graph.pb";
  Status load_graph_status = LoadGraph(graph_ssd, &session5);
  //load_graph_status = LoadGraph(graph_rnet, &session2);
  //load_graph_status = LoadGraph(graph_onet, &session3);
	
}

void load_face_embedding(string model){
  string graph_facenet=model+"/"+"facenet.pb";
  
  Status load_graph_status = LoadGraph(graph_facenet, &session4);
	
}
/*
bool face_detect(cv::Mat frame  , std::vector<BoundingBox> &faces){
	
   int isdetect = align_mtcnn(frame, faces);
	if(isdetect==-1){
		//cout<<"failed detected"<<endl;
		return false;
	}
	return true;
}*/
bool face_detect(cv::Mat frame  , std::vector<BoundingBox> &faces){
	
	string ssd_input = "image_tensor:0";
	string ssd_detection_box = "detection_boxes:0";
	string ssd_score ="detection_scores:0"; 
	string ssd_class ="detection_classes:0"; 
	string ssd_num_detections ="num_detections:0"; 
	//session5
	std::vector<Tensor> ssd_outputs;
	vector<cv::Mat> input_ssd;
	input_ssd.push_back(frame);
	Tensor in_ssd = convert_detect_tensor(input_ssd);
	Status run_status = session5->Run({{ssd_input, in_ssd}},
	                   {ssd_detection_box,ssd_score,ssd_class,ssd_num_detections}, {}, &ssd_outputs);
	if (!run_status.ok()) {
             LOG(ERROR) << "Running detection model failed: " << run_status;
             return false;
        }	
	//ssd_outputs[0]    ssd_outputs[1] 
	for(int k=0;k<100;++k){
		if(ssd_outputs[1].tensor<float,2>()(0,k)>0.7){
			BoundingBox temp_box;
			temp_box.x1 = frame.cols*ssd_outputs[0].tensor<float,3>()(0,k,1);
			temp_box.y1 = frame.rows*ssd_outputs[0].tensor<float,3>()(0,k,0);
			temp_box.x2 = frame.cols*ssd_outputs[0].tensor<float,3>()(0,k,3);
			temp_box.y2 = frame.rows*ssd_outputs[0].tensor<float,3>()(0,k,2);
			temp_box.score = ssd_outputs[1].tensor<float,2>()(0,k);
			faces.push_back(temp_box);
		}
	}
	return true;
}


bool face_embedding(vector<cv::Mat> &aligned_img, vector<cv::Mat> &embeds){
	string facenet_input_layer1="input";
    string facenet_input_layer2="phase_train";
	string facenet_output_layer="embeddings";
	
    //string graph_facenet=model+"/"+"20170512-110547/20170512-110547.pb";
    //std::unique_ptr<tensorflow::Session> session4;
    //Status load_graph_status = LoadGraph(graph_facenet, &session4,0);
	
	  std::vector<Tensor> facenet_outputs;
	  //= convert_tensor(input_facenet);
	  Tensor phase(DT_BOOL, TensorShape());
	  phase.scalar<bool>()()=false;
	  
      for(int i=0;i<aligned_img.size();++i){
		  facenet_outputs.clear();
	    cv::Mat new_align(aligned_img[i].rows, aligned_img[i].cols, CV_32FC3, cv::Scalar::all(0.0)); 
	    prewhiten(aligned_img[i],new_align);
	  
	  //vector<cv::Mat> input_facenet;
	  vector<cv::Mat> input_facenet;
	  input_facenet.push_back(new_align);
	  Tensor in_facenet = convert_tensor(input_facenet);
	  Status run_status = session4->Run({{facenet_input_layer1, in_facenet},{facenet_input_layer2,phase}},
                                   {facenet_output_layer}, {}, &facenet_outputs);
		if (!run_status.ok()) {
             LOG(ERROR) << "Running model failed: " << run_status;
             continue;
        }
	  Mat embed = Mat(1, 128, CV_32FC1, Scalar::all(0.0));
	  for(int j=0;j<facenet_outputs[0].dim_size(1);++j){
               embed.ptr<float>(0)[j]=facenet_outputs[0].tensor<float,2>()(0,j); 
		}
		embeds.push_back(embed);
	  
	}
	return true;
} 

bool single_face_embedding(cv::Mat &face_img, cv::Mat &float_feature){
	string facenet_input_layer1="input";
    string facenet_input_layer2="phase_train";
	string facenet_output_layer="embeddings";
	std::vector<Tensor> facenet_outputs;
	  Tensor phase(DT_BOOL, TensorShape());
	  phase.scalar<bool>()()=false;
	  cv::Mat new_align(face_img.rows, face_img.cols, CV_32FC3, cv::Scalar::all(0.0)); 
	    prewhiten(face_img, new_align);
		vector<cv::Mat> input_facenet;
	  input_facenet.push_back(new_align);
	  Tensor in_facenet = convert_tensor(input_facenet);
	Status run_status = session4->Run({{facenet_input_layer1, in_facenet},{facenet_input_layer2,phase}},
                                   {facenet_output_layer}, {}, &facenet_outputs);
								   
	if (!run_status.ok()) {
             LOG(ERROR) << "Running model failed: " << run_status;
             return false;
        }
	for(int j=0;j<facenet_outputs[0].dim_size(1);++j){
               float_feature.ptr<float>(0)[j]=facenet_outputs[0].tensor<float,2>()(0,j); 
		}							   
								   
	return true;
}

