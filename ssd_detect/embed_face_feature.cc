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

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <glob.h>
#include <vector>
#include <math.h>
#include "time.h"


using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using namespace tensorflow;
using namespace tensorflow::ops;
using namespace std;
using namespace cv;

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
void nms_cpu(vector<BoundingBox>& boxes, float threshold, NMS_TYPE type, vector<BoundingBox>& filterOutBoxes)
{
    filterOutBoxes.clear();
    if(boxes.size() == 0)
        return;
    //descending sort
    sort(boxes.begin(), boxes.end(), CmpBoundingBox() );
    vector<size_t> idx(boxes.size());
    //std::iota(idx.begin(), idx.end(), 0);//create index
    for(int i = 0; i < idx.size(); i++)
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
        for(int i = 1; i < tmp.size(); i++)
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
	for(int y = 0; y < h; y ++)
    {
        for(int x = 0; x < w; x ++)
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
    for(int i = 0; i < boxes.size(); i ++)
    {
        float score = cls.tensor<float,2>()(i,1);//[i * 2 + 1];
        if ( score > threshold )
        {
            BoundingBox box = boxes[i];
            float w = boxes[i].y2 - boxes[i].y1 + 1;
            float h = boxes[i].x2 - boxes[i].x1 + 1;
            if( points.dims() > 1)
            {
                for(int p = 0; p < 5; p ++)
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
    for(int i = 0; i < boxes.size(); i ++)
    {
        float score = cls.tensor<float,2>()(i,1);//[i * 2 + 1];
        if ( score > threshold )
        {
            BoundingBox box = boxes[i];
            float w = boxes[i].y2 - boxes[i].y1 + 1;
            float h = boxes[i].x2 - boxes[i].x1 + 1;
            if( points.dims() > 1)
            {
                for(int p = 0; p < 5; p ++)
                {
                    //box.points_x[p] = points.tensor<float,2>()(i,5+p) * w + boxes[i].x1 - 1;
                    box.points_x[p] = points.tensor<float,2>()(i,5+p);
                    //box.points_x[p] = points.tensor<float,2>()(i,5+p) * w  - 1;
                    //box.points_y[p] = points.tensor<float,2>()(i,p) * h + boxes[i].y1-1;
                    box.points_y[p] = points.tensor<float,2>()(i,p);
                    //box.points_y[p] = points.tensor<float,2>()(i,p) * h -1;
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
		for(int n = 0; n < totalBoxes.size(); n++)
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
		
}
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
	for(int i=0;i<size;i++){
		float value=float(pixelPtr[i]);
		temp+=pow(value-mean,2.0);
	}
	temp=sqrt(temp/size);
	float std_adj=std::max(temp,float(1.0/sqrt(size)));
	//elenment-wise operation
	for(int j=0;j<new_crop.rows;j++){
		for(int k=0;k<new_crop.cols;k++){
			new_crop.at<Vec3f>(j,k)[0]=(float(crop.at<Vec3b>(j,k)[0])-mean)/std_adj;
			new_crop.at<Vec3f>(j,k)[1]=(float(crop.at<Vec3b>(j,k)[1])-mean)/std_adj;
			new_crop.at<Vec3f>(j,k)[2]=(float(crop.at<Vec3b>(j,k)[2])-mean)/std_adj;
		}
	}
	
}

void getname(string &match_face){
	const size_t last_slash_idx = match_face.find_last_of("/");
	if (std::string::npos != last_slash_idx)
	{
		match_face.erase(0, last_slash_idx + 1);
		}
		const size_t period_idx = match_face.rfind('.');
		if (std::string::npos != period_idx)
	{
		match_face.erase(period_idx);
	}
	
}

#define IMAGE_WIDTH_STD 93
#define IMAGE_HEIGHT_STD 109
int align_mtcnn(cv::Mat &image, std::unique_ptr<tensorflow::Session> &session1, std::unique_ptr<tensorflow::Session> &session2,
   std::unique_ptr<tensorflow::Session> &session3, const int image_size, const int margin, cv::Mat &aligned_img,string image_path)
{
  string pnet_input_layer = "pnet/input";
  string pnet_output_layer1 = "pnet/conv4-2/BiasAdd";
  string pnet_output_layer2 ="pnet/prob1"; 
  string rnet_input_layer = "rnet/input";
  string rnet_output_layer1 ="rnet/conv5-2/conv5-2";
  string rnet_output_layer2 ="rnet/prob1";  
  string onet_input_layer="onet/input";
  string onet_output_layer1="onet/conv6-2/conv6-2";
  string onet_output_layer3="onet/conv6-3/conv6-3";
  string onet_output_layer2="onet/prob1"; 
  std::vector<Tensor> pnet_outputs;
  std::vector<Tensor> rnet_outputs;
  std::vector<Tensor> onet_outputs;
  int minsize =16;
  float P_thres = 0.5;
  float R_thres = 0.6;
  float O_thres =0.7;
  //float factor = 0.5;
  float factor = 0.8;
  
  float input_mean = 127.5;
  float input_std=0.0078125;
  
  //int width=300;
  //int height=int((float)(image.rows*width/image.cols));
//cv::resize(image,image,cv::Size(width,height));
    cv::Mat c_image;
    image.convertTo( c_image, CV_32FC3, input_std, -input_mean * input_std);
	//image.convertTo(image,CV_8UC3);
	image=image.t();
    c_image=c_image.t(); 
    int img_H = c_image.rows;
    int img_W = c_image.cols;
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
	for(int i = 0; i < all_scales.size(); i ++)
    {
		cur_images.clear();
		cur_scale=all_scales[i];
		int hs = cvCeil(img_H * cur_scale);
        int ws = cvCeil(img_W * cur_scale);		
		cv::resize(c_image, cur_image, cv::Size(ws, hs));
		cur_images.push_back(cur_image);
		Tensor inputImg = convert_tensor(cur_images);
		Status run_status = session1->Run({{pnet_input_layer, inputImg}},{pnet_output_layer1,pnet_output_layer2}, {}, &pnet_outputs);
        if (!run_status.ok())
			{      //cout<<"sess1"<<endl;
		           return -1;
				   }
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
        for(int i = 0; i < globalFilterBoxes.size(); i ++)
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
      		  
			  //for(int k=0;k<totalBoxes.size();k++){
	//cv::rectangle(image, cv::Point(totalBoxes[k].x1, totalBoxes[k].y1), cv::Point(totalBoxes[k].x2, totalBoxes[k].y2), cv::Scalar(255, 0, 0), 2);
	//}
	//cv::imshow("stage1_test", image);
   // cv::waitKey(0);;
	//stage2 rnet:
	//cout<<"sess1"<<" "<<totalBoxes.size()<<endl;
	if(totalBoxes.size() > 0)
    {
		vector<cv::Mat> r_candidates;
		buildInput(r_candidates, c_image, totalBoxes, cv::Size(24,24), img_H, img_W);
		Tensor rnet_input = convert_tensor(r_candidates);
		Status run_status = session2->Run({{rnet_input_layer,rnet_input }},{rnet_output_layer1,rnet_output_layer2}, {}, &rnet_outputs);
        if (!run_status.ok())
			{ 
		       //cout<<"sess2"<<endl;
		          return -1;   }
		//cout<<rnet_outputs[0].dim_size(0)<<","<<rnet_outputs[0].dim_size(1)<<endl;
		vector<BoundingBox> filterOutBoxes;
		Tensor empty;
        filteroutBoundingBox(totalBoxes, rnet_outputs[0], rnet_outputs[1], empty, R_thres, filterOutBoxes);	
        nms_cpu(filterOutBoxes, 0.7, UNION, totalBoxes);
		
	}
      ///cout<<"p5"<<endl;
	  //cout<<totalBoxes.size()<<endl;
	  //for(int k=0;k<totalBoxes.size();k++){
	//cv::rectangle(image, cv::Point(totalBoxes[k].x1, totalBoxes[k].y1), cv::Point(totalBoxes[k].x2, totalBoxes[k].y2), cv::Scalar(255, 0, 0), 2);
	//}
	//cv::imshow("stage2_test", image);
    //cv::waitKey(0);;
	//stage3: onet
	//cout<<"sess2"<<" "<<totalBoxes.size()<<endl;
	if(totalBoxes.size()>0)
	{
		vector<cv::Mat> o_candidates;
		
		buildInput(o_candidates, c_image, totalBoxes, cv::Size(48,48), img_H, img_W);
		
		Tensor onet_input = convert_tensor(o_candidates);
	
		Status run_status = session3->Run({{onet_input_layer, onet_input}},{onet_output_layer1,onet_output_layer2,onet_output_layer3}, {}, &onet_outputs);
        if (!run_status.ok()) 
		{ //cout<<"sess3"<<endl;
	              return -1;
				  }
		vector<BoundingBox> filterOutBoxes;

        filteroutBoundingBox1(totalBoxes, onet_outputs[0], onet_outputs[1], onet_outputs[2], O_thres, filterOutBoxes);
		
        nms_cpu(filterOutBoxes, 0.7, MIN, totalBoxes);
	}
	//for(int k=0;k<totalBoxes.size();k++){cv::rectangle(image, cv::Point(totalBoxes[k].x1, totalBoxes[k].y1), cv::Point(totalBoxes[k].x2, totalBoxes[k].y2), cv::Scalar(255, 0, 0), 2);}
	//cv::imshow("stage3_test", image);  cv::waitKey(0);;
	//cout<<"totalsize:"<<" "<<totalBoxes.size()<<endl;
	if(totalBoxes.size()==0){ 
        
		return -1;
	}
	//tmp = c_image.t();
    
	//align image:  const int image_size, const int margin, Tensor &aligned_img
	BoundingBox cand_rect;
	//cand_rect=totalBoxes[0];
	
	float img_cw=img_W/2.0;
	float img_ch=img_H/2.0;
	float box_size=(totalBoxes[0].x2-totalBoxes[0].x1)*(totalBoxes[0].y2-totalBoxes[0].y1);
	float off_sq = (pow((totalBoxes[0].x1+totalBoxes[0].x2)/2.0-img_cw, 2.0)+pow((totalBoxes[0].y1+totalBoxes[0].y2)/2.0-img_ch, 2.0))*2.0;
	float temp_value = box_size-off_sq;
	//cout<<"value_1:"<<temp_value<<"    "<<"size_1:"<<box_size<<"  "<<"sq_1:"<<off_sq<<endl;
	int temp_index=0;
	if(totalBoxes.size()>1){
		for(int index=1;index<totalBoxes.size();index++){
			
			box_size=(totalBoxes[index].x2-totalBoxes[index].x1)*(totalBoxes[index].y2-totalBoxes[index].y1);
			off_sq = (pow((totalBoxes[index].x1+totalBoxes[index].x2)/2.0-img_cw, 2.0)+pow((totalBoxes[index].y1+totalBoxes[index].y2)/2.0-img_ch, 2.0))*2.0;
			float index_value=box_size-off_sq;
			//cout<<"value_"<<index<<":"<<index_value<<"    "<<"size_:"<<index<<":"<<box_size<<"  "<<"sq_"<<index<<":"<<off_sq<<endl;
			if(index_value>temp_value){
				temp_index=index;
				temp_value=index_value;
			}
		}
		
	}
	cand_rect=totalBoxes[temp_index];
	cand_rect.x1=std::max(cand_rect.x1, float(0.0));
	cand_rect.y1=std::max(cand_rect.y1, float(0.0));
	cand_rect.x2=std::min(cand_rect.x2, float(img_W));
	cand_rect.y2=std::min(cand_rect.y2, float(img_H));
	//crop image to cand_rect
	cv::Rect roi;
	roi.x = cand_rect.x1;
    roi.y = cand_rect.y1;
    roi.width = cand_rect.x2-cand_rect.x1;
    roi.height = cand_rect.y2-cand_rect.y1;
	
	//ofstream infile;
	//infile.open("./test_front.txt",ios::app);
	cv::Mat crop = image(roi);
	cv::resize(crop, crop,cv::Size(image_size,image_size));
	//for(int k=0;k<totalBoxes.size();k++){
	//cv::rectangle(image, cv::Point(totalBoxes[k].x1, totalBoxes[k].y1), cv::Point(totalBoxes[k].x2, totalBoxes[k].y2), cv::Scalar(255, 0, 0), 2);
	//}
	//cv::imshow("stage1_test", image);
    //cv::waitKey(0);;
	
	//cv::rectangle(image, cv::Point(cand_rect.x1, cand_rect.y1), cv::Point(cand_rect.x2, cand_rect.y2), cv::Scalar(0, 0, 255), 2);
	
	//cv::imshow("stage1_test", image);
    //cv::waitKey(0);;
	//string cur_path="/home/ckp/New_facenet/data/cpp_db/"+image_path.substr(29); 
	//string cur_path="/home/ckp/New_facenet/data/cpp_query/"+image_path.substr(32); 
	//image=image.t();
	//cv::imwrite(cur_path,image);

	//cout<<cur_path<<endl;
	//cv::Mat new_crop(crop.rows,crop.cols,CV_32FC3, cv::Scalar::all(0.0)); 
	//prewhiten(crop,new_crop);
	//aligned_img=new_crop;//.clone();
	//imshow("1",aligned_img);
	//waitKey(0);
	
	//vector<Point2f> coord5points;
	/*
	Point2f leftEye=Point2f(cand_rect.points_y[0], cand_rect.points_x[0]);
	Point2f rightEye=Point2f(cand_rect.points_y[1], cand_rect.points_x[1]);
	Point2f eyesCenter=Point2f(cand_rect.points_y[2], cand_rect.points_x[2]);
	double dy = (rightEye.y - leftEye.y);
    double dx = (rightEye.x - leftEye.x);
    double len = sqrt(dx*dx + dy*dy);
    double angle = atan2(dy, dx) * 180.0 / CV_PI; // Convert from radians to degrees.
    double scale = 1;
	*/
	//cout<<cand_rect.points_y[0]<<","<< cand_rect.points_x[0]<<" "<<cand_rect.points_y[1]<<","<< cand_rect.points_x[1]<<endl;
	
	//for(int i=0;i<5;++i){
	//	infile<<to_string(cand_rect.points_y[i])<<" "<<to_string(cand_rect.points_x[i])<<" ";
	//}
	//infile<<"1"<<endl;
	//infile.close();
	//double dst_landmark[10] = {
    //30.2946, 65.5318, 48.0252, 33.5493, 62.7299,
   // 51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };        //base points
	//for (int i = 0; i < 5; i++)
	//{
		//coord5points.push_back(Point2f(points_x[i], points_y[i]));
	//	coord5points.push_back(Point2f(dst_landmark[i], dst_landmark[i + 5]));
	//}
	
	//vector<Point2f> facial5points;
	//for (int i = 0; i < 5; i++)
	//{
	//	facial5points.push_back(Point2f(cand_rect.points_y[i], cand_rect.points_x[i]));
	//}
	//Mat warp_mat, roImg;
	//warp_mat = estimateRigidTransform(facial5points, coord5points, false);
	
	//Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
	//if (rot_mat.empty())
	//	{
	//		std::cout << "NULL" << std::endl;
	//		return 0;
	//	}
	//cropImg = Mat::zeros(IMAGE_HEIGHT_STD, IMAGE_WIDTH_STD, aligned_img.type());
	//roImg = Mat::zeros(img_W, img_H, image.type());
	//warpAffine(image.t(), roImg, rot_mat, roImg.size());
	//imshow("1",roImg);
	//waitKey(0);
	//cv::Mat croprec = roImg(roi);
	//cv::resize(croprec,croprec,cv::Size(160,160));
	//imshow("1",croprec);
	//waitKey(0);
	//cv::Mat new_crop(croprec.rows,croprec.cols,CV_32FC3, cv::Scalar::all(0.0)); 
	//prewhiten(croprec,new_crop);
	//aligned_img=new_crop;
	//imshow("1",aligned_img);
	//waitKey(0);
	//getname(image_path);
	//string output_path = "/home/ckp/Cpp_test/align_face/align_data/query/"+image_path+".jpg";
    //string output_path = "/home/ckp/face_alignment/nrp/"+image_path+".jpg";
	//imwrite(output_path,croprec);
	//waitKey(0);
    return 0;
	
	

   
}

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
  //cout<<"p1"<<endl;
  //graph::SetDefaultDevice("/gpu:1,/cpu:0",&graph_def);
  //cout<<"p2"<<endl;
  
  opts.config.mutable_gpu_options()->set_visible_device_list("1");
  //opts.config.mutable_cpu_options()->set_visible_device_list("0");
  //opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.1);
  //opts.config.mutable_gpu_options()->set_allow_growth(true);
  opts.config.mutable_gpu_options()->set_allow_growth(true);
  session->reset(tensorflow::NewSession(opts));

  //session->reset(tensorflow::NewSession(opts));
  cout<<"p3"<<endl;
  int node_count = graph_def.node_size();
  for(int i=0;i<node_count;i++){
	  auto n = graph_def.node(i);
	  //cout<<n.name()<<"	";  
	 // if(){
	//	  auto node = graph_def.mutable_node(i);
	 //     node->set_device("/cpu:0");
	//	  continue;
	 // }
	  
	  //auto node = graph_def.mutable_node(i);
	 // cout<<node->device()<<endl;
	  //node->set_device("/gpu:1");
	  
	  //if (n.name().find("nWeights") != std::string::npos) {
	  //vNames.push_back(n.name());
  }
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  cout<<"all"<<endl;
  
  return Status::OK();
} 

/*
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
				 }
*/
				 
				 
int main(int argc, const char *argv[]) {
	
  using namespace cv;
  int arg_channel = 1;
  string db_dir = argv[arg_channel++];   //db_dir
  string model = argv[arg_channel++];    //model
  //string output_file = argv[arg_channel++];  //output_file default='data/db_embed.yml
  int image_size = 160;//=atoi(argv[arg_channel++]);  //image_size default=160
  int margin = 0;//atoi(argv[arg_channel++]); //=0
 // float gpu_memory_fraction = atof(argv[arg_channel++]);
  //int input_width = 20;
  //int input_height = 20;
 
  

  string graph_ssd=model+"/"+"frozen_inference_graph.pb";
  //string graph_ssd=model+"/"+"saved_model.pb";
  string ssd_input = "image_tensor:0";
  string ssd_detection_box = "detection_boxes:0";
  string ssd_score ="detection_scores:0"; 
  string ssd_class ="detection_classes:0"; 
  string ssd_num_detections ="num_detections:0"; 
  //string graph_facenet=model+"/"+"facenet.pb";
  //string graph_facenet=model+"/"+"facenet.pb";
  //time_t loads_time=time(NULL);
  //double t=(double)cv::getTickCount();

  
  //opts.config.mutable_gpu_options()->set_allow_growth(true);
  std::unique_ptr<tensorflow::Session> session;
  
  Status load_graph_status = LoadGraph(graph_ssd, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }
  
   
 

  db_dir=db_dir+"/*.*";
  vector<string> db_path=glob(db_dir);
  cout<<"Running forward pass on db images"<<endl;
  //cv::Mat emb_array=Mat(db_path.size(), 128, CV_32FC1, Scalar::all(0.0));
  
  
int count =0;
double total=0.0;
for(int i=0;i<db_path.size();i++)
    {
      string image_path=db_path[i];
	  cout<<image_path<<endl;
      Mat image = imread(image_path,-1);
	  //cout<<"p1"<<endl;
      if(image.empty()){
        cout<<"Error:Image cannot be loaded!"<<endl;
        continue;                       
		}
		
      else if(image.channels()<2){
        cout<<"this is gray image"<<endl;
        continue;                       
		}
		
	  else if(image.channels()==4){
		  cv::cvtColor(image, image, CV_BGRA2BGR);
	  }
	  count++;
   // cv::Mat aligned_img;
	//double t=(double)cv::getTickCount();
	std::vector<Tensor> facenet_outputs;
	vector<cv::Mat> input_facenet;
	//cout<<"p2"<<endl;
	//image.convertTo(image, CV_32FC3);
      
	input_facenet.push_back(image);
      
	Tensor in_facenet = convert_tensor(input_facenet);
    //cout<<"p3"<<endl;	
	//cout<<"p1"<<endl;
	double t=(double)cv::getTickCount();
    Status run_status = session->Run({{ssd_input, in_facenet}},
	                   {ssd_detection_box,ssd_score,ssd_class,ssd_num_detections}, {}, &facenet_outputs);
	

if (!run_status.ok()) {
             LOG(ERROR) << "Running model failed: " << run_status;
             continue;
        }	
    //cout<<"p2"<<endl;		
    //cout<<"p22"<<endl;	
//cout<<facenet_outputs.size()<<endl;
	cout<<facenet_outputs[0].dim_size(0)<<" "<<facenet_outputs[0].dim_size(1)<<" "<<facenet_outputs[0].dim_size(2)<<" "<<facenet_outputs[0].dim_size(3)<<endl;
	//cout<<facenet_outputs[1].dim_size(0)<<" "<<facenet_outputs[1].dim_size(1)<<" "<<facenet_outputs[1].dim_size(2)<<" "<<facenet_outputs[1].dim_size(3)<<endl;
	//cout<<facenet_outputs[2].dim_size(0)<<" "<<facenet_outputs[2].dim_size(1)<<" "<<facenet_outputs[2].dim_size(2)<<" "<<facenet_outputs[2].dim_size(3)<<endl;
	//cout<<facenet_outputs[3].dim_size(0)<<" "<<facenet_outputs[3].dim_size(1)<<" "<<facenet_outputs[3].dim_size(2)<<" "<<facenet_outputs[3].dim_size(3)<<endl;
    //cout<<facenet_outputs[0].dims()<<" "//facenet_outputs[1].dims()<<" "<<facenet_outputs[2].dims<<" "<<facenet_outputs[3].dims<<endl;
	cout<<facenet_outputs[0].tensor<float,3>()(0,0,0)<<" "<<facenet_outputs[0].tensor<float,3>()(0,0,1)<<" "<<facenet_outputs[0].tensor<float,3>()(0,0,2)<<" "<<facenet_outputs[0].tensor<float,3>()(0,0,3)<<endl;
	//cout<<facenet_outputs[0].tensor<float,3>()(0,1,0)<<" "<<facenet_outputs[0].tensor<float,3>()(0,1,1)<<" "<<facenet_outputs[0].tensor<float,3>()(0,1,2)<<" "<<facenet_outputs[0].tensor<float,3>()(0,1,3)<<endl;
	//cout<<facenet_outputs[0].tensor<float,3>()(0,2,0)<<" "<<facenet_outputs[0].tensor<float,3>()(0,2,1)<<" "<<facenet_outputs[0].tensor<float,3>()(0,2,2)<<" "<<facenet_outputs[0].tensor<float,3>()(0,2,3)<<endl;
	//cout<<facenet_outputs[0].tensor<float,3>()(0,0,0)<<" "<<facenet_outputs[0].tensor<float,3>()(0,0,1)<<" "<<facenet_outputs[0].tensor<float,3>()(0,0,2)<<" "<<facenet_outputs[0].tensor<float,3>()(0,0,3)<<endl;
	cout<<facenet_outputs[1].tensor<float,2>()(0,0)<<endl;//<<" "<<facenet_outputs[0].tensor<float,3>()(0,0,1)<<" "facenet_outputs[0].tensor<float,3>()(0,0,2)<<" "<<facenet_outputs[0].tensor<float,3>()(0,0,3)<<endl;
	t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
	if(count!=1){
		total+=t;
	}
	
  
}
    
  cout<<"aver time:"<<total/float(count-1)<<endl;
 
  return 0;
  
}
