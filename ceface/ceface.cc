
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
typedef std::map<string, int> Dict;



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
    int node_count = graph_def.node_size();
  //for(int i=0;i<node_count;i++){
	//  auto n = graph_def.node(i);
	//  cout<<n.name()<<"	";
  //}
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

vector<string> &split(const string &str, char delim, vector<string> &elems) {
    istringstream iss(str);
    for (string item; getline(iss, item, delim); )
        elems.push_back(item);
    return elems;
}

void load_matchlist(string match_list,Dict &match_query){
	ifstream fin(match_list);
	string s;
	while(getline(fin,s)){
		vector<string> result;
		split(s,' ', result);
		match_query[result[0]]=atoi(result[1].c_str());
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

int main(int argc, char** argv) {
  //cout<<"p1"<<endl;
  using namespace cv;
  int arg_channel = 1;
  string model = argv[arg_channel++];    //model
  string db_dir = argv[arg_channel++];
  
  //cout<<"p1"<<endl;
 
  string input="input/image";
  string input2="Placeholder_1";
  string output="MobilenetV1/Logits/SpatialSqueeze";
  string graph_face_filter=model+"/"+"filter.pb";
  std::unique_ptr<tensorflow::Session> session;
  cout<<graph_face_filter<<endl;
  Status load_graph_status = LoadGraph(graph_face_filter, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }
  
   
  //float input_mean = 127.5;
  //float input_std=0.0078125;
  
  //cv::resize(image,image,cv::Size(20,20));
  //time_t loade_time=time(NULL);
  //t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
  //cout<<"load_time:"<<t<<endl;
  db_dir=db_dir+"/*.*";
  vector<string> db_path=glob(db_dir);
  cout<<"Running forward pass on db images"<<endl;
  //cv::Mat emb_array=Mat(db_path.size(), 128, CV_32FC1, Scalar::all(0.0));
  //time_t start_time=time(NULL);
  //t=(double)cv::getTickCount();
    Dict match_query;
	//string match_list = "/home/ckp/data/ceface/validation_list.txt";
	//load_matchlist(match_list, match_query);
	int hit=0;
	int num=0;
	int p=0;
for(int i=0;i<db_path.size();i++)
    {
      string image_path = db_path[i];
	  //cout<<image_path<<endl;
	  
      Mat image = imread(image_path,-1);
	  resize(image,image,cv::Size(224,224));
	  image.convertTo( image, CV_32FC3);
	  

      if(image.empty()){
        cout<<"Error:Image cannot be loaded!"<<endl;
        continue;                       
		}
		
      else if(image.channels()<2){
        cout<<"Unable to align"<<endl;
        continue;                       
		}
		
	  else if(image.channels()==4){
		  //cout<<"p3"<<endl;
		  cv::cvtColor(image, image, CV_BGRA2BGR);
	  }
	  
	  getname(image_path);
	  //string query_full=image_path+".png";
	 // cout<<"p1"<<endl;
      std::vector<Tensor> facenet_outputs;
	  vector<cv::Mat> input_facenet;
	  input_facenet.push_back(image);
	   //cout<<"p2"<<endl;
	  Tensor in_facenet = convert_tensor(input_facenet);
	  //cout<<"p3"<<endl;
	  Tensor phase(DT_BOOL, TensorShape());
	  phase.scalar<bool>()()=false;
      Status run_status = session->Run({{input, in_facenet},{input2, phase}},
                                   {output}, {}, &facenet_outputs);
        if (!run_status.ok()) {
             LOG(ERROR) << "Running model failed: " << run_status;
             continue;
        }
		
		//cout<<facenet_outputs[0].dims()<<endl;
		//cout<<facenet_outputs[0].dim_size(0)<<" "<<facenet_outputs[0].dim_size(1)<<endl;
		if(facenet_outputs[0].tensor<float,2>()(0,0)>=facenet_outputs[0].tensor<float,2>()(0,1)){
			cout<<image_path<<"	"<<1<<"	"<<facenet_outputs[0].tensor<float,2>()(0,0)<<"	"<<facenet_outputs[0].tensor<float,2>()(0,1)<<endl;
			//p+=1;
		}
		else{
			cout<<image_path<<"	"<<0<<"	"<<facenet_outputs[0].tensor<float,2>()(0,0)<<"	"<<facenet_outputs[0].tensor<float,2>()(0,1)<<endl;
                       p+=1;
		}	/*if(match_query[query_full]==0){
				hit++;
			}
			else{
				cout<<image_path<<endl;
			}
		else{
			if(match_query[query_full]==1){
				hit++;
			}
			else{
				cout<<image_path<<endl;
			}
		}
		num++;
        //cout<<facenet_outputs[0].tensor<float,2>()(0,0)<<endl;
        //cout<<facenet_outputs[0].tensor<float,2>()(0,1)<<endl;
		*/
}
cout<<"total:"<<db_path.size()<<endl;
cout<<"p:"<<p<<endl;
cout<<"accu:"<<float(p)/db_path.size()<<endl;

  //cout<<"accu:"<<float(hit)/num<<endl;

  return 0;
  

}
