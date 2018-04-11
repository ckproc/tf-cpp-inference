cc_binary(
 name = "ceface",
 srcs = ["ceface.cc"],
    linkopts = 
		["-lopencv_core","-lopencv_imgproc", "-lopencv_highgui"],
	
    copts = ["-I/usr/local/include/", "-O3"],

 deps = [
 "//tensorflow/cc:cc_ops",
 "//tensorflow/cc:client_session",
 "//tensorflow/core:tensorflow",
 "//tensorflow/core:all_kernels",
 "//tensorflow/core:core_cpu",
 "//tensorflow/core:framework",
 "//tensorflow/core:lib",
 "//tensorflow/core:protos_all_cc",
 ],
  visibility=["//visibility:public"]
)
