cc_binary(
 name = "libdetect_embed.so",
 srcs = ["detect_embed.cc","detect_embed.hpp"],
    linkopts = 
		["-shared","-lopencv_core","-lopencv_imgproc", "-lopencv_highgui", ],
    linkshared=1,
	
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
