cc_binary(
 name = "libembed_face_feature",
 srcs = ["embed_face_feature.cc"],
    linkopts = 
		["-Wl,--version-script=tensorflow/tf_version_script.lds", "-lopencv_core","-lopencv_imgproc", "-lopencv_highgui", 
		],
 
    copts = ["-I/usr/local/include/", "-O3"],

 deps = [
 "//tensorflow/cc:cc_ops",
 "//tensorflow/cc:client_session",
 "//tensorflow/core:tensorflow",
 ],
  visibility=["//visibility:public"]
)
