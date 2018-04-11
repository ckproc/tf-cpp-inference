bazel build -c opt --copt=-mavx  --copt=-msse4.1 --copt=-msse4.2 --copt="-fPIC" :ceface
#bazel build -c opt --config=cuda :libface_recognition.so
