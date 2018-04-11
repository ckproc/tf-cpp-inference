#bazel build -c opt --copt="-fPIC" :libcame_mtcnn.so
#bazel build -c opt --copt=-mavx2 --copt=-mavx  --copt=-msse4.1 --copt=-msse4.2 --copt=-mfma --copt=-mfpmath=both --copt="-fPIC" :libembed_face_feature
bazel build -c opt --config=cuda --config=monolithic --copt="-fPIC" :libembed_face_feature
#bazel build --config=mkl -c opt --copt="-fPIC" :libembed_face_feature
