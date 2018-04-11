#bazel build -c opt --copt=-mavx  --copt=-msse4.1 --copt=-msse4.2 --copt="-fPIC" :libdetect_embed.so
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-msse4.1 --copt=-msse4.2 --copt=-mfma --copt=-mfpmath=both --copt="-fPIC" :libdetect_embed.so
#bazel build -c opt --copt="-fPIC" :libdetect_embed.so
