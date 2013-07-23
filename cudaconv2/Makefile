SRC = conv_util.cu cudaconv2_wrap.cu filter_acts.cu img_acts.cu weight_acts.cu
OBJ = $(patsubst %.cu,%.o,${SRC})

NVCC_OPTS := --pre-include undef.h #--compiler-bindir=./nvcc-bindir
CXX_FLAGS := --compiler-options='-fPIC -ggdb2'
INCLUDES := -I/usr/include/python2.7 -I. -I./cutil

all: _cudaconv2.so

_cudaconv2.so: ${OBJ}
	nvcc -shared -o $@ $^

cudaconv2_wrap.cu: cudaconv2.i Makefile
	swig -python -threads -c++ -o $@ ${INCLUDES} $<

%.o : %.cu Makefile
	nvcc -arch=sm_20 -O3 -g -c -o $@ $<  ${NVCC_OPTS} ${CXX_FLAGS} ${INCLUDES}

clean:
	rm -f *.o *.so
