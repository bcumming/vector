CXX=g++
#CXX=clang++

FLAGS+=-std=c++11 -O3
FLAGS+=-DWITH_CYME
FLAGS+=-fopenmp

FLAGS+=-msse4.1 -D__CYME_SIMD_VALUE__=sse
#FLAGS+=-mavx -D__CYME_SIMD_VALUE__=avx
#FLAGS+=-march=core-avx2  -D__CYME_SIMD_VALUE__=avx -D__FMA__

FLAGS+=-I../external -I../include

### pick your boost path
#FLAGS+=-isystem /Users/bcumming/software/boost/1_55_0/
#FLAGS+=-isystem /apps/santis/boost/1.54/gcc_481/include/
FLAGS+=-isystem /users/bcumming/repos/boost_1_55_0

stream.exe : stream.cpp
	${CXX} ${FLAGS} stream.cpp -o stream.exe

clean :
	rm -f *.exe
