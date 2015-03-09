CXX=g++
#CXX=g++-4.8
#CXX=clang++
FLAGS+=-std=c++11 -O3
FLAGS+=-mavx -D__CYME_SIMD_VALUE__=avx
#FLAGS+=-msse4.2
FLAGS+=-DWITH_CYME
#FLAGS+=-march=core-avx2
#FLAGS+=-fopenemp
FLAGS+=-I../external
#FLAGS+=-isystem /Users/bcumming/software/boost/1_55_0/
FLAGS+=-isystem /apps/santis/boost/1.54/gcc_481/include/

stream.exe : stream.cpp
	${CXX} ${FLAGS} -I ../include stream.cpp -o stream.exe

clean :
	rm -f *.exe
