FLAGS+=-std=c++11 -O3
FLAGS+=-march=core-avx2
#FLAGS+=-mavx

stream.gnu : stream.cpp
	g++ ${FLAGS} -I ../include stream.cpp -fopenmp -o stream.gnu

stream.intel : stream.cpp
	icpc ${FLAGS} -I ../include stream.cpp -openmp -o stream.intel

clean :
	rm -rf stream.gnu
	rm -rf stream.intel
