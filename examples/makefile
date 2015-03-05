FLAGS+=-std=c++11 -O3
#FLAGS+=-march=core-avx2
FLAGS+=-mavx

stream : stream.cpp
	g++ ${FLAGS} -I ../include stream.cpp -fopenmp -o stream
