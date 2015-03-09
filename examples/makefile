#CXX=g++
#CXX=g++-4.8
CXX=clang++
FLAGS+=-std=c++11 -O3
FLAGS+=-mavx
#FLAGS+=-msse4.2
FLAGS+=-DWITH_CYME
#FLAGS+=-march=core-avx2
#FLAGS+=-fopenemp
FLAGS+=-I../external -isystem /Users/bcumming/software/boost/1_55_0/

stream.exe : stream.cpp
	${CXX} ${FLAGS} -I ../include stream.cpp -o stream.exe

clean :
	rm -f *.exe
