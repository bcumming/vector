CXX=clang++

##### SANTIS #####
#GTEST_PATH=/project/csstaff/bcumming/bbp/gtest
#GTEST_LIB=$(GTEST_PATH)/santis
#INCLUDE_ARGS=-I$(GTEST_INCLUDE) -I../include -I$(GCCROOT)/include/g++/ -I$(GCCROOT)/include/g++/x86_64-suse-linux/
#GCCROOT=/opt/gcc/4.8.2/snos

##### MAC #####
GTEST_PATH=/Users/bcumming/install/googletest
GTEST_LIB=$(GTEST_PATH)/lib
GTEST_INCLUDE=$(GTEST_PATH)/include
INCLUDE_ARGS=-I$(GTEST_INCLUDE) -I../include

LINK_ARGS=-L$(GTEST_LIB) -lgtest -pthread

ARGS=-O2 -std=c++11

TESTS=storage_unittest.o range_unittest.o hostcoordinator_unittest.o


all : driver.exe

hostcoordinator_unittest.o: hostcoordinator_unittest.cpp
	$(CXX) hostcoordinator_unittest.cpp -c -o hostcoordinator_unittest.o $(ARGS) $(INCLUDE_ARGS)

storage_unittest.o: storage_unittest.cpp
	$(CXX) storage_unittest.cpp -c -o storage_unittest.o $(ARGS) $(INCLUDE_ARGS)

range_unittest.o: range_unittest.cpp
	$(CXX) range_unittest.cpp -c -o range_unittest.o $(ARGS) $(INCLUDE_ARGS)

driver.exe: driver.cpp $(TESTS)
	$(CXX) driver.cpp -o driver.exe $(ARGS) $(TESTS) $(INCLUDE_ARGS) $(LINK_ARGS)

clean:
	rm -f *.o
	rm -f driver.exe
