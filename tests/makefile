GTEST_PATH=/project/csstaff/bcumming/bbp/gtest
GTEST_LIB=$(GTEST_PATH)/santis
GTEST_INCLUDE=$(GTEST_PATH)/include

LINK_ARGS=-L$(GTEST_LIB) -lgtest -pthread
INCLUDE_ARGS=-I$(GTEST_INCLUDE) -I../vector

ARGS=-O2

TESTS=storage_unittest.o


all : driver

storage_unittest.o: storage_unittest.cpp
	g++ storage_unittest.cpp -c -o storage_unittest.o $(ARGS) $(INCLUDE_ARGS)

driver.exe: driver.cpp $(TESTS)
	g++ driver.cpp -o driver.exe $(ARGS) $(TESTS) $(INCLUDE_ARGS) $(LINK_ARGS) -o driver

clean:
	rm -f *.o
	rm -f driver.exe
