GTEST_PATH=/project/csstaff/bcumming/bbp/gtest
GTEST_LIB=$(GTEST_PATH)/santis
GTEST_INCLUDE=$(GTEST_PATH)/include

LINK_ARGS=-L$(GTEST_LIB) -lgtest -pthread
INCLUDE_ARGS=-I$(GTEST_INCLUDE)

ARGS=-O2

TESTS=gtest.o


all : driver

gtest.o: gtest.cpp
	g++ gtest.cpp -c -o gtest.o $(ARGS) $(INCLUDE_ARGS)

driver: driver.cpp $(TESTS)
	g++ driver.cpp -o driver $(ARGS) $(TESTS) $(INCLUDE_ARGS) $(LINK_ARGS) -o driver

clean:
	rm -f *.o
	rm -f driver
