CC=mpic++
HOME=/home/liberty
MKLROOT=/opt/intel/mkl
all:kmeans

kmeans:main.cpp Manager.cpp Manager.h makefile
	$(CC) -Wall -O3 -o kmeans main.cpp Manager.cpp -lsatlas
#%.o:%.cp
#	$(CC) -Wall -I/home/liberty/lib/armadillo-6.700.4/include/ -O3 -c $*.cpp

.PHONY:clean

clean:
	@rm kmeans
	
