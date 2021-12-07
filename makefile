CC=mpic++
ARMA_PATH=/home/kouyang/lib/armadillo
OPENBLAS_PATH=/home/kouyang/lib/openblas
all:kmeans

kmeans:main.cpp Manager.cpp Manager.h makefile
	$(CC) -std=gnu++11 -Wall -fpie -pie -rdynamic -O3 -g3 -o kmeans main.cpp Manager.cpp -L$(OPENBLAS_PATH)/lib -Wl,-rpath=$(OPENBLAS_PATH)/lib -lopenblas -pthread -I$(ARMA_PATH) -I./
#%.o:%.cp
#	$(CC) -Wall -I/home/liberty/lib/armadillo-6.700.4/include/ -O3 -c $*.cpp

.PHONY:clean

clean:
	@rm kmeans
	
