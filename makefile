CC=mpic++
HOME=/home/liberty

all:kmeans

kmeans:main.cpp Manager.cpp Manager.h
	$(CC) -Wall -O3 -o kmeans main.cpp Manager.cpp -lopenblas
#%.o:%.cpp
#	$(CC) -Wall -I/home/liberty/lib/armadillo-6.700.4/include/ -O3 -c $*.cpp

.PHONY:clean

clean:
	@rm kmeans
	
