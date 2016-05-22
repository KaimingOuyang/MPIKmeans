#ifndef MANAGER_H
#define MANAGER_H

#include <armadillo>
using namespace arma;
class Manager {
private:
    mat datasetT;
    mat dataset;
    mat centroids;
    mat centroidsOther;
    mat variance;
    Mat<long> counts;
    mat ddt;
    uword* assignments;
    int rank,size;
    long head;
    long rowNum,nodes,clusters,features;
    //typedef mat* MatP;
    char* outcsv;

    void scatterNodeToProcess(char* strFile);
    void broadcastCentroidsToProcess(char* strFile);
    int adjustEmptyCluster(int index, mat& oldCentroids, mat& newCentroids);
    double powDistEvaluate(const mat& a, const mat& b);
    double iterationKmeans(mat& oldCentroids, mat& newCentroids);
    void loadData(double* buffer, int row, FILE* file);
public:
    Manager(int rank, int size, int argc, char* argv[]);
    ~Manager();
    long iteration;
    double iterate();
    void outputResult();
};

#endif // MANAGER_H
