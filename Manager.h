#ifndef MANAGER_H
#define MANAGER_H

#include <armadillo>
using namespace arma;
class Manager {
private:

    struct{
        double v;
        int id;
    }local,global;

    mat dataset;
    mat centroids;
    mat centroidsOther;
    Col<double> variance;
    Col<long> counts;
    //mat ddt;
    uword* assignments;
    int rank,size;
    long head;
    long rowNum,nodes,clusters,features;
    //typedef mat* MatP;
    char* outcsv;
    int emptyFlag;
    void scatterNodeToProcess(char* strFile);
    void broadcastCentroidsToProcess(char* strFile);
    int adjustEmptyCluster(int index, mat& oldCentroids, mat& newCentroids);
    double powDistEvaluate(const colvec& a, const colvec& b);
    double iterationKmeans(mat& oldCentroids, mat& newCentroids);
    void loadData(double* buffer, int row, FILE* file);
public:
    Manager(int rank, int size, int argc, char* argv[]);
    ~Manager();
    long iteration;
    double iterate();
    void compute();
    void outputResult(double time);
};

#endif // MANAGER_H
