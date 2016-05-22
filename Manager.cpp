#include <Manager.h>
#include <stdlib.h>
#include <armadillo>
#include <mpi.h>
#include <float.h>
using namespace arma;
// define DBL_MAX
// manager initializes
// argv: filename0 nodes1 features2 clusters3 nodes.in4 centroids.in5 output.out6
/*********************************************************************************/
Manager::Manager(int rank, int size, int argc, char* argv[]) {
    this->rank = rank;
    this->size = size;
    this->nodes = atoi(argv[1]);
    this->features = atoi(argv[2]);
    this->clusters = atoi(argv[3]);
    this->rowNum = (rank + 1) * nodes / size - rank * nodes / size;
    this->outcsv = argv[6];
    this->head = rank * nodes / size;
    iteration = 0;
    variance = mat(clusters, 1);
    counts = Mat<long>(clusters, 1);
    assignments = new uword[rowNum];
    scatterNodeToProcess(argv[4]);
    broadcastCentroidsToProcess(argv[5]);
    //printf("Finish load.\n");
}

void Manager::loadData(double* buffer, int row, FILE* file) {
    for(int index2 = 0; index2 < row; index2++)
        for(int index3 = 0; index3 < features; index3++)
            fscanf(file, "%lf", &buffer[index2 * features + index3]);
    return;
}

void Manager::scatterNodeToProcess(char* strFile) {
    double* tmpDataset = new double[(rowNum + 1) * features]; // choose best process to broadcast
    if(rank == size - 1) {
        FILE* nodeFile = fopen(strFile, "r");
        //MPI_Request rq[size];
        //MPI_Status st[size];
        long tmpRowNum;
        for(int index1 = 0; index1 < size; index1++) {
            tmpRowNum = (index1 + 1) * nodes / size - index1 * nodes / size;

            // read data from nodes.in file
            loadData(tmpDataset, tmpRowNum, nodeFile);

            // send data to other processes
            if(index1 != size - 1)
                MPI_Send(tmpDataset, tmpRowNum * features, MPI_DOUBLE, index1, 0, MPI_COMM_WORLD);
        }
        fclose(nodeFile);
        //MPI_Waitall(size - 1, rq, st);
    } else {
        MPI_Status st;
        MPI_Recv(tmpDataset, rowNum * features, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, &st);
    }
    datasetT = mat(tmpDataset, features, rowNum);
    dataset = datasetT.t();
    /*
    if(rank == 0) {
        int flag = 1;
        dataset.print("Rank:0\n");
        MPI_Send(&flag, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else {
        int flag;
        MPI_Status st;
        MPI_Recv(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
        dataset.print("Rank:1\n");
    }
    */
    //while(1);
    //dataset.print("here");
    // get dataset col quadratic sum
    double sum;
    ddt = mat(rowNum, 1);
    for(int index1 = 0; index1 < rowNum; index1++) {
        sum = 0;
        for(int index2 = 0; index2 < features; index2++)
            sum += datasetT(index2, index1) * datasetT(index2, index1);
        ddt(index1, 0) = sum;
    }
    return;
}

void Manager::broadcastCentroidsToProcess(char* strFile) {

    double* tmpCent = new double[clusters * features];
    if(rank == 0) {
        FILE* centFile = fopen(strFile, "r");
        for(int index1 = 0; index1 < clusters; index1++)
            for(int index2 = 0; index2 < features; index2++) {
                fscanf(centFile, "%lf", &tmpCent[index1 * features + index2]);
            }
        fclose(centFile);
    }
    MPI_Bcast(tmpCent, clusters * features, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    centroids = mat(tmpCent, features, clusters);
    centroidsOther = mat(features, clusters);
    /*
    if(rank == 0) {
        int flag = 1;
        centroids.print("Rank:0\n");
        MPI_Send(&flag, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else {
        int flag;
        MPI_Status st;
        MPI_Recv(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
        centroids.print("Rank:1\n");
    }
    */
    return;
}
/*********************************************************************************/

double Manager::iterate() {
    double cNorm;

    if(iteration % 2 == 0)
        cNorm = iterationKmeans(centroids, centroidsOther);
    else
        cNorm = iterationKmeans(centroidsOther, centroids);

    for(int index1 = 0; index1 < clusters; index1++) {
        if(counts[index1] == 0) {
            if(iteration % 2 == 0)
                adjustEmptyCluster(index1, centroids, centroidsOther);
            else
                adjustEmptyCluster(index1, centroidsOther, centroids);
        }
    }

    iteration++;
    return cNorm;
}

double Manager::iterationKmeans(mat& oldCentroids, mat& newCentroids) {
    double sum;
    // get centroids col quadratic sum
    mat cct(clusters, 1);
    for(int index1 = 0; index1 < clusters; index1++) {
        sum = 0;
        for(int index2 = 0; index2 < features; index2++)
            sum += oldCentroids(index2, index1) * oldCentroids(index2, index1);
        cct(index1, 0) = sum;
    }

    // get every distance between all nodes and centroids
    mat dist = dataset * oldCentroids;
    dist *= -2;
    dist.each_col() += ddt;
    mat distT = dist.t();
    distT.each_col() += cct;

    // set counts and variance to zero
    counts.zeros();
    variance.zeros();

    // set zero to each of elements in newCentroids
    newCentroids.zeros();

    // assign each of nodes to its nearest centroids
    uword minc; // closest cluster
    for(int index1 = 0; index1 < rowNum; index1++) {
        distT.col(index1).min(minc);
        assignments[index1] = minc;
        counts[minc] += 1;
        newCentroids.col(minc) += datasetT.col(index1);
        variance[minc] += powDistEvaluate(datasetT.col(index1), oldCentroids.col(minc));
    }

    if(size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, counts.memptr(), clusters, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, newCentroids.memptr(), clusters * features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, variance.memptr(), clusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    // This can be optimized (may be deleted)
    // get new centroids and deal with empty centroids
    for(int index1 = 0; index1 < clusters; ++index1)
        if(counts[index1] != 0)
            newCentroids.col(index1) /= counts(index1);
        else
            newCentroids.col(index1).fill(DBL_MAX); // Invalid value. Maybe changed

    //printf("Rank:%d\n",rank);
    //counts
    // Calculate cluster distortion for this iteration.
    double cNorm = 0.0;
    for(int index1 = 0; index1 < clusters; ++index1)
        cNorm += powDistEvaluate(oldCentroids.col(index1), newCentroids.col(index1)); // problem: why not
    // calculate it after get newCentroids
    // get variance
    for(int index1 = 0; index1 < clusters; ++index1)
        if(counts[index1] <= 1)
            variance[index1] = 0;
        else
            variance[index1] /= counts[index1];

    return std::sqrt(cNorm);
}

int Manager::adjustEmptyCluster(int emptyCluster, mat& oldCentroids, mat& newCentroids) {
    // find the cluster with maximum variance.
    uword maxVarCluster;
    variance.max(maxVarCluster);
    printf("Enter\n");
    // If the cluster with maximum variance has variance of 0, then we can't
    // continue.  All the points are the same.
    if(variance[maxVarCluster] == 0.0)
        return 0;

    // Now, inside this cluster, find the point which is furthest away.
    int furthestPoint = nodes;
    double maxDistance = -DBL_MAX;
    for(int index1 = 0; index1 < rowNum; ++index1) {
        if(assignments[index1] == maxVarCluster) {
            const double distance = powDistEvaluate(datasetT.col(index1), newCentroids.col(maxVarCluster)); // newCentroids?

            if(distance > maxDistance) {
                maxDistance = distance;
                furthestPoint = index1;
            }
        }
    }

    // need to be modified maxDistance should reduce!!

    // Take that point and add it to the empty cluster.
    newCentroids.col(maxVarCluster) *= (double(counts[maxVarCluster])
                                        / double(counts[maxVarCluster] - 1));
    newCentroids.col(maxVarCluster) -= (1.0
                                        / (counts[maxVarCluster] - 1.0))
                                       * arma::vec(datasetT.col(furthestPoint));
    counts[maxVarCluster]--;
    counts[emptyCluster]++;
    newCentroids.col(emptyCluster) = datasetT.col(furthestPoint);
    assignments[furthestPoint] = emptyCluster;

    return 1;
}



double Manager::powDistEvaluate(const mat& a, const mat& b) {
    double sum = 0;
    for(int index1 = 0; index1 < features; index1++)
        sum += (a(index1, 0) - b(index1, 0)) * (a(index1, 0) - b(index1, 0));
    return sum;
}

void Manager::outputResult() {
    uword* finalAssign;
    int* displs;
    int* recvcounts;
    if(rank == 0) {
        FILE* fp = fopen(outcsv, "w");
        displs = new int[size];
        recvcounts = new int[size];
        for(int index1 = 0; index1 < size; index1++) {
            displs[index1] = index1 * nodes / size;
            recvcounts[index1] = (index1 + 1) * nodes / size - displs[index1];
        }
        finalAssign = new uword[nodes]; // memory may not hold whole dataset, should be changed in future
        MPI_Gatherv(assignments, rowNum, MPI_INT, finalAssign, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
        for(int index = 0; index < nodes; index++)
            fprintf(fp, "%u\n", finalAssign[index]);
        delete[] finalAssign;
        delete[] displs;
        delete[] recvcounts;
        fclose(fp);
    } else
        MPI_Gatherv(assignments, rowNum, MPI_INT, finalAssign, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    return;
}

Manager::~Manager() {
    delete[] assignments;
}
