#include <Manager.h>
#include <stdlib.h>
#include <armadillo>
#include <mpi.h>
#include <float.h>
#include <fstream>
#include <iomanip>
#include <time.h>
using namespace arma;
using namespace std;
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
    this->max_iter = atoi(argv[4]);
    this->rowNum = (rank + 1) * nodes / size - rank * nodes / size;
    //printf("rank:%d,rowNum:%d\n",rank,clusters);

    // this->outcsv = argv[6];
    this->head = rank * nodes / size;
    iteration = 0;
    // variance = Col<double>(clusters);
    // counts = Col<long>(clusters);
    assignments = new uword[rowNum];
    // scatterNodeToProcess(argv[4]);
    // broadcastCentroidsToProcess(argv[5]);
    //printf("Finish load.\n");
}

void Manager::loadData(double* buffer, int row, FILE* file) {
    for (int index2 = 0; index2 < row; index2++)
        for (int index3 = 0; index3 < features; index3++)
            if (index3 != features - 1)
                fscanf(file, "%lf,", &buffer[index2 * features + index3]);
            else
                fscanf(file, "%lf", &buffer[index2 * features + index3]);
    return;
}

// void Manager::scatterNodeToProcess(char* strFile) {
//     double* tmpDataset = new double[(rowNum + 1) * features]; // choose best process to broadcast
//     if(rank == size - 1) {
//         FILE* nodeFile = fopen(strFile, "r");
//         //MPI_Request rq[size];
//         //MPI_Status st[size];
//         long tmpRowNum;
//         for(int index1 = 0; index1 < size; index1++) {
//             tmpRowNum = (index1 + 1) * nodes / size - index1 * nodes / size;

//             // read data from nodes.in file
//             loadData(tmpDataset, tmpRowNum, nodeFile);

//             // send data to other processes
//             if(index1 != size - 1)
//                 MPI_Send(tmpDataset, tmpRowNum * features, MPI_DOUBLE, index1, 0, MPI_COMM_WORLD);
//         }
//         fclose(nodeFile);
//         //MPI_Waitall(size - 1, rq, st);
//     } else {
//         MPI_Status st;
//         MPI_Recv(tmpDataset, rowNum * features, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, &st);
//     }

//     dataset = mat(tmpDataset, rowNum, features);
//     datasetT = dataset.t();
//     /*
//     if(rank == 0) {
//         int flag = 1;
//         dataset.print("Rank:0\n");
//         fflush(stdout);
//         MPI_Send(&flag, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
//     } else {
//         int flag;
//         MPI_Status st;
//         MPI_Recv(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
//         dataset.print("Rank:1\n");
//         fflush(stdout);
//     }

//     while(1);
//     */
//     //dataset.print("here");
//     // get dataset col quadratic sum
//     /*
//     double sum;
//     ddt = mat(rowNum, 1);
//     for(int index1 = 0; index1 < rowNum; index1++) {
//         sum = 0;
//         for(int index2 = 0; index2 < features; index2++)
//             sum += datasetT(index2, index1) * datasetT(index2, index1);
//         ddt(index1, 0) = sum;
//     }
//     */
//     return;
// }

// void Manager::broadcastCentroidsToProcess(char* strFile) {
//     mat *tmp_matrix;
//     double* tmpCent = new double[clusters * features];
//     if(rank == 0) {
//         FILE* centFile = fopen(strFile, "r");
//         loadData(tmpCent, clusters, centFile);
//         /*
//         for(int index1 = 0; index1 < clusters; index1++)
//             for(int index2 = 0; index2 < features; index2++) {
//                 if(index2 != features - 1)
//                     fscanf(centFile, "%lf,", &tmpCent[index1 * features + index2]);
//                 else
//                     fscanf(centFile, "%lf", &tmpCent[index1 * features + index2]);
//             }
//         */
//         fclose(centFile);
//     }
//     MPI_Bcast(tmpCent, clusters * features, MPI_DOUBLE, 0, MPI_COMM_WORLD);

//     delete tmpCent;
//     //if(rank == 0)
//     //centroids.print();
//     /*
//     if(rank == 0) {
//         int flag = 1;
//         centroids.print("Rank:0\n");
//         fflush(stdout);
//         MPI_Send(&flag, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
//     } else {
//         int flag;
//         MPI_Status st;
//         MPI_Recv(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
//         centroids.print("Rank:1\n");
//         fflush(stdout);
//     }
//     while(1);
//     */
//     return;
// }
/*********************************************************************************/

void Manager::compute() {
    double diff;
    arma_rng::set_seed(0);
    dataset = mat(rowNum, features, fill::randu);
    centroids = mat(features, clusters, fill::randn);
    centroidsOther = mat(features, clusters, fill::zeros);
    counts = Col<long>(clusters, fill::zeros);
    variance = Col<double>(clusters, fill::zeros);

    comm_time = 0.0;
    compute_time = 0.0;
    double st = MPI_Wtime();
    do {
        diff = iterate();
        //fprintf(fp,"Iteration:%d,diff:%lf\n",manager.iteration,diff);
        //printf("diff:%lf,Iteration:%ld,rank:%d\n",manager.iteration,diff,rank);
        //fflush(stdout);
    } while (diff > 1e-5 && iteration < max_iter);
    //fclose(fp);
    st = MPI_Wtime() - st;
    double max_time, sum_comm_time, sum_compute_time;
    MPI_Reduce(&st, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &sum_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &sum_compute_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("%d %lf %lf %lf %lf\n", size / 18, max_time / iteration * 1e3, sum_comm_time / size / iteration * 1e3, sum_compute_time  / size / iteration * 1e3, (sum_comm_time + sum_compute_time) / iteration * 1e3);
}

double Manager::iterate() {
    double cNorm;

    if (iteration % 2 == 0)
        cNorm = iterationKmeans(centroids, centroidsOther);
    else
        cNorm = iterationKmeans(centroidsOther, centroids);

    if (this->emptyFlag) {
        for (int index1 = 0; index1 < clusters; index1++) {
            if (counts(index1) == 0) {
                if (iteration % 2 == 0)
                    adjustEmptyCluster(index1, centroids, centroidsOther);
                else
                    adjustEmptyCluster(index1, centroidsOther, centroids);
            }
        }
    }

    iteration++;
    return cNorm;
}

double Manager::iterationKmeans(mat& oldCentroids, mat& newCentroids) {
    double sum;
    // get centroids col quadratic sum
    Col<double> cct(clusters);
    for (int index1 = 0; index1 < clusters; index1++) {
        sum = 0;
        for (int index2 = 0; index2 < features; index2++)
            sum += oldCentroids(index2, index1) * oldCentroids(index2, index1);
        cct(index1) = sum;
    }

    // get every distance between all nodes and centroids
    struct timespec st, ed;
    //clock_gettime(CLOCK_REALTIME,&st);
    compute_time -= MPI_Wtime();
    mat dist = dataset * oldCentroids;
    //clock_gettime(CLOCK_REALTIME,&ed);
    //double tol = (double)ed.tv_sec - st.tv_sec + (double)(ed.tv_nsec - st.tv_nsec) / 1000000000;
    //printf("Gflops:%lfG/s\n", 2.0 * features * nodes * clusters / tol / 1000000000.0);
    dist *= -2;
    compute_time += MPI_Wtime();
    //dist.each_col() += ddt;
    mat distT = dist.t();
    distT.each_col() += cct;
    //distT.print();
    // set counts and variance to zero
    counts.zeros(clusters);
    variance.zeros(clusters);
    //assignments.set_size(rowNum);
    // set zero to each of elements in newCentroids
    newCentroids.zeros(features, clusters);
    //oldCentroids.print();
    //while(1);
    // assign each of nodes to its nearest centroids
    uword minc; // closest cluster
    uword* mincArray = new uword[rowNum];
    for (int index1 = 0; index1 < rowNum; index1++) {
        distT.col(index1).min(minc);
        assignments[index1] = minc;
        counts(minc) += 1;
        newCentroids.col(minc) += dataset.row(index1).t();
        mincArray[index1] = minc;
    }
    //ofstream out("MPIassi",ios::app);
    //for(int i=0;i<rowNum;i++)
    //    out << assignments[i] << " ";
    //out << endl;
    //printf("Here\n");
    comm_time -= MPI_Wtime();
    if (size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, counts.memptr(), clusters, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, newCentroids.memptr(), clusters * features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    comm_time += MPI_Wtime();
    /*
    if(rank == 0) {
        int flag = 1;
        variance.print("Rank:0\n");
        fflush(stdout);
        MPI_Send(&flag, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else {
        int flag;
        MPI_Status st;
        MPI_Recv(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
        variance.print("Rank:1\n");
        fflush(stdout);
    }
    while(1);
        */
        // This can be optimized (may be deleted)
        // get new centroids and deal with empty centroids
        // get variance
    this->emptyFlag = 0;
    for (int index1 = 0; index1 < clusters; ++index1) {
        if (counts(index1) != 0)
            newCentroids.col(index1) /= counts(index1);
        else {
            this->emptyFlag = 1;
            newCentroids.col(index1).fill(DBL_MAX); // Invalid value. Maybe changed
        }
    }

    if (this->emptyFlag) {
        for (int index1 = 0;index1 < rowNum;index1++)
            variance(mincArray[index1]) += powDistEvaluate(dataset.row(index1).t(), oldCentroids.col(mincArray[index1]));
        if (size > 1)
            MPI_Allreduce(MPI_IN_PLACE, variance.memptr(), clusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (int index1 = 0; index1 < clusters; ++index1) {
            if (counts(index1) <= 1)
                variance(index1) = 0;
            else
                variance(index1) /= counts(index1);
        }
    }
    delete[] mincArray;


    // Calculate cluster distortion for this iteration.
    double cNorm = 0.0;
    for (int index1 = 0; index1 < clusters; ++index1)
        cNorm += powDistEvaluate(oldCentroids.col(index1), newCentroids.col(index1)); // problem: why not
    return std::sqrt(cNorm);
}

int Manager::adjustEmptyCluster(int emptyCluster, mat& oldCentroids, mat& newCentroids) {
    // find the cluster with maximum variance.
    uword maxVarCluster;
    variance.max(maxVarCluster);
    //printf("Enter\n");
    // If the cluster with maximum variance has variance of 0, then we can't
    // continue.  All the points are the same.
    if (variance[maxVarCluster] == 0.0)
        return 0;
    /*
    if(rank == 0) {
        int flag = 1;
        printf("Rank:0,max:%d,iteration:%d\n",maxVarCluster,iteration);
        variance.print();
        fflush(stdout);
        MPI_Send(&flag, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else {
        int flag;
        MPI_Status st;
        MPI_Recv(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
        printf("Rank:1,max:%d,iteration:%d\n",maxVarCluster,iteration);
        variance.print();
        fflush(stdout);
    }
    */
    //while(1);

    //static int flag = 0;
    //if(flag != 2){
    //ofstream iof("MPI.va",ios::app);
    //variance.print(iof,"MPI");

    //printf("rank:%d,MPI:%d\n",rank,maxVarCluster);
    //fflush(stdout);
    // Now, inside this cluster, find the point which is furthest away.
    int furthestPoint = nodes;
    double maxDistance = -1;
    //ofstream out2("MPIdis",ios::app);

    for (int index1 = 0; index1 < rowNum; ++index1) {
        if (assignments[index1] == maxVarCluster) {
            const double distance = powDistEvaluate(dataset.row(index1).t(),
                newCentroids.col(maxVarCluster)); // newCentroids?
//out2 << setprecision(20) << distance << " " << index1 << " " << maxVarCluster << endl;
            if (distance - maxDistance > 1e-4) {
                maxDistance = distance;
                furthestPoint = index1;
                //out2 << "MAX:" << maxDistance << " " << furthestPoint << endl;
            }
        }
    }
    /*
    if(rank == 0) {
        int flag = 1;
        printf("Rank:0,max:%lf,index:%d,MPIMAX:%d\n",maxDistance,furthestPoint+head,maxVarCluster);
        variance.print();
        fflush(stdout);
        MPI_Send(&flag, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else {
        int flag;
        MPI_Status st;
        MPI_Recv(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
        printf("Rank:1,max:%lf,index:%d,MPIMAX:%d\n",maxDistance,furthestPoint+head,maxVarCluster);
        variance.print();
        fflush(stdout);
    }
    */
    Col<double> vecdata(features);
    //out2 << "End" << endl;
    double tmpMaxDist;
    int tmpRank = (1 << 31) - 1;

    MPI_Allreduce(&maxDistance, &tmpMaxDist, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (abs(maxDistance - tmpMaxDist) < 1e-5) {
        MPI_Allreduce(&rank, &tmpRank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    } else {
        MPI_Allreduce(MPI_IN_PLACE, &tmpRank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }
    //MPI_Allreduce(&local,&global,1,MPI_DOUBLE_INT,MPI_MAXLOC,MPI_COMM_WORLD);
    if (rank == tmpRank) {
        assignments[furthestPoint] = emptyCluster;
        vecdata = dataset.row(furthestPoint).t();
        //printf("id:%d,empty:%d\n",furthestPoint+head,emptyCluster);
        MPI_Bcast(vecdata.memptr(), features, MPI_DOUBLE, tmpRank, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(vecdata.memptr(), features, MPI_DOUBLE, tmpRank, MPI_COMM_WORLD);
    }

    // Take that point and add it to the empty cluster.
    newCentroids.col(maxVarCluster) *= (double(counts[maxVarCluster])
        / double(counts[maxVarCluster] - 1));
    newCentroids.col(maxVarCluster) -= (1.0
        / (counts[maxVarCluster] - 1.0))
        * vec(vecdata);
    counts[maxVarCluster]--;
    counts[emptyCluster]++;
    newCentroids.col(emptyCluster) = vec(vecdata);

    variance[emptyCluster] = 0;
    if (counts[maxVarCluster] <= 1)
        variance[maxVarCluster] = 0;
    else {
        variance[maxVarCluster] = (1.0 / counts[maxVarCluster])
            * ((counts[maxVarCluster] + 1) * variance[maxVarCluster]
                - tmpMaxDist);
    }
    return 1;
}

double Manager::powDistEvaluate(const colvec& a, const colvec& b) {
    double sum = 0;
    for (int index1 = 0; index1 < features; index1++)
        sum += (a(index1) - b(index1)) * (a(index1) - b(index1));
    return sum;
}

void Manager::outputResult(double time) {

    if (rank == 0) {
        uword* finalAssign;
        int* displs;
        int* recvcounts;
        FILE* ftime = fopen("MPItime", "a");
        fprintf(ftime, "%d,%d,%d,%lfs\n", nodes, clusters, features, time);
        fclose(ftime);
        FILE* fp = fopen(outcsv, "w");
        displs = new int[size];
        recvcounts = new int[size];
        for (int index1 = 0; index1 < size; index1++) {
            displs[index1] = index1 * nodes / size;
            recvcounts[index1] = (index1 + 1) * nodes / size - displs[index1];
        }
        //for(int i=0;i<rowNum;i++)
        //    printf("%u ",assignments[i]);
        finalAssign = new uword[nodes]; // memory may not hold whole dataset, should be changed in future
        MPI_Gatherv(assignments, rowNum, MPI_INT, finalAssign, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
        for (int index = 0; index < nodes; index++)
            fprintf(fp, "%u\n", finalAssign[index]);
        delete[] finalAssign;
        delete[] displs;
        delete[] recvcounts;
        fclose(fp);
    } else
        MPI_Gatherv(assignments, rowNum, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);
    return;
}

Manager::~Manager() {
    delete[] assignments;
    // delete dataset;
    // delete centroids;
    // delete centroidsOther;
}
