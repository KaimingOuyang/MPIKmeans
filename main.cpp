#include <iostream>
#include <mpi.h>
#include <armadillo>
#include <Manager.h>

using namespace std;

// argv: filename0 nodes1 features2 clusters3 nodes.in4 centroids.in5 output.out6
// Now nodes.in and centroids.in is row-major ordering
int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(argc != 7) {  // Until now, it is 7 parameters
        cout << "Parameter Error. Input format: ./applicationName nodes features clusters"
             " nodes.in centroids.in output.out" << endl;
        exit(1);
    }

    Manager manager(rank, size, argc, argv);
    //FILE* fp = fopen("diffout","w");
    double diff;
    do {
        diff = manager.iterate();
        //fprintf(fp,"Iteration:%d,diff:%lf\n",manager.iteration,diff);
        //printf("diff:%lf,Iteration:%ld,rank:%d\n",manager.iteration,diff,rank);
        //fflush(stdout);
    } while(diff > 1e-5 && manager.iteration < 1000);
    //fclose(fp);
    manager.outputResult();
    MPI_Finalize();
    return 0;
}

