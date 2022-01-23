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
    if(argc < 5) {  // Until now, it is 7 parameters
        cout << "Parameter Error. Input format: ./applicationName nodes features clusters max_iteration" << endl;
        exit(1);
    }

    Manager manager(rank, size, argc, argv);
    manager.compute();
    //FILE* fp = fopen("diffout","w");
    // manager.outputResult(st);
    MPI_Finalize();
    return 0;
}
