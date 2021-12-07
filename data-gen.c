#include <stdio.h>
#include <stdlib.h>

#define MAX_VAL 1073741824
#define DOUBLE_EPS 1e10

int main(int argc, char* argv[]) {
    int nodes = atoi(argv[1]);
    int features = atoi(argv[2]);
    int clusters = atoi(argv[3]);

    char nodes_file[64];
    char centroids_file[64];
    sprintf(nodes_file, "node-%d-%d.in", nodes, features);
    sprintf(centroids_file, "centroids-%d-%d.in", clusters, features);

    FILE *nodes_fp = fopen(nodes_file, "w");
    FILE *centroids_fp = fopen(centroids_file, "w");

    for (int i = 0; i < nodes; ++i) {
        for (int j = 0; j < features; ++j){
            if (j == features - 1){
                fprintf(nodes_fp, "%lf\n", (double) (rand() % MAX_VAL) / DOUBLE_EPS + (double) (rand() % MAX_VAL));
                fprintf(centroids_fp, "%lf\n", (double) (rand() % MAX_VAL) / DOUBLE_EPS + (double) (rand() % MAX_VAL));
            } else {
                fprintf(nodes_fp, "%lf,", (double) (rand() % MAX_VAL) / DOUBLE_EPS + (double) (rand() % MAX_VAL));
                fprintf(centroids_fp, "%lf,", (double) (rand() % MAX_VAL) / DOUBLE_EPS + (double) (rand() % MAX_VAL));
            }
        }
    }

    fclose(nodes_fp);
    fclose(centroids_fp);
    return 0;
}