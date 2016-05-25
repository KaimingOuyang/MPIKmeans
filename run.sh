nodes=10000000
clusters=10
features=200

c++ -o datage datage.c
./datage $nodes $clusters $features
mpiexec -n 1 ./kmeans $nodes $features $clusters ik.csv centroids.csv out.csv
