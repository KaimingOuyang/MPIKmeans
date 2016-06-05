nodes=200000
clusters=50
features=2000

#c++ -o datage datage.c
#./datage $nodes $clusters $features
mpiexec -n 3 ./kmeans $nodes $features $clusters ik.csv centroids.csv out.csv
