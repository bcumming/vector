for threads in 1 2 3 4 5 6 7 8 9 10 11 12
do
    OMP_NUM_THREADS=$threads numactl --physcpubind=0-$[$threads -1] ./stream.exe
done
