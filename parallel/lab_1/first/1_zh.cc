#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "matrix.hh"

int main(int argc, char **argv)
{    
    MPI_Init(&argc, &argv);
    int i_size = 1000;
    int j_size = 1000;

    if (argc == 2)
    {
        std::cerr << "give i AND j" << std::endl; 
        return -1;
    }
    if (argc >= 3)
    {
        i_size = std::atoi(argv[2]);
        j_size = std::atoi(argv[3]);
    }

    int mpi_size{0};
    int mpi_rank{0};
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    Matrix<double> a_mat(i_size, j_size);

    double* a = a_mat.data();
    FILE *ff;
    for (int i = 0; i < i_size; i++)
    {
        for (int j = 0; j < j_size; j++)
        {
            a[i * j_size + j] = 10 * i + j;
        }
    }

    double time = MPI_Wtime();
    #ifndef PAR
    for (int i = 4; i < i_size; i++)
    {
        for (int j = 0; j < j_size - 2; j++)
        {
            a[i * j_size + j] = sin(6 * a[(i - 4) * j_size + j + 2]);
        }
    }
    #else
    int size = j_size - 2;
    int range = size / mpi_size;
    int begin = mpi_rank * range;
    int end = (mpi_rank == mpi_size - 1) ? size : (mpi_rank + 1) * range;
    for (int i = 4; i < i_size; i++)
    {
        for (int j = begin; j < end; j++)
        {
            a[i * j_size + j] = sin(6 * a[(i - 4) * j_size + j + 2]);
        }
        for (int r = 0; r < mpi_size; r++)
        {
            if (mpi_rank == r)
            {
                for (int k = 0; k < mpi_size; k++)
                {
                    if (k != r)
                    {
                        int count = (k == mpi_size - 1) ? size - k * range : range;
                        MPI_Recv(&a[i * j_size + k * range], count, MPI_DOUBLE, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }
            else
            {
                MPI_Send(&a[i * j_size + mpi_rank * range], end - begin, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    #endif
    if (mpi_rank == 0) printf("%lf\n", time - MPI_Wtime());

    if ((argc > 1) && (mpi_rank == 0))
    {
        ff = fopen(argv[1], "wb");
        fwrite(a, sizeof(double), i_size * j_size, ff);
        fclose(ff);
    }

    MPI_Finalize();
}
