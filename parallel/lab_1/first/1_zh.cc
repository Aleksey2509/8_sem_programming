#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <mpi.h>
#include "matrix.hh"

std::pair<int, int> get_start_ind(int rank, int world_size, int row_len)
{
    int cur_row_worker_amount = int(world_size / 4);
    int cur_worker_row = rank % 4;
    if ((cur_worker_row) < (world_size - cur_row_worker_amount * 4))
        cur_row_worker_amount++;
    
    int cur_row_rank = rank / 4;
    int delta = row_len / cur_row_worker_amount;
    int begin = row_len * cur_row_rank / cur_row_worker_amount;
    int end = begin + delta;
    if (end > row_len)
        end = row_len;

    return {begin, end};
}


int main(int argc, char **argv)
{    
    MPI_Init(&argc, &argv);
    int i_size = 1000;
    int j_size = 1000;

    if (argc == 3)
    {
        std::cerr << "give i AND j" << std::endl; 
        return -1;
    }
    if (argc >= 4)
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
    int starting_row = mpi_rank % 4;
    // printf("hey i am %d out of %d, starting\n", mpi_rank, mpi_size);
    for (int i = 4 + starting_row; i < i_size; i += 4)
    {
        for (int j = 0; j < j_size - 2; j++)
        {
            a[i * j_size + j] = sin(6 * a[(i - 4) * j_size + j + 2]);
        }
    }

    #endif
    if (mpi_rank == 0)
        printf("time: %lf\n", MPI_Wtime() - time);
    
    #ifdef PAR
    for (int i = 4; i < i_size; i++)
    {
        int copy_amount = 0;
        if ((i - starting_row) % 4 == 0)
            copy_amount = j_size;
        MPI_Gather(a + i * j_size, copy_amount, MPI_DOUBLE, a + i * j_size, j_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    #endif

    // a_mat.print();
    FILE* dump_file;
    if ((argc > 1) && (mpi_rank == 0))
    {
        dump_file = fopen(argv[1], "w");
        fwrite(a, sizeof(double), i_size * j_size, dump_file);
        fclose(dump_file);
    }

    MPI_Finalize();
}
