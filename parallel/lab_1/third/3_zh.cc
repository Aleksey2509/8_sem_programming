#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>
#include <unistd.h>

#include "matrix.hh"
#include <mpi.h>

int rows_amount_for_rank(int rank, int world_size, int i_size)
{
    int rows_calc_amount = i_size / world_size;
    if (i_size % world_size > rank)
        rows_calc_amount++;

    return rows_calc_amount;
}

 

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int i_size = 5000;
    int j_size = 5000;

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

    Matrix<double> a(i_size, j_size);
    Matrix<double> b(i_size, j_size);
    
    for (int i = 0; i < i_size; i++)
        for (int j = 0; j < j_size; j++)
        {
            a(i, j) = 10 * i + j;
            b(i, j) = 0;
        }
    
    int mpi_size{0};
    int mpi_rank{0};
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    double time = MPI_Wtime();
    double* b_arr = b.data();
    double* a_arr = a.data();
    #ifndef PAR
    for (int i = 0; i < i_size; i++)
       for (int j = 0; j < j_size; j++)
       {
           a_arr[i * j_size + j] = sin(0.1 * a_arr[i * j_size + j]);
       }
    for (int i = 0; i < i_size - 1; i++)
       for (int j = 2; j < j_size; j++)
       {
           b_arr[i * j_size + j] = a_arr[(i + 1) * j_size + j - 2] * 1.5;
       }

    #else
    std::vector<MPI_Request> send_arr(i_size, MPI_REQUEST_NULL);
    std::vector<MPI_Request> recv_arr(i_size, MPI_REQUEST_NULL);

    for (int i = mpi_rank; i < i_size; i += mpi_size)
    {
       if ((i != i_size - 1))
       {
            if (mpi_rank != mpi_size - 1)
                MPI_Irecv(a.data() + (i + 1) * j_size, j_size, MPI_DOUBLE, mpi_rank + 1, 0, MPI_COMM_WORLD, recv_arr.data() + i);
            else
                MPI_Irecv(a.data() + (i + 1) * j_size, j_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, recv_arr.data() + i);
       }
       for (int j = 0; j < j_size; j++)
       {
           // printf("rank %d at %d %d, a was %lf\n", mpi_rank, i, j, a(i, j));
           a_arr[i * j_size + j] = sin(0.1 * a_arr[i * j_size + j]);
           // printf("rank %d at %d %d, a now %lf\n", mpi_rank, i, j, a(i, j));
       }

       if (i != 0)
       {
           if (mpi_rank != 0)
           {
               double* a_line = a.data() + i * j_size;
               // printf("rank %d sending to rank %d %lf %lf %lf %lf\n", mpi_rank, mpi_rank - 1, 
               //        (a_line)[0], (a_line)[1], (a_line)[2], (a_line)[3] );
               MPI_Isend(a.data() + i * j_size, j_size, MPI_DOUBLE, mpi_rank - 1, 0, MPI_COMM_WORLD, send_arr.data() + i);
           }
           else
           {
               double* a_line = a.data() + i * j_size;
               // printf("rank %d sending to rank %d %lf %lf %lf %lf\n", mpi_rank, mpi_rank - 1, 
               //        (a_line)[0], (a_line)[1], (a_line)[2], (a_line)[3] );
               MPI_Isend(a.data() + i * j_size, j_size, MPI_DOUBLE, mpi_size - 1, 0, MPI_COMM_WORLD, send_arr.data() + i);
           }
       }
       
    }

    for (int i = mpi_rank; i < i_size; i += mpi_size)
    {
        if (i != 0)
            MPI_Wait(send_arr.data() + i, MPI_STATUS_IGNORE);
        if (i != i_size - 1)
            MPI_Wait(recv_arr.data() + i, MPI_STATUS_IGNORE);
    }

    // if (mpi_rank == 0)
    // {
    //     std::cout << "a as i: " << mpi_rank << " have it" << std::endl;
    //     a.print();
    // }


    for (int i = mpi_rank; i < i_size - 1; i += mpi_size)
       for (int j = 2; j < j_size; j++)
       {
           b_arr[i * j_size + j] = a_arr[(i + 1) * j_size + j - 2] * 1.5;
       }

    #endif
    if (mpi_rank == 0)
        fprintf(stderr, "time: %lf\n", MPI_Wtime() - time);
    
    #ifdef PAR
    for (int i = 0; (i < i_size) && (mpi_rank != 0); i++)
    {
        int rank_to_send = i % mpi_size;
        if (rank_to_send != mpi_rank)
            continue;
        MPI_Send(b_arr + i * j_size, j_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
 
    for (int i = 0; (i < i_size) && (mpi_rank == 0); i++)
    {
        // printf("receving %d\n", i);
        int rank_to_recv_from = i % mpi_size;
        if (rank_to_recv_from == 0)
            continue;
        MPI_Recv(b_arr + i * j_size, j_size, MPI_DOUBLE, rank_to_recv_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // printf("receved %d\n", i);
    }
    #endif
    
    FILE* dump_file;
    if ((argc > 1) && (mpi_rank == 0))
    {
        dump_file = fopen(argv[1], "w");
        fwrite(b_arr, sizeof(double), i_size * j_size, dump_file);
        fclose(dump_file);
    }

    MPI_Finalize();
}
