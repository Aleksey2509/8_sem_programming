__kernel void matrix_multiply(__global int *A, __global int *B, __global int *C, int AX, int AY, int BY)
{
  int k, t;
  const int row = get_local_id(0);                  // Local row ID (max: TILE_SIZE)
  const int col = get_local_id(1);                  // Local col ID (max: TILE_SIZE)
  const int globalRow = TILE_SIZE * get_group_id(0) + row; // Row ID of C (0..M)
  const int globalCol = TILE_SIZE * get_group_id(1) + col; // Col ID of C (0..N)

  __local int Asub[TILE_SIZE][TILE_SIZE];
  __local int Bsub[TILE_SIZE][TILE_SIZE];

  int acc = 0;

  const int numTiles = AY / TILE_SIZE;

  for (t = 0; t < numTiles; t++)
  {
    const int tiledRow = TILE_SIZE * t + row;
    const int tiledCol = TILE_SIZE * t + col;
    Asub[col][row] = A[globalRow * AY + tiledCol];
    Bsub[col][row] = B[tiledRow * BY + globalCol];

    // Synchronise to make sure the tile is loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    for (k = 0; k < TILE_SIZE; k++)
      acc += Asub[k][row] * Bsub[col][k];

    // Synchronise before loading the next tile
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Store the final result in C
  C[globalRow * BY + globalCol] = acc;
}
