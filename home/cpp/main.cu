#include <stdio.h>
#include <stdint.h>
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

#define MPICHECK(cmd) do { \
    int e = cmd; \
    if (e != MPI_SUCCESS) { \
        fprintf(stderr, "MPI error %d at %s:%d\n", e, __FILE__, __LINE__); \
        MPI_Abort(MPI_COMM_WORLD, e); \
    } \
} while(0)

#define CUDACHECK(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main(int argc, char *argv[]) {
    int size = 32 * 1024 * 1024;  // buffer size
    int myRank, nRanks, localRank = 0;

    // Initialize MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    ncclUniqueId id;
    ncclComm_t comm;
    float *sendbuff, *recvbuff;
    cudaStream_t s;

    // Rank 0 generates the NCCL unique ID
    if (myRank == 0) {
        ncclGetUniqueId(&id);
    }

    // Broadcast the NCCL unique ID to all ranks
    MPICHECK(MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Optional: print to confirm all ranks received it
    printf("Rank %d received NCCL unique ID\n", myRank);

    // Determine local GPU (here, simple mapping rank -> device)
    localRank = myRank; // for single node with 2 ranks
    CUDACHECK(cudaSetDevice(0));

    // Allocate device buffers
    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));

    // Initialize NCCL communicator
    ncclCommInitRank(&comm, nRanks, id, myRank);

    // Now you can use NCCL to communicate between GPUs

    // Cleanup
    ncclCommDestroy(comm);
    cudaFree(sendbuff);
    cudaFree(recvbuff);
    cudaStreamDestroy(s);
    MPICHECK(MPI_Finalize());

    return 0;
}