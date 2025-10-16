#include <stdio.h>
#include <stdint.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <nccl.h>


#define CUDACHECK(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define NCCLCHECK(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        fprintf(stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Send ncclUniqueId over a connected socket
inline bool sendNcclId(int sock, const ncclUniqueId &id) {
    ssize_t n = send(sock, &id, sizeof(id), 0);
    return n == sizeof(id);
}

// Receive ncclUniqueId over a connected socket
inline bool recvNcclId(int sock, ncclUniqueId &id) {
    ssize_t n = recv(sock, &id, sizeof(id), 0);
    return n == sizeof(id);
}

// Start TCP server and accept one connection
inline int startServer(int port) {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) { perror("socket"); return -1; }
    int opt = 1;
    if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt"); return -1;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return -1; }
    if (listen(listen_fd, 1) < 0) { perror("listen"); return -1; }

    std::cout << "Server listening on port " << port << "...\n";
    int client_fd = accept(listen_fd, nullptr, nullptr);
    if (client_fd < 0) { perror("accept"); return -1; }

    close(listen_fd); // no longer need listening socket
    return client_fd;
}

// Connect to TCP server
inline int connectToServer(const char* ip, int port) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return -1; }
    int opt = 1;
    if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt"); return -1;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, ip, &server_addr.sin_addr);

    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect"); return -1;
    }
    return sock;
}

int main(int argc, char *argv[]) {
    int size = 32 * 1024 * 1024;  // buffer size
    int myRank, nRanks, localRank = 0;

    ncclUniqueId id;
    ncclComm_t comm;
    float *sendbuff, *recvbuff;
    cudaStream_t s;

    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    
    int sock;

    if(strcmp(hostname, "hpe") == 0) {
        myRank = 0;
        NCCLCHECK(ncclGetUniqueId(&id));

        sock = startServer(50001);
        sendNcclId(sock, id);
    }
    else {
        myRank = 1;
        sock = connectToServer("192.168.120.1", 50001);
        recvNcclId(sock, id);
    }

    close(sock);
    
    nRanks = 2;
    
    // Optional: print to confirm all ranks received it
    printf("Rank %d received NCCL unique ID\n", myRank);

    CUDACHECK(cudaSetDevice(localRank));

    // Allocate device buffers
    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * nRanks * sizeof(float))); // Allgather needs space for all ranks

    CUDACHECK(cudaMemset(sendbuff, 0, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff, 0, size * sizeof(float)));

    CUDACHECK(cudaStreamCreate(&s));

    // Initialize NCCL communicator
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    // Example: Initialize send buffer wi   th some values
    // (for demonstration, could be random or rank-specific)
    CUDACHECK(cudaMemset(sendbuff, myRank, size * sizeof(float)));

    // Perform NCCL AllGather
    NCCLCHECK(ncclAllGather(
        (const void*)sendbuff,     // send buffer
        (void*)recvbuff,           // receive buffer
        size,                      // number of elements per rank
        ncclFloat,                 // data type
        comm,                      // NCCL communicator
        s                          // CUDA stream
    ));

    // Wait for completion
    CUDACHECK(cudaStreamSynchronize(s));

    printf("Rank %d completed NCCL AllGather\n", myRank);

    // Cleanup
    NCCLCHECK(ncclCommDestroy(comm));
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));
    CUDACHECK(cudaStreamDestroy(s));

    return 0;
}
