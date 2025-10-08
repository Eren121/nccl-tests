#!/bin/bash

wd="/app/nccl-tests/build"

mpirun -x NCCL_DEBUG=INFO --pernode --host smartedge,hpe \
    nsys profile --trace=cuda,nvtx,mpi --stats=true -o nccl_profile \
    /app/nccl-tests/build/all_gather_perf -n 100000

