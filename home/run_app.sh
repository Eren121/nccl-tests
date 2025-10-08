#!/bin/bash

wd="/app/nccl-tests/build"

mpirun -x NCCL_DEBUG=INFO --pernode --host smartedge,hpe \
    /app/nccl-tests/build/all_gather_perf -n 100000
