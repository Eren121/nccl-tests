#!/bin/bash

wd="/app/nccl-tests/build"

mpirun --pernode --host nccl-hpe,nccl-smartedge --wdir "$wd" ./all_gather_perf"