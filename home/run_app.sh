#!/bin/bash

wd="/app/nccl-tests/build"

mpirun --pernode --host nccl-hpe,nccl-smartedge "$wd/ ./all_gather_perf"