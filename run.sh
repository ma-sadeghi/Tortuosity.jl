#!/bin/bash

gpu_id=0
path_matrix="/mnt/optane/Amin/matrixlib/sample_B/chunked"
path_export="/home/amin/Code/Tortuosity.jl/results/sample_B/chunked"

for fpath in $path_matrix/*
do
    fname=$(basename $fpath)
    julia --project=@. src/main.jl --fpath="$fpath" --path_export="$path_export" --gpu_id=$gpu_id
done

# Uncomment the next 4 lines for a test run
# julia --project=@. src/main.jl \
#     --fpath="/mnt/optane/Amin/matrixlib/sample_B/chunked/sample_B6_v0_eps=0.791_large_s0.h5" \
#     --path_export="/home/amin/Code/Tortuosity.jl/results/sample_B/chunked" \
#     --gpu_id=$gpu_id
