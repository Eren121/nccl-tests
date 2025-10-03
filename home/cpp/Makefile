# Info:
# - gcc forwards -Wl,... to linker
# - nvcc does not supports -Xcompiler with -Wl, (it parses comma-separated arguments)
# - Instead we should use -Xlinker
# - Therefore the preprocessing.

MPI_COMPILE_FLAGS = $(shell mpicxx --showme:compile)
MPI_LINK_FLAGS    = $(shell mpicxx --showme:link | sed 's/-Wl,/ -Xlinker /g')

a.out:
	echo $(MPI_LINK_FLAGS)
	nvcc main.cu -o main \
	    -Xcompiler "$(MPI_COMPILE_FLAGS)" \
	    $(MPI_LINK_FLAGS) \
	    -lnccl

mpirun:
	 mpirun -x NCCL_DEBUG=INFO --pernode --host smartedge,hpe /root/cpp/main