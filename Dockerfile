FROM nvcr.io/nvidia/doca/doca:3.1.0-devel-cuda12.8.0-host

# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
ARG NCCL_IB_HCA
ARG NCCL_IB_GID_INDEX

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install openmpi-bin openmpi-common libopenmpi-dev librdmacm-dev libpsm2-dev openmpi-bin libopenmpi-dev git sshpass

WORKDIR /app/nccl-tests
RUN git clone https://github.com/NVIDIA/nccl-tests .
RUN make -j MPI=1 MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1/

ENV OMPI_ALLOW_RUN_AS_ROOT=1 
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
#
# ssh-server
#

RUN apt-get install -y openssh-server vim nano
RUN echo "root:root" | chpasswd

RUN mkdir /var/run/sshd && \
    sed -i 's/#Port 22/Port 1954/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitUserEnvironment no/PermitUserEnvironment yes/' /etc/ssh/sshd_config

RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

COPY home/ /root/

# Necessary otherwise 'mpirun' is Ubuntu's and not doca's
ENV PATH="/usr/mpi/gcc/openmpi-4.1.9a1/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/mpi/gcc/openmpi-4.1.9a1/lib:$LD_LIBRARY_PATH"

RUN mkdir -p /root/.ssh


RUN echo "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}" >> /root/.ssh/environment
ENV NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}

ENV NCCL_IB_HCA=${NCCL_IB_HCA}
RUN echo "NCCL_IB_HCA=${NCCL_IB_HCA}" >> /root/.ssh/environment

RUN chmod 600 /root/.ssh/environment
RUN make -C $HOME/cpp

ENTRYPOINT $HOME/init_ssh.sh
