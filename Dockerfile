FROM nvcr.io/nvidia/doca/doca:3.1.0-devel-cuda12.8.0-host

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
    sed -i 's/#Port 22/Port 1954/' /etc/ssh/sshd_config

RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

COPY home/ /root/

# Necessary otherwise 'mpirun' is Ubuntu's and not doca's
ENV PATH="/usr/mpi/gcc/openmpi-4.1.9a1/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/mpi/gcc/openmpi-4.1.9a1/lib:$LD_LIBRARY_PATH"

ENTRYPOINT $HOME/init_ssh.sh && make -C $HOME/cpp && bash
