FROM nvcr.io/nvidia/doca/doca:3.1.0-devel-cuda12.8.0-host

RUN apt-get -y update
RUN apt-get -y install openmpi-bin openmpi-common libopenmpi-dev librdmacm-dev libpsm2-dev openmpi-bin libopenmpi-dev git

WORKDIR /app/nccl-tests
RUN git clone https://github.com/NVIDIA/nccl-tests .
RUN make -j MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/

ENV OMPI_ALLOW_RUN_AS_ROOT=1 
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
#
# ssh-server
#

RUN apt-get install -y openssh-server
RUN apt-get install -y vim nano
RUN echo "root:root" | chpasswd

RUN mkdir /var/run/sshd && \
    sed -i 's/#Port 22/Port 1954/' /etc/ssh/sshd_config

RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

RUN apt -y install sshpass
WORKDIR /root
COPY init_ssh.sh hosts.txt .

ENTRYPOINT $HOME/init_ssh.sh && bash
