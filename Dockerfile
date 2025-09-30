FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

RUN apt-get -y update
RUN apt-get -y install openmpi-bin openmpi-common libopenmpi-dev librdmacm-dev libpsm2-dev openmpi-bin libopenmpi-dev git

WORKDIR /app/nccl-tests
RUN git clone https://github.com/NVIDIA/nccl-tests .
RUN make -j MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/

#
# ssh-server
#

RUN apt-get install -y openssh-server
RUN apt-get install -y vim nano
RUN echo "root:root" | chpasswd

RUN mkdir /var/run/sshd && \
    sed -i 's/#Port 22/Port 1954/' /etc/ssh/sshd_config

RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

ENTRYPOINT service ssh restart && bash
