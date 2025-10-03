#!/bin/sh

cd "$HOME"
mkdir -p .ssh

service ssh restart
ssh-keygen -t rsa -b 4096 -N "" -f ".ssh/id_rsa"

# Populate .ssh/config
awk '{print "Host "$1"\n    HostName "$2"\n    Port 1954\n"}' hosts.txt >> .ssh/config

copy_ssh_key()
{
    if [ -z "$1" ]; then
        echo "Usage: copy_ssh_key <host>"
        return 1
    fi

    host="$1"

    echo "Waiting for $host to become reachable..."

    # Loop until ssh works
    until sshpass -p 'root' ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no "$host" true 2>/dev/null; do
        sleep 2
    done

    echo "$host is up, copying SSH key..."
    sshpass -p 'root' ssh-copy-id -o StrictHostKeyChecking=no "$host"
}

copy_ssh_key smartedge
copy_ssh_key hpe

if [ "$(hostname)" = "smartedge" ]; then
    mpirun -x NCCL_DEBUG=INFO --pernode --host smartedge,hpe /root/cpp/main -n 10000
fi

exec bash