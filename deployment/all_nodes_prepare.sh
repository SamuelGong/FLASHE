#!/bin/bash
PROJECT_DIR=`pwd`/../

# cannot even proceed with FATE if not having sudo privilege
echo "[INFO] Check sudo privilege..."
if ! groups | grep "\<sudo\>" &> /dev/null; then
   echo "[FAILED] You need to have sudo priviledge for deploying FATE"
fi

# install anaconda if necessary
CONDA_DIR=$HOME/anaconda3
if [ ! -d $CONDA_DIR ]; then
  echo "[INFO] Install Anaconda Package Manager..."
  wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
  bash Anaconda3-2020.11-Linux-x86_64.sh -b -p $CONDA_DIR
  export PATH=$CONDA_DIR/bin:$PATH
  rm Anaconda3-2020.11-Linux-x86_64.sh
  conda init bash
  source ~/.bashrc
fi

# These are explicitly required by FATE's cluster deployment
echo "[INFO] Make deployment directory and extend system limits..."
sudo mkdir -p /data/projects
sudo chown -R ubuntu:ubuntu /data/projects
sudo sed -i '/^\*\s\{1,\}soft\s\{1,\}nofile/{h;s/nofile\s\{1,\}.*/nofile 65536/};${x;/^$/{s//\* soft nofile 65536/;H};x}' /etc/security/limits.conf
sudo sed -i '/^\*\s\{1,\}hard\s\{1,\}nofile/{h;s/nofile\s\{1,\}.*/nofile 65536/};${x;/^$/{s//\* hard nofile 65536/;H};x}' /etc/security/limits.conf
sudo sed -i '/^\*\s\{1,\}soft\s\{1,\}nproc/{h;s/nproc\s\{1,\}.*/nproc unlimited/};${x;/^$/{s//\* soft nproc unlimited/;H};x}' /etc/security/limits.d/20-nproc.conf

# These are necessary to accommodate FATE's CentOS scripts
# when you are running atop Ubuntu
# make /bin/sh symlink to bash instead of dash
echo "[INFO] Adapt to Ubuntu..."
echo "dash dash/sh boolean false" | sudo debconf-set-selections
sudo dpkg-reconfigure -f noninteractive dash
sudo apt update
sudo apt-get install -y gcc g++ make openssl supervisor libgmp-dev  libmpfr-dev libmpc-dev libaio1 libaio-dev numactl autoconf automake libtool libffi-dev libssl1.0.0 libssl-dev liblz4-1 liblz4-dev liblz4-1-dbg liblz4-tool  zlib1g zlib1g-dbg zlib1g-dev libgflags-dev
cd /usr/lib/x86_64-linux-gnu
if [ ! -f "libssl.so.10" ];then
    sudo ln -s libssl.so.1.0.0 libssl.so.10
    sudo ln -s libcrypto.so.1.0.0 libcrypto.so.10
fi
if [ ! -f "libgflags.so.2" ];then
    sudo ln -s libgflags.so.2.2 libgflags.so.2
fi

# FLASHE-related
cd ${PROJECT_DIR}
cat ./cluster-deploy/original_requirements.txt > ./cluster-deploy/requirements.txt
cat ./deployment/additional_requirements.txt >> ./cluster-deploy/requirements.txt
echo "[SUCCEEDED] Finish"
