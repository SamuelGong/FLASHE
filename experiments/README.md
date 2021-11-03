# Experiments

This folder aims to be an all-in-one guidance for playing with FLASHE atop FATE from scratch and reproducing results in the paper. This document assumes the use of **AWS EC2** service, though, it should be easy to generalize the steps to other types of environments.

## 1. Installation

### 1.1 Prerequisites

At first, one needs to

1. git pull exp_manager and modify the init_env.sh
2. aws configure

### 1.2 Launching a Cluster
Files [ec2_cluster_config.yml](./exp_manager/ec2_cluster_config.yml) and [ec2_node_template.yml](./exp_manager/ec2_node_template.yml) dictates the necessary details for launch an EC2 cluster. Consider modifying them to best suit your need (refer to exp_manager's [manual](#) for detailed information). Now you should be able to launch your customized cluster at the current folder by

```bash
bash exp_manager/manage_cluster.sh launch
```

*p.s., One can stop/reboot/start/terminate the cluster now simply by executing command `bash exp_manager/manage_cluster.sh [stop/reboot/start/terminate]` at the current folder.*

### 1.3 Installing FATE and FLASHE

Execute the shell commands below which:
1. Clone the project FATE at each cluster node and install necessary dependencies;

```bash
bash exp_manager/manage_app.sh install
```

## 2. Replicating Results in the Paper