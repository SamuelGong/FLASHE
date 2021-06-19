# Cloud Deployment

This folder contains scripts and instructions for reproducing the experiments conducted under geo-distributed settings in our SoCC '21 paper. 
Our training evaluations rely on a distributed setting of ***multiple machines*** via the Parameter-Server (PS) architecture.
In our paper, we used 10 `c5.4xlarge` machines to simulate 10 participants and 1 `r5.4xlarge` machine to emulate the server in each round.
To be exact, their configurations are:

* OS kernel image: `ubuntu-bionic-18.04-amd64-server-20200903`;
* Computational capabilities: `16` vCPUs, `32` GB (`128` GB) memory (with `16G` swap) and `100` GiB SSD Storage.

## Overview

* [Prerequisites](#prerequisites)
* [Cluster Setup](#cluster-setup)
* [Job Configuration](#job-configuration)
* [Evaluation](#evaluation)

## Prerequisites

In order to have FATE v1.2.0 run at your cluster, in addition to using similar machines as what we played with, the cluster also needs to meet the following requirements:
1. Nodes within should be able to reach one another via **private IPs**. In other words, what you fill in `./cluster_conf.yml` should be private IPs instead of public ones.
2. Nodes should be able to perform remote login to one another **without** password or identification files specified.

If you are using resources offered by AWS, we are happy to provide you with some handy scripts at `../utils`, which should help you prepare an eligible cluster from scratch. Check out the [README](../utils/README.md) for practical instructions!

## Cluster Setup

Once you have established a several-node cluster which meets the prerequisites mentioned above,
things become simple as we have packed every necessary setup commands into one script or two. 
*Note that some commands related to FATE v1.2.0's [deployment](https://github.com/FederatedAI/FATE/tree/v1.2.0/cluster-deploy) may be ***intrusive*** to your systems. For example, it will extend some system limits and install dependencies in a global sense.*

1. At ***all nodes (server & clients)***, execute the following command.
    ```bash
    source ./all_nodes_prepare.sh
    ```
2. Then at ***server node***, make appropriate modifications on the configuration file [cluster_conf.yml](./cluster_conf.yml) (comments there can help you quickly understand how to specify the parameters) and run
    ```bash
    source ./server_deploy.sh cluster_conf.yml
    ```

## Job Configuration



## Evaluation

***NOTE: .***

The output of the experiment will validate the following major claims in our paper:
1. FLASHE outperforms batching versions of the three baselines by 3.2×-15.1× in iteration time and 2.1×-42.4× in network footprint. Compared to plaintext training, FLASHE achieves near-optimality efficiency with overhead ≤6% and 0% in traffic (§6.2) —> Figure 7 and 8.
2. FLASHE brings down at least 13×-63× of the computation overhead and 48× of the communication overhead of general HEs when sparsification is employed (§6.4) —> Figure 10 and 11.
3. FLASHE's mask precomputation effectively eliminates the cryptographic overhead (rendering it < 0.1s) (§6.6) —> Figure 13.

### Per Iteration Efficiency (Figure 7 and Figure 8)

#### Figure 7

#### Figure 8
