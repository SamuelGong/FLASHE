# Utilities

This directory contains several auxiliary scripts and files that help improve your user experience.

## Overview

* [Preparing an Eligible Cluster atop AWS EC2](#preparing-an-eligible-cluster-atop-aws-ec2)

## Preparing an Eligible Cluster atop AWS EC2
This guide assumes that you are willing to use AWS EC2 resources to deploy FLASHE in the cloud leveraging [AWS SDK for Python](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html).
***Note that we do not combine all the steps into one single script here, as we would like to show you how each of them function separately so that you may be able to also transfer some of them to your other projects for improving the engineering efficiency there :)***

### Step 1: Launch Instances in the Same Private Network

#### If You Want to Build a Cluster in the Same Region

Dependent configuration files:
* `./cluster_plan.yml` specifies the profile of the cluster;
* `./instance_template.yml` enables customization of each instance.

Have them all set before running this at your local host:

```bash
bash launch.sh
```

##### Remarks

* On success, the launching details will be silently saved to `./launch_result.json` and `./simplified_launch_result.json` for your further reference. Particularly, with `./simplified_launch_result.json` well preserved, we are pleased to provide you `./batch_manager.py` for further facilitating the cluster management. You can start/stop/teminate all nodes in the cluster in one command by:

    ```bash
    python batch_manager.py start/stop/terminate
    ```

#### If You Want to Build a Geo-Distributed Cluster

In this case, the command is the same, except that before executing you first need to assure that every geo-distributed private network (i.e., every subnet specified in `./cluster_plan.yml`) is able to directly connect with each other. For instance, node `a` in region `A` can ping node `b` in region `B` only with `b`'s private IP specified. While this is by default violating the definition of ''private networks'', we can actually achieve it in AWS by [VPC Peering](https://docs.aws.amazon.com/zh_cn/vpc/latest/userguide/vpc-subnets-commands-example.html).

### Step 2: Configure Node-Wise Passwordless SSH Login

Dependent configuration file:

* `./ssh_conf.yml` where you only need to specify:
    1. the SSH login username (which is assumed to be identical across all nodes in the cluster, e.g., `ubuntu` by default for AWS EC2 Ubuntu machines), and
    2. the path to your local machine's private key which has been authorized by all nodes in the cluster (e.g., `MyKeyPair.pem` by default for AWS EC2 instances).

After the file is set, just run

```bash
bash passwordless.sh
```

#### Remarks
* From now on, you can (1) perform SSH login to any node in the cluster from your local machine without the need for a password or identification file (e.g., MyKeyPair.pem); and (2) nodes in the cluster can also perform SSH login to one another in such passwordless fashion. While (2) may not be necessary for you and thus the related logic can be commented in the script, (1) is indispensable if you want to deploy FLASHE. 
 


