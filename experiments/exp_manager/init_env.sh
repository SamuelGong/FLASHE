#!/bin/bash

WORKING_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# 1. Variables that should need modifying to suit your project
# 1.1 Repo information
GITHUB_REPO="git@github.com:SamuelGong/flashe.git"
REPO_BRANCH="master"
PROJECT_DIR="/home/ubuntu/flashe"  # where you install the project at each cluster node
LOCAL_PROJECT_DIR="${WORKING_DIR}/../.." # where you install the project at local host
APP_REL="experiments/app"

# 1.2 Credentials
LOCAL_PRIVATE_KEY="${HOME}/.ssh/MyKeyPair.pem"  # used by local host to access cluster nodes
NODE_PRIVATE_KEY="${HOME}/.ssh/id_rsa"  # used by cluster nodes to deal with private repos in Github

# 2. Variables that need not to change
# 2.1 Cluster configurations
EC2_CONFIG="${WORKING_DIR}/ec2_cluster_config.yml"
EC2_NODE_TEMPLATE="${WORKING_DIR}/ec2_node_template.yml"

# 2.2 Related programs
EC2_HANDLER="${WORKING_DIR}/ec2_related.py"
GITHUB_HANDLER="${WORKING_DIR}/github_related.py"
APP_HANDLER="${LOCAL_PROJECT_DIR}/${APP_REL}/main.py"  # one needs to customized this file

# 2.3 Places to store the logging results
EC2_LAUNCH_RESULT="${WORKING_DIR}/ec2_launch_result.json"
LAST_RESPONSE="${WORKING_DIR}/last_response.json"