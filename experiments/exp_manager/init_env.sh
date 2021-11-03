#!/bin/bash

WORKING_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# Your configurations
EC2_CONFIG="${WORKING_DIR}/ec2_cluster_config.yml"
EC2_NODE_TEMPLATE="${WORKING_DIR}/ec2_node_template.yml"

# Places to store the results
EC2_LAUNCH_RESULT="${WORKING_DIR}/ec2_launch_result.json"
LAST_RESPONSE="${WORKING_DIR}/last_response.json"

# Related programs
EC2_HANDLER="${WORKING_DIR}/ec2_related.py"
GITHUB_HANDLER="${WORKING_DIR}/github_related.py"
APP_HANDLER="${WORKING_DIR}/app/main.py"  # one needs to customized this file

# Repo information
GITHUB_REPO="git@github.com:SamuelGong/flashe.git"
REPO_BRANCH="master"
PROJECT_DIR="/home/ubuntu/flashe"  # where you install the project at each cluster node
EXP_MGR_REL="experiments"

# Credentials
LOCAL_PRIVATE_KEY="${HOME}/.ssh/MyKeyPair.pem"  # used by local host to access cluster nodes
NODE_PRIVATE_KEY="${HOME}/.ssh/id_rsa"  # used by cluster nodes to deal with private repos in Github