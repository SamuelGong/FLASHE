#!/bin/bash

# the absolute path of this script
WORKING_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

source $WORKING_DIR/init_env.sh
pip install -q paramiko

case "$1" in
    install)
        python ${GITHUB_HANDLER} initialize ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE} ${NODE_PRIVATE_KEY} \
          ${GITHUB_REPO} ${PROJECT_DIR}
        python ${GITHUB_HANDLER} checkout ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE} ${GITHUB_REPO} \
          ${REPO_BRANCH} ${PROJECT_DIR}
        python ${APP_HANDLER} standalone ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE} ${PROJECT_DIR} ${EXP_MGR_REL}
        ;;
    update)
        python ${GITHUB_HANDLER} pull ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE} ${GITHUB_REPO} ${PROJECT_DIR}
        ;;
    *)
        echo "Unknown command!"
        ;;
esac