#!/bin/bash

# the absolute path of this script
WORKING_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

source $WORKING_DIR/init_env.sh
pip install -q boto3

case "$1" in
    launch)
        python ${EC2_HANDLER} launch ${EC2_CONFIG} ${EC2_NODE_TEMPLATE} ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT}
        ;;
    start)
        python ${EC2_HANDLER} start ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT}
        ;;
    stop)
        python ${EC2_HANDLER} stop ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT}
        ;;
    terminate)
        python ${EC2_HANDLER} terminate ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT}
        ;;
    reboot)
        python ${EC2_HANDLER} reboot ${LAST_RESPONSE} ${EC2_LAUNCH_RESULT}
        ;;
    show)
        python ${EC2_HANDLER} show ${EC2_LAUNCH_RESULT}
        ;;
    *)
        echo "Unknown command!"
        ;;
esac