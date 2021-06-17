#!/usr/bin/env bash

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

cd $(dirname "$0")
curtime=$(date +%Y%m%d%H%M%S)
work_mode=0
jobid="hetero_logistic_regression_example_standalone_"$curtime
guest_partyid=10000
host_partyid=9999
arbiter_partyid=10001

bash run_logistic_regression.sh $work_mode $jobid $guest_partyid $host_partyid $arbiter_partyid
