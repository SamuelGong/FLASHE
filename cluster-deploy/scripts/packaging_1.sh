#!/bin/bash

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

set -e
cwd=$(cd `dirname $0`; pwd)
cd ${cwd}
source_code_dir=$(cd `dirname ${cwd}`; cd ../; pwd)
echo "[INFO] Source code dir is ${source_code_dir}"
packages_dir=${source_code_dir}/cluster-deploy/packages
mkdir -p ${packages_dir}

cd ${source_code_dir}
eggroll_git_url=`grep -A 3 '"eggroll"' .gitmodules | grep 'url' | awk -F '= ' '{print $2}'`
eggroll_git_branch=`grep -A 3 '"eggroll"' .gitmodules | grep 'branch' | awk -F '= ' '{print $2}'`
echo "[INFO] Git clone eggroll submodule source code from ${eggroll_git_url} branch ${eggroll_git_branch}"
if [[ ! -e "eggroll" ]];then
    git clone ${eggroll_git_url} -b ${eggroll_git_branch} eggroll
fi

cd ${source_code_dir}
fateboard_git_url=`grep -A 3 '"fateboard"' .gitmodules | grep 'url' | awk -F '= ' '{print $2}'`
fateboard_git_branch=`grep -A 3 '"fateboard"' .gitmodules | grep 'branch' | awk -F '= ' '{print $2}'`
echo "[INFO] Git clone fateboard submodule source code from ${fateboard_git_url} branch ${fateboard_git_branch}"
if [[ ! -e "fateboard" ]];then
    git clone ${fateboard_git_url} -b ${fateboard_git_branch} fateboard
fi
