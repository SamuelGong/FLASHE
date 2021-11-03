"""
Author: Zhifeng Jiang From HKUST
Home-made logic for manipulating Github repo across multiple nodes.
"""

import sys
import json
from utils import ExecutionEngine


def initialize(launch_result_path, local_private_key_path,
               last_response_path, node_private_key_path,
               github_repo, project_dir):
    with open(launch_result_path, 'r') as fin:
        launch_result = json.load(fin)

    execution_plan_list = []
    for idx, simplified_response in enumerate(launch_result):
        name = simplified_response['name']
        public_ip = simplified_response['public_ip']
        execution_sequence = []

        copy_node_private_key = f"scp -q -i {local_private_key_path} " \
                                f"-o StrictHostKeyChecking=no " \
                                f"-o UserKnownHostsFile=/dev/null " \
                                f"{node_private_key_path} " \
                                f"ubuntu@{public_ip}:/home/ubuntu/.ssh/"
        execution_sequence.append((
            'local', [copy_node_private_key]
        ))
        execution_sequence.append((
            'remote',
            {
                'username': 'ubuntu',
                'key_filename': local_private_key_path,
                'commands': [
                    "ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts",
                    f"git clone {github_repo} {project_dir}",
                ]
            }
        ))
        execution_sequence.append((
            'prompt', [f"Initialized node {name} ({public_ip})."]
        ))

        execution_plan = {
            'name': name,
            'public_ip': public_ip,
            'execution_sequence': execution_sequence
        }
        execution_plan_list.append((execution_plan,))

    engine = ExecutionEngine()
    last_response = engine.run(execution_plan_list)
    with open(last_response_path, 'w') as fout:
        json.dump(last_response, fout, indent=4)


def general(launch_result_path, local_private_key_path,
            last_response_path, project_dir, core_command):
    with open(launch_result_path, 'r') as fin:
        launch_result = json.load(fin)

    execution_plan_list = []
    for idx, simplified_response in enumerate(launch_result):
        name = simplified_response['name']
        public_ip = simplified_response['public_ip']
        execution_sequence = [
            ('remote', {
                'username': 'ubuntu',
                'key_filename': local_private_key_path,
                'commands': [f'cd {project_dir} && git {core_command}']
            }),
            ('prompt', [f"Updated the repo "
                        f"{project_dir.split('/')[0]} on {name} ({public_ip})."])
        ]

        execution_plan = {
            'name': name,
            'public_ip': public_ip,
            'execution_sequence': execution_sequence
        }
        execution_plan_list.append((execution_plan,))

    engine = ExecutionEngine()
    last_response = engine.run(execution_plan_list)
    with open(last_response_path, 'w') as fout:
        json.dump(last_response, fout, indent=4)


def main(args):
    command, launch_result_path, \
        local_private_key_path, last_response_path = args[0:4]

    if command == "initialize":
        node_private_key_path, github_repo, project_dir = args[4:7]
        initialize(launch_result_path, local_private_key_path,
                   last_response_path, node_private_key_path,
                   github_repo, project_dir)
    elif command == "pull":
        project_dir = args[4]
        general(launch_result_path, local_private_key_path,
                last_response_path, project_dir, "pull")
    elif command == "checkout":
        repo_branch, project_dir = args[4:6]
        general(launch_result_path, local_private_key_path,
                last_response_path, project_dir, f"checkout {repo_branch}")
    else:
        print('Unknown commands!')


if __name__ == '__main__':
    main(sys.argv[1:])