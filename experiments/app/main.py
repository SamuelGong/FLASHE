"""
Author: Zhifeng Jiang From HKUST
Home-made logic for installing, deploying and using FATE it in a cluster
"""

import sys
import json
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import ExecutionEngine


def main(args):
    command, launch_result_path, local_private_key_path, \
        last_response_path, project_dir, app_rel = args[0:6]
    with open(launch_result_path, 'r') as fin:
        launch_result = json.load(fin)

    execution_plan_list = []
    remote_template = {
        'username': 'ubuntu',
        'key_filename': local_private_key_path
    }

    for idx, simplified_response in enumerate(launch_result):
        name = simplified_response['name']
        public_ip = simplified_response['public_ip']
        execution_sequence = []

        if command == "standalone":
            remote_template.update({
                'commands': [
                    f"cd {project_dir}/{app_rel}/ "
                    "&& source standalone_install.sh"
                ]
            })
            execution_sequence = [
                ('remote', remote_template),
                ('prompt', [f'Standalone installation finished on node '
                            f'{name} ({public_ip}).'])
            ]

        elif command == "deploy_cluster":
            if 'coordinator' in name:
                remote_template.update({
                    'commands': [
                        f"cd {project_dir}/{app_rel}/ && source cluster_install.sh"
                    ]
                })
                execution_sequence = [
                    ('remote', remote_template),
                    ('prompt', [f'Cluster server deployed on '
                                f'{name} ({public_ip}).'])
                ]

        elif command == "start_cluster":
            if 'coordinator' in name:
                remote_template.update({
                    'commands': [
                        "sudo systemctl start nginx"
                    ]
                })
                execution_sequence = [
                    ('remote', remote_template),
                    ('prompt', [f'Cluster server started on '
                                f'{name} ({public_ip}).'])
                ]

        else:
            execution_sequence = []

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


if __name__ == '__main__':
    main(sys.argv[1:])