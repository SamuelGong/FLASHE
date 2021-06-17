import yaml
import os
import sys
import subprocess
import json
from collections import defaultdict

all_ips = []


def rec_d():
    return defaultdict(rec_d)


def load_yaml_conf(yaml_file):
    with open(yaml_file) as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data


def multinode_cluster_configurations(yaml_file):
    yaml_conf = load_yaml_conf(yaml_file)

    all_ips.append(yaml_conf['server_ip'])
    all_ips.append(yaml_conf['leader_client_ip'])
    for ip in yaml_conf['other_clients_ips']:
        all_ips.append(ip)
    common_ssh_user = yaml_conf['auth']['common_ssh_user']

    lines = ["#!/bin/bash\n", f'user={common_ssh_user}', "deploy_dir=/data/projects/fate",
             "db_auth=(root fate_dev)", "redis_password=fate_dev", "cxx_compile_flag=false\n"]

    party_list = []
    party_names = []
    for idx, ip in enumerate(all_ips):
        lines.append(f"# services for {str(idx)}")
        lines.append(f"p{str(idx)}_mysql={ip}")
        lines.append(f"p{str(idx)}_redis={ip}")
        lines.append(f"p{str(idx)}_fate_flow={ip}")
        lines.append(f"p{str(idx)}_fateboard={ip}")
        lines.append(f"p{str(idx)}_federation={ip}")
        lines.append(f"p{str(idx)}_proxy={ip}")
        lines.append(f"p{str(idx)}_roll={ip}")
        lines.append(f"p{str(idx)}_metaservice={ip}")
        lines.append(f"p{str(idx)}_egg={ip}\n")

        party_names.append('p' + str(idx))
        party_list.append(str(idx))

    lines.append(f"party_list=({' '.join(party_list)})")
    lines.append(f"party_names=({' '.join(party_names)})")

    multinode_config_path = os.path.join(os.getcwd(), '..', 'cluster-deploy',
                                         'scripts', 'multinode_cluster_configurations.sh')
    with open(multinode_config_path, 'w') as file:
        file.writelines('\n'.join(lines))


def route_table():
    route_table_path = os.path.join(os.getcwd(), '..', 'arch', 'networking',
                                    'proxy', 'src', 'main', 'resources',
                                    'route_tables', 'route_table.json')
    result = rec_d()
    result["permission"]["default_allow"] = True
    result["route_table"]["default"]["default"] = [{
        "ip": all_ips[0],
        "port": 9370
    }]

    for idx, ip in enumerate(all_ips):
        result["route_table"][str(idx)]["fate"] = [{
            "ip": ip,
            "port": 9394
        }]
        result["route_table"][str(idx)]["fateflow"] = [{
            "ip": ip,
            "port": 9360
        }]

    with open(route_table_path, 'w') as file:
        json.dump(result, file, indent=4)


multinode_cluster_configurations(sys.argv[1])
route_table()
