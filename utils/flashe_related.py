import json
import sys
import os
import boto3
import botocore
from ruamel.yaml import YAML


yaml = YAML()
yaml.explicit_start = True
yaml.indent(mapping=2)

launch_result_path = os.path.join(os.getcwd(), 'simplified_launch_result.json')
flashe_config_path = os.path.join(os.getcwd(), '..',
                                  'deployment', 'cluster_conf.yml')
log_path = os.path.join(os.getcwd(), 'latest_cluster_management.log')
region_client_d = {}
result_d = {}

def load_yaml_conf(yaml_file):
    with open(yaml_file, 'r') as fin:
        data = yaml.load(fin)
    return data


def dump_yaml_conf(yaml_file, obj):
    with open(yaml_file, 'w') as fout:
        yaml.dump(obj, fout)


def update_flashe_config():
    server_ip = launch_result["server"]["private_ip"]
    leader_client_ip = launch_result["client"][0]["private_ip"]  # hard-coded

    other_clients_ips = []
    for client_record in launch_result["client"][1:]:
        other_clients_ips.append(client_record["private_ip"])

    orginal_conf = load_yaml_conf(flashe_config_path)
    orginal_conf['server_ip'] = server_ip
    orginal_conf['leader_client_ip'] = leader_client_ip
    orginal_conf['other_clients_ips'] = other_clients_ips
    dump_yaml_conf(flashe_config_path, orginal_conf)


def allow_ingress():
    region = launch_result["server"]["region"]
    region_client_d[region] = boto3.client('ec2', region_name=region)
    for client_record in launch_result["client"]:
        region = client_record["region"]
        if region not in region_client_d:
            region_client_d[region] = boto3.client('ec2', region_name=region)

    sgroup_region_d = {
        launch_result["server"]["security_group"] : launch_result["server"]["region"]
    }
    for client_record in launch_result["client"]:
        region = client_record["region"]
        security_group = client_record["security_group"]
        if security_group not in sgroup_region_d:
            sgroup_region_d[security_group] = region

    for sgroup, region in sgroup_region_d.items():
        try:
            _ = region_client_d[region].authorize_security_group_ingress(
                GroupId=sgroup,
                IpPermissions=[
                    {
                        'IpProtocol': 'all',
                        'IpRanges': [
                            {
                                'CidrIp': '0.0.0.0/0'
                            },
                        ],
                        'Ipv6Ranges': [
                            {
                                'CidrIpv6': '::/0'
                            },
                        ],
                    },
                ],
            )
        except botocore.exceptions.ClientError:
            # ignore if inbound rules exist
            pass

with open(launch_result_path, 'r') as file:
    launch_result = json.load(file)

if sys.argv[1] == "update_flashe_config":
    update_flashe_config()
elif sys.argv[1] == "allow_ingress":
    allow_ingress()
else:
    pass