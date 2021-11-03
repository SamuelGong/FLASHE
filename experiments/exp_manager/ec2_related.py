"""
Author: Zhifeng Jiang from HKUST

A handy tool for launching/starting/stopping/restarting/terminating
AWS EC2 instances in a command line way
"""

import sys
import boto3
import botocore
import json
from datetime import date, datetime
import yaml


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, date):
            return obj.strftime('%Y-%m-%d')
        else:
            return json.JSONEncoder.default(self, obj)


def launch_a_node(client, node_config):
    return client.run_instances(BlockDeviceMappings=node_config['BlockDeviceMappings'],
                                InstanceType=node_config['InstanceType'],
                                TagSpecifications=node_config['TagSpecifications'],
                                NetworkInterfaces=node_config['NetworkInterfaces'],
                                ImageId=node_config['ImageId'],
                                KeyName=node_config['KeyName'],
                                MinCount=1, MaxCount=1)


def parse_cluster_config(cluster_config_path):
    with open(cluster_config_path, 'r') as fin:
        cluster_config = yaml.load(fin, Loader=yaml.FullLoader)

    node_list = []
    for client_dict in cluster_config['clients']:
        the_only_key = list(client_dict.keys())[0]
        node_list.append(client_dict[the_only_key])
    node_list.append(cluster_config['server'])

    subnet_dict = cluster_config['subnets']
    image_dict = cluster_config['images']
    return node_list, subnet_dict, image_dict


def open_inbound_ports(client, security_group, ports):
    response = []
    for port in ports:
        try:
            resp = client.authorize_security_group_ingress(
                GroupId=security_group,
                IpPermissions = [{
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
                    'FromPort': port,
                    'ToPort': port,
                }]
            )
        except botocore.exceptions.ClientError as e:  # ignore if inbound rules exist
            resp = e.response['Error']['Code']
        finally:
            response.append({port: resp})
    return response


def launch_a_cluster(cluster_config_path, node_template_path,
                     last_response_path, launch_result_path):
    with open(node_template_path, 'r') as fin:
        node_template = yaml.load(fin, Loader=yaml.FullLoader)

    launch_result = []
    region_to_boto3_client_mapping = {}  # only create one client for a region
    region_to_instance_id_mapping = {}
    node_list, subnet_dict, image_dict = parse_cluster_config(cluster_config_path)
    num_nodes_to_launch = len(node_list)
    last_response = []

    print(f"Launching {num_nodes_to_launch} nodes ...")
    for node_base_config in node_list:
        region = node_base_config['region']
        instance_type = node_base_config['type']
        name = node_base_config['name']

        if region in region_to_boto3_client_mapping:
            client = region_to_boto3_client_mapping[region]
        else:
            client = boto3.client('ec2', region_name=region)
            region_to_boto3_client_mapping[region] = client

        node_template["InstanceType"] = instance_type
        node_template["TagSpecifications"][0]["Tags"][0]["Value"] = name
        node_template["NetworkInterfaces"][0]["SubnetId"] = subnet_dict[region]
        node_template["ImageId"] = image_dict[region]

        launch_response = launch_a_node(client, node_template)
        useful_part = launch_response['Instances'][0]
        instance_id = useful_part['InstanceId']
        private_ip = useful_part['NetworkInterfaces'][0]['PrivateIpAddress']
        security_group = useful_part['SecurityGroups'][0]["GroupId"]
        if "coordinator" in name:
            ports = [22, 80]
        else:
            ports = [22]
        security_response = open_inbound_ports(client, security_group, ports)
        last_response.append({
            'name': name,
            'launch_response': launch_response,
            'allow_ingress_response': security_response
        })

        simplified_response = {
            "name": name,
            "id": instance_id,
            "region": region,
            "private_ip": private_ip,
            "security_group": security_group
        }
        launch_result.append(simplified_response)

        if region not in region_to_instance_id_mapping:
            region_to_instance_id_mapping[region] = [instance_id]
        else:
            region_to_instance_id_mapping[region].append(instance_id)

    print(f"All {num_nodes_to_launch} nodes are launched! Waiting for ready ...")
    with open(last_response_path, 'w') as fout:
        json.dump(last_response, fout, indent=4, cls=ComplexEncoder)

    for region, instance_ids in region_to_instance_id_mapping.items():
        client = region_to_boto3_client_mapping[region]
        waiter = client.get_waiter('instance_running')
        waiter.wait(
            InstanceIds=instance_ids,
            WaiterConfig={
                'Delay': 1,
                'MaxAttempts': 120
            }
        )

    print(f"All {num_nodes_to_launch} nodes are ready. Collecting public IP addresses ...")
    for idx, simplified_response in enumerate(launch_result):
        instance_id = simplified_response['id']
        region = simplified_response['region']
        client = region_to_boto3_client_mapping[region]
        description = client.describe_instances(InstanceIds=[instance_id])
        public_ip = description['Reservations'][0]['Instances'][0]['PublicIpAddress']
        launch_result[idx]['public_ip'] = public_ip

    with open(launch_result_path, 'w') as fout:
        json.dump(launch_result, fout, indent=4, cls=ComplexEncoder)


def merge_instances_by_region(launch_result):
    region_to_instance_ids_mapping = {}
    for simplified_response in launch_result:
        region = simplified_response['region']
        instance_id = simplified_response['id']
        if region not in region_to_instance_ids_mapping:
            region_to_instance_ids_mapping[region] = [instance_id]
        else:
            region_to_instance_ids_mapping[region].append(instance_id)

    return region_to_instance_ids_mapping


def ec2_actions_on_a_cluster(action, last_response_path, launch_result_path):
    with open(launch_result_path, 'r') as fin:
        launch_result = json.load(fin)

    region_to_instance_ids_mapping = merge_instances_by_region(launch_result)
    region_to_boto3_clients_mapping = {}
    last_response = {}
    for region, instance_ids in region_to_instance_ids_mapping.items():
        client = boto3.client('ec2', region_name=region)
        region_to_boto3_clients_mapping[region] = client
        if action == 'start':
            response = client.start_instances(InstanceIds=instance_ids)
        elif action == 'reboot':
            response = client.reboot_instances(InstanceIds=instance_ids)
        elif action == 'stop':
            response = client.stop_instances(InstanceIds=instance_ids)
        elif action == 'terminate':
            response = client.terminate_instances(InstanceIds=instance_ids)
        else:
            response = None

        last_response[region] = response

    # if it is a start action,
    # we additionally need to record the new public addresses
    if action == "start":
        print("Started and waiting for ready ...")
        for region, instance_ids in region_to_instance_ids_mapping.items():
            client = region_to_boto3_clients_mapping[region]
            waiter = client.get_waiter('instance_running')
            waiter.wait(
                InstanceIds=instance_ids,
                WaiterConfig={
                    'Delay': 1,
                    'MaxAttempts': 120,
                }
            )

        print("All are ready. Collecting public IP addresses ...")
        for idx, node_config in enumerate(launch_result):
            region = node_config['region']
            client = region_to_boto3_clients_mapping[region]

            instance_id = node_config['id']
            description = client.describe_instances(InstanceIds=[instance_id])
            public_ip = description['Reservations'][0]['Instances'][0]['PublicIpAddress']
            launch_result[idx]['public_ip'] = public_ip

        with open(launch_result_path, 'w') as fout:
            json.dump(launch_result, fout, indent=4)

    with open(last_response_path, 'w') as fout:
        json.dump(last_response, fout, indent=4)

    print('Done.')


def main(args):
    command = args[0]

    if command == "launch":
        cluster_config_path = args[1]
        node_template_path = args[2]
        last_response_path = args[3]
        launch_result_path = args[4]
        launch_a_cluster(cluster_config_path, node_template_path,
                         last_response_path, launch_result_path)
    elif command in ["start", "stop", "reboot", "terminate"]:
        last_response_path = args[1]
        launch_result_path = args[2]
        ec2_actions_on_a_cluster(command, last_response_path,
                                 launch_result_path)
    elif command == 'show':
        launch_result_path = args[1]
        with open(launch_result_path, 'r') as fin:
            launch_result = json.load(fin)
        for node in launch_result:
            print(f'{node["name"]}: {node["public_ip"]}')


if __name__ == '__main__':
    main(sys.argv[1:])
