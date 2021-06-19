import json
import sys
import os
import boto3
from ruamel.yaml import YAML

yaml = YAML()
yaml.explicit_start = True
yaml.indent(mapping=2)

launch_result_path = os.path.join(os.getcwd(), 'simplified_launch_result.json')
log_path = os.path.join(os.getcwd(), 'latest_cluster_management.log')
region_client_d = {}
region_idlist_d = {}
region_publicip_d = {}
result_d = {}


def load_yaml_conf(yaml_file):
    with open(yaml_file, 'r') as fin:
        data = yaml.load(fin)
    return data


def act(action, client, id_list):
    if action == 'start':
        resp = client.start_instances(
            InstanceIds=id_list
        )
    elif action == "stop":
        resp = client.stop_instances(
            InstanceIds=id_list
        )
    elif action == "terminate":
        resp = client.terminate_instances(
            InstanceIds=id_list
        )
    else:
        resp = None
    return resp


def start(action):
    region = launch_result["server"]["region"]
    region_idlist_d[region] = [launch_result["server"]["id"]]

    region_client_d[region] = boto3.client('ec2', region_name=region)
    for client_record in launch_result["client"]:
        region = client_record["region"]
        id = client_record["id"]
        if region not in region_client_d:
            region_client_d[region] = boto3.client('ec2', region_name=region)
        if region not in region_idlist_d:
            region_idlist_d[region] = [id]
        else:
            region_idlist_d[region].append(id)

    for region, id_list in region_idlist_d.items():
        resp = act(action, region_client_d[region], id_list)
        result_d["region"] = resp
    with open(log_path, 'w') as file:
        json.dump(result_d, file, indent=4)

    # if start, wait for running (so that IPs are available)
    if action == "start":
        print("Started and waiting for ready...")
        for region, id_list in region_idlist_d.items():
            waiter = region_client_d[region].get_waiter('instance_running')
            waiter.wait(
                InstanceIds=id_list,
                WaiterConfig={
                    'Delay': 1,
                    'MaxAttempts': 120
                }
            )
        print("All are ready. Collect Public Ips...")
        id = launch_result["server"]["id"]
        region = launch_result["server"]["region"]
        desc = region_client_d[region].describe_instances(InstanceIds=[id])
        launch_result["server"]["public_ip"] \
            = desc['Reservations'][0]['Instances'][0]['PublicIpAddress']
        for idx, client_record in enumerate(launch_result["client"]):
            region = client_record["region"]
            id = client_record["id"]
            desc = region_client_d[region].describe_instances(InstanceIds=[id])
            launch_result["client"][idx]["public_ip"] \
                = desc['Reservations'][0]['Instances'][0]['PublicIpAddress']

        with open(launch_result_path, 'w') as file:
            json.dump(launch_result, file, indent=4)
        print("Done.")


with open(launch_result_path, 'r') as file:
    launch_result = json.load(file)
start(sys.argv[1])