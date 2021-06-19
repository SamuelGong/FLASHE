import yaml
import sys
import boto3
import json
import os
from datetime import date, datetime

name_prefix = 'flashe.'
region_client_d = {}
region_subnet_d = {}
region_image_d = {}
region_idlist_d = {}
result_d = {}
simplified_result_d = {}

class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, date):
            return obj.strftime('%Y-%m-%d')
        else:
            return json.JSONEncoder.default(self, obj)


def load_yaml_conf(yaml_file):
    with open(yaml_file) as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data


def launch(client, template):
    return client.run_instances(BlockDeviceMappings=template['BlockDeviceMappings'],
                                InstanceType=template['InstanceType'],
                                TagSpecifications=template['TagSpecifications'],
                                NetworkInterfaces=template['NetworkInterfaces'],
                                ImageId=template['ImageId'],
                                KeyName=template['KeyName'],
                                MinCount=1, MaxCount=1)

def start(cluster_conf_file, node_template_file):
    cluster_conf = load_yaml_conf(cluster_conf_file)
    node_template = load_yaml_conf(node_template_file)

    for region, subnet_id in cluster_conf["subnet"].items():
        client = boto3.client('ec2', region_name=region)
        region_client_d[region] = client
        region_subnet_d[region] = subnet_id
        region_image_d[region] = cluster_conf["image"][region]

    inst_count = 1
    region = cluster_conf["server"]["region"]

    node_template["InstanceType"] = cluster_conf["server_type"]
    node_template["TagSpecifications"][0]["Tags"][0]["Value"] = name_prefix + str(inst_count)
    node_template["NetworkInterfaces"][0]["SubnetId"] = region_subnet_d[region]
    node_template["ImageId"] = region_image_d[region]
    resp = launch(region_client_d[region], node_template)
    result_d["server"] = resp['Instances'][0]
    simplified_result_d["server"] = {
        "name": name_prefix + str(inst_count),
        "id": resp['Instances'][0]['InstanceId'],
        "region": region,
        "private_ip":  resp['Instances'][0]['NetworkInterfaces'][0]['PrivateIpAddress'],
        "security_group": resp['Instances'][0]['SecurityGroups'][0]["GroupId"]
    }
    region_idlist_d[region] = [resp['Instances'][0]['InstanceId']]

    result_d["client"] = []
    simplified_result_d["client"] = []
    node_template["InstanceType"] = cluster_conf["client_type"]
    for region, count in cluster_conf["clients"]["region"].items():
        for _ in range(count):
            inst_count += 1

            node_template["TagSpecifications"][0]["Tags"][0]["Value"] = name_prefix + str(inst_count)
            node_template["NetworkInterfaces"][0]["SubnetId"] = region_subnet_d[region]
            node_template["ImageId"] = region_image_d[region]
            resp = launch(region_client_d[region], node_template)
            result_d["client"].append(resp['Instances'][0])

            simplified_result_d["client"].append({
                "name": name_prefix + str(inst_count),
                "id": resp['Instances'][0]['InstanceId'],
                "region": region,
                "private_ip": resp['Instances'][0]['NetworkInterfaces'][0]['PrivateIpAddress'],
                "security_group": resp['Instances'][0]['SecurityGroups'][0]["GroupId"]
            })
            if region not in region_idlist_d:
                region_idlist_d[region] = [resp['Instances'][0]['InstanceId']]
            else:
                region_idlist_d[region].append(resp['Instances'][0]['InstanceId'])

    output_path = os.path.join(os.getcwd(), 'launch_result.json')
    with open(output_path, 'w') as file:
        json.dump(result_d, file, indent=4, cls=ComplexEncoder)


    print("All are launched! Waiting for ready...")
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
    id = simplified_result_d["server"]["id"]
    region = simplified_result_d["server"]["region"]
    desc = region_client_d[region].describe_instances(InstanceIds=[id])
    simplified_result_d["server"]["public_ip"] \
        = desc['Reservations'][0]['Instances'][0]['PublicIpAddress']
    for idx, client_record in enumerate(simplified_result_d["client"]):
        region = client_record["region"]
        id = client_record["id"]
        desc = region_client_d[region].describe_instances(InstanceIds=[id])
        simplified_result_d["client"][idx]["public_ip"] \
            = desc['Reservations'][0]['Instances'][0]['PublicIpAddress']

    output_path = os.path.join(os.getcwd(), 'simplified_launch_result.json')
    with open(output_path, 'w') as file:
        json.dump(simplified_result_d, file, indent=4)


start(sys.argv[1], sys.argv[2])