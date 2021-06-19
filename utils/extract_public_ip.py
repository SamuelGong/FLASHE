import json
import os

launch_result_path = os.path.join(os.getcwd(), 'simplified_launch_result.json')
output_path = os.path.join(os.getcwd(), 'temp.sh')

with open(launch_result_path, 'r') as file:
    launch_result = json.load(file)
public_ips = [launch_result["server"]["public_ip"]]

for client_record in launch_result["client"]:
    public_ips.append(client_record["public_ip"])

with open(output_path, 'w') as file:
    file.write(f"ips=({' '.join(public_ips)})")