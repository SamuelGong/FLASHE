import json
import os
import sys
import subprocess

repo_url = "git@github.com:SamuelGong/FLASHE.git"
launch_result_path = os.path.join(os.getcwd(), 'simplified_launch_result.json')
log_path = os.path.join(os.getcwd(), 'latest_github.log')

with open(launch_result_path, 'r') as file:
    launch_result = json.load(file)
public_ips = [launch_result["server"]["public_ip"]]

for client_record in launch_result["client"]:
    public_ips.append(client_record["public_ip"])

# clone if not exist
def git_clone():
    # actually new a log file here
    with open(log_path, 'wb') as _:
        pass

    for ip in public_ips:
        print(f"git clone at {ip}...")
        cmd = f"ssh ubuntu@{ip} '[ -d FLASHE ] || git clone {repo_url}' &"
        with open(log_path, 'a') as fout:
            subprocess.Popen(cmd, shell=True, stdout=fout, stderr=fout)

if sys.argv[1] == "clone":
    git_clone()
else:
    pass