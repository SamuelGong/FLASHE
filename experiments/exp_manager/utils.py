import subprocess
import time
from multiprocessing import Pool, cpu_count
from paramiko.client import SSHClient, AutoAddPolicy
from datetime import datetime


def chunks_idx(l, n):
    d, r = divmod(l, n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield si, si + (d + 1 if i < r else d)


def execute_locally(commands):
    resps = []
    for command in commands:
        process = subprocess.Popen(command.split(),
                                   shell=False,
                                   stdout=subprocess.PIPE)
        resp = process.communicate()[0].strip().decode("utf-8")
        resps.append({
            'local_command': command,
            'time': datetime.now().strftime("(%Y-%m-%d) %H:%M:%S"),
            'resp': resp,
        })
    return resps


def ends_with_and(command):
    l = command.split(' ')
    for i in range(len(l)-1, -1, -1):
        word = l[i]
        if len(word) == 0:
            continue
        if word == '&':
            return True
        else:
            return False
    return False


def execute_remotely(commands, hostname, username, key_filename):
    ssh_client = SSHClient()
    ssh_client.set_missing_host_key_policy(AutoAddPolicy())
    while True:
        try:
            ssh_client.connect(hostname, username=username, key_filename=key_filename,
                               banner_timeout=15)
        except Exception as e:
            print(f'Encountered exception: {e}, will retry soon ...')
            time.sleep(1)
        else:
            break

    resps = []
    for command in commands:
        _, stdout, stderr = ssh_client.exec_command(command)
        if not ends_with_and(command):
            resps.append({
                'remote_command': command,
                'time': datetime.now().strftime("(%Y-%m-%d) %H:%M:%S"),
                'stdout': stdout.readlines(),
                'stderr': stderr.readlines(),
            })
        else:
            resps.append({
                'remote_command': command,
                'time': datetime.now().strftime("(%Y-%m-%d) %H:%M:%S"),
                'stdout': "",
                'stderr': "",
            })
    ssh_client.close()
    return resps


def execute_for_a_node(execution_plan):
    name = execution_plan['name']
    public_ip = execution_plan['public_ip']
    execution_sequence = execution_plan['execution_sequence']
    response = []

    for action, payload in execution_sequence:
        if action == "prompt":
            for line in payload:
                print(line)
        elif action == "local":
            response.extend(execute_locally(payload))
        elif action == "remote":
            commands = payload["commands"]
            username = payload["username"]
            key_filename = payload["key_filename"]
            response.extend(execute_remotely(commands, public_ip,
                                             username, key_filename))

    result = {
        'name': name,
        'public_ip': public_ip,
        'response': response
    }
    return result


class ExecutionEngine(object):
    def __init__(self):
        super(ExecutionEngine, self).__init__()
        self.n_jobs = cpu_count()

    def run(self, execution_plan_list):
        with Pool(processes=self.n_jobs) as pool:
            result_list = pool.starmap(execute_for_a_node, execution_plan_list)
        return result_list