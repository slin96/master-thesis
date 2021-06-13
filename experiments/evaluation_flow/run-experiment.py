import argparse
import os
import pipes
import subprocess
import time


def exists_remote(host, path):
    """Test if a file exists at path on a host accessible with SSH.
    from https://stackoverflow.com/questions/14392432/checking-a-file-existence-on-a-remote-ssh-server-using-python"""
    status = subprocess.call(
        ['ssh', host, 'test -f {}'.format(pipes.quote(path))])
    if status == 0:
        return True
    if status == 1:
        return False
    raise Exception('SSH failed')


def execute_commands(hostname, commands, to_background=False):
    joint_commands = ';'.join(commands)
    cmd = "ssh -t {} '{}'".format(hostname, joint_commands)
    if to_background:
        cmd += ' &'

    os.system(cmd)


def main(args):
    activate_env = "conda activate {}".format(args.env_name)

    set_node_python_path = "export PYTHONPATH=\"{}\"".format(args.node_pythonpath)
    set_server_python_path = "export PYTHONPATH=\"{}\"".format(args.server_pythonpath)

    cd_to_node_script = "cd {}".format(args.node_script_root)
    cd_to_server_script = "cd {}".format(args.server_script_root)

    tmp_dir_arg = "--tmp_dir {}".format("/hpi/fs00/home/nils.strassenburg/tmp-dir")
    server_ip_arg = "--server_ip {}".format('172.20.26.34')
    node_ip_arg = "--node_ip {}".format('172.20.26.35')
    mongo_host_arg = "--mongo_host {}".format('172.20.26.33')
    model_arg = "--model {}".format('mobilenet')
    approach_arg = "--approach {}".format('param_update')
    model_snapshot_args = "--model_snapshots {}".format(
        '/hpi/fs00/share/fg-rabl/strassenburg/version-snapshots/mobilenet-versions-outdoor')
    snapshot_type_arg = "--snapshot_type {}".format('fine-tuned')
    u3_count_arg = "--u3_count {}".format('3')

    node_out_file = "> {}".format('node-remote-test.txt')
    server_out_file = "> {}".format('server-remote-test.txt')

    node_parameters = [tmp_dir_arg, node_ip_arg, server_ip_arg, mongo_host_arg, model_arg, approach_arg,
                       model_snapshot_args, snapshot_type_arg, u3_count_arg, node_out_file]
    run_node_cmd = "python node.py {}".format(" ".join(node_parameters))

    server_parameters = [tmp_dir_arg, server_ip_arg, node_ip_arg, mongo_host_arg, model_arg, approach_arg,
                         model_snapshot_args, snapshot_type_arg, server_out_file]
    run_server_cmd = "python server.py {}".format(" ".join(server_parameters))

    ###############
    # run experiment
    ###############
    # start mongo
    os.system("ssh -t {} '{};{}' &".format('dlab-n04', 'export XDG_DATA_HOME=/scratch/$(id -un)/enroot',
                                           'enroot start mongo'))

    print('wait for mongo ...')
    time.sleep(10)

    # run node
    execute_commands('dlab-n06', [activate_env, set_node_python_path, cd_to_node_script, run_node_cmd],
                     to_background=True)
    # run server
    execute_commands('dlab-n05', [activate_env, set_server_python_path, cd_to_server_script, run_server_cmd],
                     to_background=True)

    done = False
    while not done:
        time.sleep(10)
        done = exists_remote('dlab-n05',
                             '/hpi/fs00/home/nils.strassenburg/evaluation/server/experiments/evaluation_flow/done.txt')
        print('done: {}'.format(done))

    # stop mongo
    os.system('ssh -t {} pkill -f mongo'.format('dlab-n04'))

    print('wait for before starting new experiment')
    time.sleep(100)


# result = exists_remote('dlab-n05',
#                        '/hpi/fs00/home/nils.strassenburg/evaluation/server/experiments/evaluation_flow/done.txt')
# print(result)
#
# # start mongo
# os.system("ssh -t {} '{};{}' &".format('dlab-n04', 'export XDG_DATA_HOME=/scratch/$(id -un)/enroot',
#                                        'enroot start mongo'))
#
# time.sleep(20)
#
# # kill mongoDB
# os.system('ssh -t {} pkill -f mongo'.format('dlab-n04'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host-name', type=str, default='dlab-n06')
    parser.add_argument('--env-name', type=str, default='myenv')
    parser.add_argument('--node-pythonpath', type=str, default='/hpi/fs00/home/nils.strassenburg/evaluation/node/')
    parser.add_argument('--server-pythonpath', type=str, default='/hpi/fs00/home/nils.strassenburg/evaluation/server/')
    parser.add_argument('--node-script-root', type=str,
                        default='/hpi/fs00/home/nils.strassenburg/evaluation/node/experiments/evaluation_flow')
    parser.add_argument('--server-script-root', type=str,
                        default='/hpi/fs00/home/nils.strassenburg/evaluation/server/experiments/evaluation_flow')

    args = parser.parse_args()
    main(args)
