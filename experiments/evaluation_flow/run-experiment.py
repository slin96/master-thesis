import argparse
import os
import pipes
import subprocess
import time

from experiments.evaluation_flow.shared import add_all_parameters


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
    ####################################################
    # parameters that stay fixed for all experiments
    ####################################################
    activate_env = "conda activate {}".format(args.env_name)

    set_node_python_path = "export PYTHONPATH=\"{}\"".format(args.node_pythonpath)
    set_server_python_path = "export PYTHONPATH=\"{}\"".format(args.server_pythonpath)

    cd_to_node_script = "cd {}".format(args.node_script_root)
    cd_to_server_script = "cd {}".format(args.server_script_root)

    tmp_dir_arg = "--tmp_dir {}".format(args.tmp_dir)
    mongo_host_arg = "--mongo_host {}".format(args.mongo_host)
    server_ip_arg = "--server_ip {}".format(args.server_ip)
    node_ip_arg = "--node_ip {}".format(args.node_ip)

    u3_count_arg = "--u3_count {}".format(args.u3_count)

    ####################################################
    # parameters that change
    ####################################################
    model_arg = "--model {}".format(args.model)
    approach_arg = "--approach {}".format(args.approach)
    model_snapshot_args = "--model_snapshots {}".format(args.model_snapshots)
    snapshot_type_arg = "--snapshot_type {}".format(args.snapshot_type)

    node_log_name = 'node-remote-test.txt'
    node_log = os.path.join(args.log_dir, node_log_name)
    node_out_file = "> {}".format(str(node_log))

    server_log_name = 'server-remote-test.txt'
    server_log = os.path.join(args.log_dir, server_log_name)
    server_out_file = "> {}".format(server_log)

    ####################################################
    # put commands together
    ####################################################
    node_parameters = [tmp_dir_arg, node_ip_arg, server_ip_arg, mongo_host_arg, model_arg, approach_arg,
                       model_snapshot_args, snapshot_type_arg, u3_count_arg, node_out_file]
    run_node_cmd = "python node.py {}".format(" ".join(node_parameters))

    server_parameters = [tmp_dir_arg, server_ip_arg, node_ip_arg, mongo_host_arg, model_arg, approach_arg,
                         model_snapshot_args, snapshot_type_arg, server_out_file]
    run_server_cmd = "python server.py {}".format(" ".join(server_parameters))

    ####################################################
    # run experiment
    ####################################################
    # start mongo
    os.system("ssh -t {} '{};{}' &".format(args.mongo_host_name, 'export XDG_DATA_HOME=/scratch/$(id -un)/enroot',
                                           'enroot start mongo'))
    # wait for mongoDB to be ready
    print('wait for mongo ...')
    time.sleep(10)

    # run node
    execute_commands(args.node_host_name, [activate_env, set_node_python_path, cd_to_node_script, run_node_cmd],
                     to_background=True)
    # run server
    execute_commands(args.server_host_name, [activate_env, set_server_python_path, cd_to_server_script, run_server_cmd],
                     to_background=True)

    # check if the experiment is done
    done = False
    while not done:
        time.sleep(10)
        done = exists_remote(args.server_host_name,
                             '/hpi/fs00/home/nils.strassenburg/evaluation/server/experiments/evaluation_flow/done.txt')
        print('done: {}'.format(done))

    # when done stop mongo and clean up tmp directory
    os.system('ssh -t {} pkill -f mongo'.format(args.mongo_host_name))
    os.system("ssh -t {} '{}; rm -rf *'".format(args.mongo_host_name, args.tmp_dir))

    print('wait for before starting new experiment')
    time.sleep(100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_all_parameters(parser)
    parser.add_argument('--mongo_host_name', type=str, default='dlab-n04')
    parser.add_argument('--server_host_name', type=str, default='dlab-n05')
    parser.add_argument('--node_host_name', type=str, default='dlab-n06')
    parser.add_argument('--env_name', type=str, default='myenv')
    parser.add_argument('--node_pythonpath', type=str, default='/hpi/fs00/home/nils.strassenburg/evaluation/node/')
    parser.add_argument('--server_pythonpath', type=str, default='/hpi/fs00/home/nils.strassenburg/evaluation/server/')
    parser.add_argument('--node_script_root', type=str,
                        default='/hpi/fs00/home/nils.strassenburg/evaluation/node/experiments/evaluation_flow')
    parser.add_argument('--server_script_root', type=str,
                        default='/hpi/fs00/home/nils.strassenburg/evaluation/server/experiments/evaluation_flow')

    args = parser.parse_args()
    main(args)
