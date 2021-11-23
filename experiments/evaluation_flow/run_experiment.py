import argparse
import os
import pipes
import subprocess

from experiments.evaluation_flow.shared import *


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


MODELS = [MOBILENET, GOOGLENET, RESNET_18, RESNET_50, RESNET_152]

APPROACHES = [BASELINE, PARAM_UPDATE, PARAM_UPDATE_IMPROVED, PROVENANCE]

SNAPSHOT_TYPES = [VERSION, FINE_TUNED]

SNAPSHOT_DIST = ['outdoor', 'food']

REPEAT = 5


def write_completed_log(_str, completed):
    if completed:
        write_string = 'COMPLETED---{}\n'.format(_str)
    else:
        write_string = 'ABORT---{}\n'.format(_str)

    with open('completed-log.txt', 'a') as f:
        f.write(write_string)


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

    done_path = '{}/done.txt'.format(args.server_script_root)

    server_train_data_root = args.server_training_data_path
    node_train_data_root = args.node_training_data_path

    snapshot_root = args.snapshot_root
    log_dir = args.log_dir

    wait_counter_max = args.wait_counter_max

    for r in range(REPEAT):
        run = r + args.run_offset
        for model in MODELS:
            for approach in APPROACHES:
                for snapshot_type in SNAPSHOT_TYPES:
                    for snapshot_dist in SNAPSHOT_DIST:
                        wait_counter = 0

                        run_name = 'model:{}--approach:{}--snapshot_type:{}--snapshot_dist:{}--run:{}' \
                            .format(model, approach, snapshot_type, snapshot_dist, run)
                        print('START RUN: {}'.format(run_name))

                        model_arg = "--model {}".format(model)
                        approach_arg = "--approach {}".format(approach)

                        snapshot_name = '{}-versions-{}'.format(model, snapshot_dist)
                        snapshot_path = os.path.join(snapshot_root, snapshot_name)
                        model_snapshot_args = "--model_snapshots {}".format(snapshot_path)
                        snapshot_type_arg = "--snapshot_type {}".format(snapshot_type)

                        node_log_name = 'node--{}.txt'.format(run_name)
                        node_log = os.path.join(log_dir, node_log_name)
                        node_out_file = "> {}".format(str(node_log))

                        server_log_name = 'server--{}.txt'.format(run_name)
                        server_log = os.path.join(log_dir, server_log_name)
                        server_out_file = "> {}".format(server_log)

                        node_train_data_name = '{}-512'.format(snapshot_dist)
                        node_training_data_path = os.path.join(node_train_data_root, node_train_data_name)
                        node_training_data_path_arg = "--training_data_path {}".format(node_training_data_path)

                        server_training_data_path_arg = "--training_data_path {}".format(server_train_data_root)

                        node_config = args.node_config
                        node_config_arg = "--config {}".format(node_config)

                        server_config = args.server_config
                        server_config_arg = "--config {}".format(server_config)

                        ####################################################
                        # put commands together
                        ####################################################
                        node_parameters = [tmp_dir_arg, node_ip_arg, server_ip_arg, mongo_host_arg, model_arg,
                                           approach_arg, model_snapshot_args, snapshot_type_arg, u3_count_arg,
                                           node_out_file]

                        if approach == PROVENANCE:
                            node_parameters.append(node_training_data_path_arg)
                            node_parameters.append(node_config_arg)

                        run_node_cmd = "python node.py {}".format(" ".join(node_parameters))

                        server_parameters = [tmp_dir_arg, server_ip_arg, node_ip_arg, mongo_host_arg, model_arg,
                                             approach_arg, model_snapshot_args, snapshot_type_arg, server_out_file]

                        if approach == PROVENANCE:
                            server_parameters.append(server_training_data_path_arg)
                            server_parameters.append(server_config_arg)

                        run_server_cmd = "python server.py {}".format(" ".join(server_parameters))

                        ####################################################
                        # run experiment
                        ####################################################
                        # start mongo
                        os.system("ssh -t {} '{};{}' &".format(args.mongo_host_name,
                                                               'export XDG_DATA_HOME=/scratch/$(id -un)/enroot',
                                                               'enroot start mongo'))
                        # wait for mongoDB to be ready
                        print('wait for mongo ...')
                        time.sleep(10)

                        # run node
                        execute_commands(args.node_host_name,
                                         [activate_env, set_node_python_path, cd_to_node_script, run_node_cmd],
                                         to_background=True)
                        # run server
                        execute_commands(args.server_host_name,
                                         [activate_env, set_server_python_path, cd_to_server_script, run_server_cmd],
                                         to_background=True)

                        # check if the experiment is done
                        done = False
                        while not done:
                            time.sleep(10)
                            done = exists_remote(args.server_host_name, done_path)
                            print('done: {}, wait_counter: {}'.format(done, wait_counter))
                            wait_counter += 1

                            # if for some reason something got stuck
                            if wait_counter > wait_counter_max:
                                write_completed_log(run_name, completed=False)
                                done = True

                        # when done, stop all processes and clean up tmp directory
                        os.system('ssh -t {} pkill -f mongo'.format(args.mongo_host_name))
                        os.system('ssh -t {} pkill -f server.py'.format(args.server_host_name))
                        os.system('ssh -t {} pkill -f node.py'.format(args.node_host_name))

                        # the pattern *-*-*-*-* should cover all uuids
                        assert not args.tmp_dir == '' and not args.tmp_dir == '.'
                        os.system("ssh -t {} 'cd {}; rm -rf *-*-*-*-*'".format(args.mongo_host_name, args.tmp_dir))
                        print('wait before starting new experiment')
                        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_evaluation_parameters(parser)
    parser.add_argument('--mongo_host_name', type=str, required=True,
                        help='The hostname or IP address of the machine that should model the database.')
    parser.add_argument('--server_host_name', type=str,  required=True,
                        help='The hostname or IP address of the machine that should model the server.')
    parser.add_argument('--node_host_name', type=str, required=True,
                        help='The hostname or IP address of the machine that should model the node.')
    parser.add_argument('--env_name', type=str, required=True,
                        help='The name of the conda environment')
    parser.add_argument('--node_pythonpath', type=str, required=True,
                        help='The PYTHONPATH for the node ending with /evaluation/server/')
    parser.add_argument('--server_pythonpath', type=str, required=True,
                        help='The PYTHONPATH for the server ending with /evaluation/server/')
    parser.add_argument('--node_script_root', type=str, required=True,
                        help='The path ending with "server/experiments/evaluation_flow" '
                             'to specify where to find the node.py script')
    parser.add_argument('--server_script_root', type=str, required=True,
                        help='The path ending with "server/experiments/evaluation_flow" '
                             'to specify where to find the server.py script')
    parser.add_argument('--snapshot_root', type=str, required=True,
                        help='The root directory to load the pretrained model snapshots from.')
    parser.add_argument('--server_training_data_path', type=str, required=True,
                        help='The path to the training data used by the server, only relevant for provenance approach.')
    parser.add_argument('--node_training_data_path', type=str, required=True,
                        help='The path to the training data used by the node, only relevant for provenance approach.')
    parser.add_argument('--run_offset', type=int, default=0)
    parser.add_argument('--wait_counter_max', type=int, default=30)

    args = parser.parse_args()
    main(args)