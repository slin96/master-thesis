import argparse
import os


def main(args):
    # os.system("ssh -t dlab-n06 'conda activate myenv;"
    #           "export PYTHONPATH=\"/hpi/fs00/home/nils.strassenburg/evaluation/node/\";"
    #           "cd /hpi/fs00/home/nils.strassenburg/evaluation/node/experiments/evaluation_flow;"
    #           "python node.py --tmp_dir /hpi/fs00/home/nils.strassenburg/tmp-dir --server_ip 172.20.26.34 --node_ip 172.20.26.35 --mongo_host 172.20.26.33 --model resnet152 --approach param_update --model_snapshots /hpi/fs00/share/fg-rabl/strassenburg/version-snapshots/resnet152-versions-outdoor --snapshot_type fine-tuned --u3_count 3 > remote-test.txt'")

    cmd = ''
    cmd += "ssh -t {} 'conda activate {};".format(args.host_name, args.env_name)
    cmd += "export PYTHONPATH=\"{}\";".format(args.pythonpath)
    cmd += "cd {};".format(args.script_root)
    cmd += "python node.py --tmp_dir /hpi/fs00/home/nils.strassenburg/tmp-dir --server_ip 172.20.26.34 --node_ip 172.20.26.35 --mongo_host 172.20.26.33 --model resnet152 --approach param_update --model_snapshots /hpi/fs00/share/fg-rabl/strassenburg/version-snapshots/resnet152-versions-outdoor --snapshot_type fine-tuned --u3_count 3 > node-remote-test.txt'"
    cmd += ' &'
    os.system(cmd)

    cmd = ''
    cmd += "ssh -t {} 'conda activate {};".format('dlab-n05', args.env_name)
    cmd += "export PYTHONPATH=\"{}\";".format('/hpi/fs00/home/nils.strassenburg/evaluation/server/')
    cmd += "cd {};".format('/hpi/fs00/home/nils.strassenburg/evaluation/server/experiments/evaluation_flow')
    cmd += "python server.py --tmp_dir /hpi/fs00/home/nils.strassenburg/tmp-dir --server_ip 172.20.26.34 --node_ip 172.20.26.35 --mongo_host 172.20.26.33 --model resnet152 --approach param_update --model_snapshots /hpi/fs00/share/fg-rabl/strassenburg/version-snapshots/resnet152-versions-outdoor --snapshot_type fine-tuned > server-remote-test.txt'"
    cmd += ' &'
    os.system(cmd)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host-name', type=str, default='dlab-n06')
    parser.add_argument('--env-name', type=str, default='myenv')
    parser.add_argument('--pythonpath', type=str, default='/hpi/fs00/home/nils.strassenburg/evaluation/node/')
    parser.add_argument('--script-root', type=str,
                        default='/hpi/fs00/home/nils.strassenburg/evaluation/node/experiments/evaluation_flow')

    args = parser.parse_args()
    main(args)
