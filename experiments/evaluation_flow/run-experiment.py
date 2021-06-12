import os

if __name__ == '__main__':
    os.system("ssh -t dlab-n06 'conda activate myenv;"
              "export PYTHONPATH=\"/hpi/fs00/home/nils.strassenburg/evaluation/node/\";"
              "cd /hpi/fs00/home/nils.strassenburg/evaluation/node/experiments/evaluation_flow;"
              "python node.py --tmp_dir /hpi/fs00/home/nils.strassenburg/tmp-dir --server_ip 172.20.26.34 --node_ip 172.20.26.35 --mongo_host 172.20.26.33 --model resnet152 --approach param_update --model_snapshots /hpi/fs00/share/fg-rabl/strassenburg/version-snapshots/resnet152-versions-outdoor --snapshot_type fine-tuned --u3_count 3 > remote-test.txt'")
