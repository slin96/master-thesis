import argparse
import os

from experiments.evaluation_flow.shared import RESNET_18, RESNET_50, RESNET_152


def parse_args():
    parser = argparse.ArgumentParser(description='Script to measure the time used for training a model')
    parser.add_argument('--coco-root', type=str, help='root directory for the coco data', required=True)
    parser.add_argument('--coco-annotations', type=str, help='path to the coco_meta.json file', required=True)
    parser.add_argument('--visible-devices', type=str, help='string to indicate the visible GPUSs', required=True)

    args = parser.parse_args()

    return args


def main(args):
    visible_devices = args.visible_devices
    coco_root = args.coco_root
    coco_annotations = args.coco_annotations
    num_epochs = 5
    runs = 5

    for i in range(runs):
        for model in [RESNET_18, RESNET_50, RESNET_152]:
            non_determ_out_file = F"non-deterministic-{model}-{i}.txt"
            determ_out_file = F"deterministic-{model}-{i}.txt"

            base_command = F"CUDA_VISIBLE_DEVICES={visible_devices} " \
                           F"python time_training.py " \
                           F"--coco-root {coco_root} " \
                           F"--coco-annotations {coco_annotations} " \
                           F"--model {model} " \
                           F"--num-epochs {num_epochs}"

            non_determ_cmd = F"{base_command} > {non_determ_out_file}"
            determ_cmd = F"{base_command} --deterministic t > {determ_out_file}"

            print(non_determ_cmd)
            os.system(non_determ_cmd)

            print("sleep")
            os.system("sleep 10")

            print(determ_cmd)
            os.system(determ_cmd)


if __name__ == '__main__':
    args = parse_args()
    main(args)
