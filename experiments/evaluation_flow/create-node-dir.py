import os
import shutil

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
NODE = os.path.join(FILE_PATH, 'node')
EXPERIMENTS = os.path.join(NODE, 'experiments')
BASELINE_FLOW = os.path.join(EXPERIMENTS, 'evaluation_flow')

if __name__ == '__main__':
    if os.path.isdir(NODE):
        shutil.rmtree(NODE)

    os.mkdir(NODE)
    os.mkdir(EXPERIMENTS)
    os.mkdir(BASELINE_FLOW)

    files_to_copy = [
        (os.path.join(FILE_PATH, './node.py'), os.path.join(BASELINE_FLOW, './node.py')),
        (os.path.join(FILE_PATH, './shared.py'), os.path.join(BASELINE_FLOW, './shared.py')),
        (os.path.join(FILE_PATH, './custom_coco.py'), os.path.join(BASELINE_FLOW, './custom_coco.py')),
        (os.path.join(FILE_PATH, './imagenet_train.py'), os.path.join(BASELINE_FLOW, './imagenet_train.py')),
    ]

    dirs_to_copy = [
        (os.path.join(FILE_PATH, '../models'), os.path.join(EXPERIMENTS, 'models'))
    ]

    for s, d in files_to_copy:
        shutil.copyfile(s, d)

    for s, d in dirs_to_copy:
        shutil.copytree(s, d)
