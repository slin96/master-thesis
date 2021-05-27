import os
import shutil

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
NODE = os.path.join(FILE_PATH, 'node')
EXPERIMENTS = os.path.join(NODE, 'experiments')
BASELINE_FLOW = os.path.join(EXPERIMENTS, 'baseline_flow')

if __name__ == '__main__':
    os.mkdir(NODE)
    os.mkdir(EXPERIMENTS)
    os.mkdir(BASELINE_FLOW)

    files_to_copy = [
        (os.path.join(FILE_PATH, './node.py'), os.path.join(BASELINE_FLOW, './node.py')),
        (os.path.join(FILE_PATH, './shared.py'), os.path.join(BASELINE_FLOW, './node.py'))
    ]

    for s, d in files_to_copy:
        shutil.copyfile(s, d)
