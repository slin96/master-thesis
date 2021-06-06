import os
import shutil

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
ADMIN = os.path.join(FILE_PATH, 'admin')
EXPERIMENTS = os.path.join(ADMIN, 'experiments')
BASELINE_FLOW = os.path.join(EXPERIMENTS, 'baseline_flow')

if __name__ == '__main__':
    if os.path.isdir(ADMIN):
        shutil.rmtree(ADMIN)

    os.mkdir(ADMIN)
    os.mkdir(EXPERIMENTS)
    os.mkdir(BASELINE_FLOW)

    files_to_copy = [
        (os.path.join(FILE_PATH, './admin.py'), os.path.join(BASELINE_FLOW, './admin.py')),
        (os.path.join(FILE_PATH, './shared.py'), os.path.join(BASELINE_FLOW, './shared.py'))
    ]

    for s, d in files_to_copy:
        shutil.copyfile(s, d)
