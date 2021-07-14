import os
import shutil

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
SERVER = os.path.join(FILE_PATH, 'server')
EXPERIMENTS = os.path.join(SERVER, 'experiments')
BASELINE_FLOW = os.path.join(EXPERIMENTS, 'evaluation_flow')

if __name__ == '__main__':
    if os.path.isdir(SERVER):
        shutil.rmtree(SERVER)

    os.mkdir(SERVER)
    os.mkdir(EXPERIMENTS)
    os.mkdir(BASELINE_FLOW)

    files_to_copy = [
        (os.path.join(FILE_PATH, './server.py'), os.path.join(BASELINE_FLOW, './server.py')),
        (os.path.join(FILE_PATH, './shared.py'), os.path.join(BASELINE_FLOW, './shared.py')),
        (os.path.join(FILE_PATH, './imagenet_optimizer.py'), os.path.join(BASELINE_FLOW, './imagenet_optimizer.py')),
        (os.path.join(FILE_PATH, './imagenet_train.py'), os.path.join(BASELINE_FLOW, './imagenet_train.py')),
        (os.path.join(FILE_PATH, './create_finetuned.py'), os.path.join(BASELINE_FLOW, './create_finetuned.py')),
        (os.path.join(FILE_PATH, './imagenet_optimizer.py'), os.path.join(BASELINE_FLOW, './imagenet_optimizer.py')),
        (os.path.join(FILE_PATH, './imagenet_train.py'), os.path.join(BASELINE_FLOW, './imagenet_train.py')),
        (os.path.join(FILE_PATH, './imagenet_train_loader.py'), os.path.join(BASELINE_FLOW, './imagenet_train_loader.py')),
        (os.path.join(FILE_PATH, './server-config.ini'), os.path.join(BASELINE_FLOW, './server-config.ini')),
    ]

    dirs_to_copy = [
        (os.path.join(FILE_PATH, '../models'), os.path.join(EXPERIMENTS, 'models'))
    ]

    for s, d in files_to_copy:
        shutil.copyfile(s, d)

    for s, d in dirs_to_copy:
        shutil.copytree(s, d)
