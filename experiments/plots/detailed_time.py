import time

import torch
from mmlib.persistence import FileSystemPersistenceService
from mmlib.schema.file_reference import FileReference
from torchvision.models import mobilenet_v2

DUMMY_STATE_DICT_PT = './dummy-state-dict.pt'

if __name__ == '__main__':

    model = mobilenet_v2(pretrained=True)
    torch.save(model.state_dict(), DUMMY_STATE_DICT_PT)

    fs = FileSystemPersistenceService('/Users/nils/Desktop/tmp/master-thesis/experiments/plots/dummy-tmp-dir')
    for i in range(20):
        start = time.time_ns()
        fr = FileReference(path=DUMMY_STATE_DICT_PT)
        fs.save_file(fr)
        fs.save_file(fr)
        stop = time.time_ns()
        time_diff = (stop - start) * 10 ** -9
        print('elapsed: {}s'.format(time_diff))
        time.sleep(0.2)
