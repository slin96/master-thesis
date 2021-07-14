import time

import torch
from torch.utils.collect_env import get_pretty_env_info

if __name__ == '__main__':
    for i in range(100):
        start = time.time_ns()
        torch.utils.collect_env.get_env_info()
        stop = time.time_ns()
        elapsed = (stop - start) * 10 ** -9
        print('time_elapsed: {}'.format(elapsed))
        time.sleep(2)

