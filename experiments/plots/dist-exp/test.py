# %%

from experiments.plots.util import *

# %%
NODES = 5
FILTER_PROV = False
PROV_ONLY = True

assert not (FILTER_PROV and PROV_ONLY)

if NODES == 1:
    ROOT_LOG_DIR = f'/Users/nils/downloads/normal'
else:
    ROOT_LOG_DIR = f'/Users/nils/downloads/dist-{NODES}'

VERSION = 'version'
FINE_TUNED = 'fine-tuned'
OUTDOOR = 'outdoor'
FOOD = 'food'


def get_times():
    # %%
    # get all file sin the directory
    all_files = all_files_in_dir(ROOT_LOG_DIR)
    if FILTER_PROV:
        all_files = [f for f in all_files if not 'provenance' in f]
    elif PROV_ONLY:
        all_files = [f for f in all_files if 'provenance' in f]
    node_server_files = [f for f in all_files if 'server' in f or 'node' in f]
    # for all files extract the metadata (e.g. what model and dataset is used)
    U_IDS = 'u_ids'
    files_and_meta = [(extract_file_meta(f), f) for f in node_server_files]
    # add a mapping: use_case -> model id
    files_and_meta = [({**f[0], **{U_IDS: use_case_ids(f[1])}}, f[1]) for f in files_and_meta]
    # add the parsed events
    files_and_meta = [({**f[0], **{EVENTS: parse_events(f[1])}}, f[1]) for f in files_and_meta]
    # %%
    valid_joined = join_server_and_node_meta(files_and_meta)
    # %%
    times = extract_times(valid_joined, num_nodes=NODES, high_level_only=True)

    return times


times = get_times()

# %%

print('test')
