import argparse
import json

from termcolor import colored
import torchvision

CATEGORY_ID = 'category_id'
OK = colored('OK', 'green')
WARNING = colored('WARNING', 'yellow')


def main(args):
    # print(OK, 'Read COCO train ...')
    # # train_coco_data = torchvision.datasets.CocoDetection(args.coco_train_root_path, args.coco_train_annotations)
    # print(OK, 'done')
    #
    print(OK, 'Read COCO val ...')
    val_coco_data = torchvision.datasets.CocoDetection(args.coco_val_root_path, args.coco_val_annotations)
    print(OK, 'done')
    #
    # # filter the data s. th. all samples only have __exactly__ one assigned category
    # print(OK, 'Filter train ...')
    # train_one_cat = filter_number_of_categories(train_coco_data, 1)
    # print(OK, 'done')
    # print(OK, 'found {} elements with exactly one category in COCO training data'.format(len(train_one_cat)))
    # #
    print(OK, 'Filter val ...')
    val_one_cat = filter_number_of_categories(val_coco_data, 1)
    print(OK, 'done')
    print(OK, 'found {} elements with exactly one category in COCO validation data'.format(len(val_one_cat)))

    print(OK, 'Load Imagenet data ...')
    imagenet_data = torchvision.datasets.ImageNet(args.imagenet_root, split=args.imagenet_split)
    print(OK, 'done')

    # both indexes for train and val will be the same
    coco_cat_index = id_to_class_index(args.coco_val_annotations)

    print(OK, 'Match data ...')
    val_match = match_classes(val_one_cat, coco_cat_index, imagenet_data.class_to_idx)
    # train_match = match_classes(train_one_cat, imagenet_data.class_to_idx)
    print(OK, 'done')

    # e = val_one_cat

    el = imagenet_data[1]
    vm = val_match[0]


def id_to_class_index(annotations_path):
    index = {}
    with open(annotations_path, 'r') as COCO:
        js = json.loads(COCO.read())
        cats = js['categories']

        for cat in cats:
            index[cat['id']] = cat['name']

    return index

def match_classes(val_one_cat, coco_index, imagenet_index):
    matched = []
    for element in val_one_cat:
        coco_category = get_coco_category(element, coco_index)
        if coco_category and coco_category in imagenet_index.keys():
            matched.append((element, imagenet_index[coco_category]))

    return matched

def get_coco_category(element, cat_index):
    _, annot = element
    cat_ids = category_ids(annot)
    # sometimes no category is set
    assert len(cat_ids) <= 1
    if len(cat_ids) == 1:
        cat = cat_index[cat_ids.pop()]
        return cat


def filter_number_of_categories(elements, num_categories):
    return [e for e in elements if len(category_ids(e[1])) == num_categories]


def category_ids(annotation):
    cat_ids = set()
    for a in annotation:
        cat_ids.add(a[CATEGORY_ID])

    return cat_ids


def parse_args():
    parser = argparse.ArgumentParser(description='Creation of customized COCO dataset')
    parser.add_argument('--coco-train-root-path', help='coco root path for training data; \'train2017\'', )
    parser.add_argument('--coco-train-annotations',
                        help='coco path for training annotations; \'instances_train2017.json\'')
    parser.add_argument('--coco-val-root-path', help='coco root path for validation data; \'val2017\'')
    parser.add_argument('--coco-val-annotations',
                        help='coco path for validation annotations; \'instances_val2017.json\'')
    parser.add_argument('--imagenet-root', help='imagenet root path for')
    parser.add_argument('--imagenet-split', default='val', choices=['train', 'val'])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
