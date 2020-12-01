import argparse

from termcolor import colored
import torchvision

CATEGORY_ID = 'category_id'
OK = colored('OK', 'green')
WARNING = colored('WARNING', 'yellow')


def main(args):
    print(OK, 'Read COCO train ...')
    # train_coco_data = torchvision.datasets.CocoDetection(args.coco_train_root_path, args.coco_train_annotations)
    print(OK, 'done')

    print(OK, 'Read COCO val ...')
    val_coco_data = torchvision.datasets.CocoDetection(args.coco_val_root_path, args.coco_val_annotations)
    print(OK, 'done')

    # filter the data s. th. all samples only have __exactly__ one assigned category
    print(OK, 'Filter train ...')
    # train_one_cat = filter_number_of_categories(train_coco_data, 1)
    print(OK, 'done')

    print(OK, 'Filter val ...')
    val_one_cat = filter_number_of_categories(val_coco_data, 1)
    print(OK, 'done')

    print(OK, 'Load Imagenet data')
    imagenet_data = torchvision.datasets.ImageNet(args.imagenet_root_path, split=args.imagenet_split)
    print(OK, 'done')

    e = val_one_cat


    root_path = '/hpi/fs00/share/fg/rabl/strassenburg/datasets/imgnet'
    imagenet_data = torchvision.datasets.ImageNet(root_path, split='val')

    el = imagenet_data[1]

    print(len(val_one_cat))


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
    parser.add_argument('--coco-train-annotations', help='coco path for training annotations; \'instances_train2017.json\'')
    parser.add_argument('--coco-val-root-path', help='coco root path for validation data; \'val2017\'')
    parser.add_argument('--coco-val-annotations', help='coco path for validation annotations; \'instances_val2017.json\'')
    parser.add_argument('--imagenet-root', help='imagenet root path for')
    parser.add_argument('--imagenet-split', default='val', choices=['train', 'val'])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    tmp = args.train_annotations
    main(args)
