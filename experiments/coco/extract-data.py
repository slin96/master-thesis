import argparse

def main(args):
    print(args)


def parse_args():
    parser = argparse.ArgumentParser(description='Creation of customized COCO dataset')
    parser.add_argument('--train-root-path', help='root path for training data; \'train2017\'', )
    parser.add_argument('--train-annotations', help='path for training annotations; \'instances_train2017.json\'')
    parser.add_argument('--val-root-path', help='root path for validation data; \'val2017\'')
    parser.add_argument('--val-annotations', help='path for validation annotations; \'instances_val2017.json\'')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    tmp = args.train_annotations
    main(args)
