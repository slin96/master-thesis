def add_shared_args(parser):
    parser.add_argument('--imagenet-root', help='imagenet root path for')
    parser.add_argument('--tmp-output-root', help='dir where tmp output is written to')
    parser.add_argument('--number-batches', type=int,
                        help='the number of batches that should be included in the output')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='the to execute on')


def add_compare_args(parser):
    parser.add_argument('--input-root', help='root dir for model outputs')
    parser.add_argument('--compare-to-root', help='root dir for outputs to compare against')
