import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('train_path')
    parser.add_argument('valid_path')
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1126)
    return parser.parse_args()


def load_lines(path):
    with open(path) as infile:
        return [x.strip() for x in infile.readlines()]


def save_lines(path, lines):
    with open(path, 'w') as outfile:
        for line in lines:
            outfile.write(line.strip() + '\n')


def main():
    args = parse_args()

    lines = load_lines(args.input_path)
    random.seed(args.seed)
    random.shuffle(lines)

    valid = lines[:args.size]
    save_lines(args.valid_path, valid)
    train = lines[args.size:]
    save_lines(args.train_path, train)


if __name__ == '__main__':
    main()
