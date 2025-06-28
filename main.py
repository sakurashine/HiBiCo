#Copyright (C) 2020 Xiao Wang
#License: MIT for academic use.
#Contact: Xiao Wang (wang3702@purdue.edu, xiaowang20140001@gmail.com)

from ops.argparser import  argparser


def main(args):
    from training.main_worker import main_worker
    main_worker(args.gpu, args)  # 0,1
if __name__ == '__main__':
    parser = argparser()
    args = parser.parse_args()
    main(args)