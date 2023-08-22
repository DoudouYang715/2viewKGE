import argparse
import os
from utils import init_dir, set_seed
from subgraph import gen_subgraph_datasets
from pre_process import data2pkl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='nell_v1')

    parser.add_argument('--state_dir', '-state_dir', default='./state', type=str)
    parser.add_argument('--log_dir', '-log_dir', default='./log', type=str)
    parser.add_argument('--tb_log_dir', '-tb_log_dir', default='./tb_log', type=str)

    args = parser.parse_args()
    init_dir(args)

    # specify the paths for original data and subgraph db
    args.data_path = f'./data/{args.data_name}.pkl'
    args.db_path = f'./data/{args.data_name}_subgraph'

    # load original data and make index
    if not os.path.exists(args.data_path):
        data2pkl(args.data_name)

    if not os.path.exists(args.db_path):
        gen_subgraph_datasets(args)


