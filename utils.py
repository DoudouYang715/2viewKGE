import pickle
import dgl
import numpy as np
import torch
import random
import os
import logging
from collections import defaultdict as ddict


def get_g(tri_list):
    triples = np.array(tri_list)
    g = dgl.graph((triples[:, 0].T, triples[:, 2].T))
    g.edata['rel'] = torch.tensor(triples[:, 1].T)
    return g

def get_hr2t_rt2h_sup_que(sup_tris, que_tris):
    hr2t = ddict(list)
    rt2h = ddict(list)
    for tri in sup_tris:
        h, r, t = tri
        hr2t[(h, r)].append(t)
        rt2h[(r, t)].append(h)

    for tri in que_tris:
        h, r, t = tri
        hr2t[(h, r)].append(t)
        rt2h[(r, t)].append(h)

    que_hr2t = dict()
    que_rt2h = dict()
    for tri in que_tris:
        h, r, t = tri
        que_hr2t[(h, r)] = hr2t[(h, r)]
        que_rt2h[(r, t)] = rt2h[(r, t)]

    return que_hr2t, que_rt2h


def serialize(data):
    return pickle.dumps(data)


def deserialize(data):
    data_tuple = pickle.loads(data)
    return data_tuple

def init_dir(args):
    # state
    if not os.path.exists(args.state_dir):
        os.makedirs(args.state_dir)

    # tensorboard log
    if not os.path.exists(args.tb_log_dir):
        os.makedirs(args.tb_log_dir)

    # logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

def set_seed(seed):
    dgl.seed(seed)
    dgl.random.seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


