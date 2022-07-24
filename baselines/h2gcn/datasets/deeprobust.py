from . import *
import tensorflow as tf
from tensorflow import keras
from ._dataset import DeepRobustData, TransformAdj, TransformSPAdj, sp
from deeprobust.graph.data import Dataset
import pickle


def add_subparser_args(parser):
    subparser = parser.add_argument_group(
        "DeepRobust Format Data Arguments (datasets/deeprobust.py)")
    subparser.add_argument('--dataset', type=str, default='citeseer',
                           choices=['cora', 'cora_ml',
                                    'citeseer', 'polblogs', 'pubmed'],
                           help='dataset', dest="dataset")
    parser.function_hooks["argparse"].appendleft(argparse_callback)


def argparse_callback(args):
    if "communicator" in args.objects:
        communicator = args.objects["communicator"]
        print("Reading deeprobust dataset from pipe...")
        dr_dataset = communicator.recvObject()
    else:
        dr_dataset = Dataset(root='/tmp/', name=args.dataset)
    dataset = DeepRobustData(dr_dataset)
    args.objects["dataset"] = dataset
    print(f"===> Dataset loaded: {args.dataset}")
