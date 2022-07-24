import models
import datasets
from modules import arguments, logger, monitor, controller
from modules.communicate import PipeCommunicator
from models import tf
from models import toNumpy
import os
import sys
tf.config.experimental_run_functions_eagerly(True)

parser = arguments.create_parser()
parser.add_argument("--debug", action="store_true", help="Debug in VS Code")
parser.add_argument("--random_seed", type=int, default=123)
parser.add_argument("--interactive", "-i", action="store_true", dest="_interactive")
parser.add_argument("--outpipe", required=True, type=int, dest="_outpipe")
parser.add_argument("--inpipe", default=0, type=int, dest="_inpipe", help="Default to stdin")
known_args, _ = parser.parse_known_args()
if known_args.debug:
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
    ptvsd.wait_for_attach()
    breakpoint()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

if known_args.random_seed:
    tf.random.set_seed(known_args.random_seed)

subparser = parser.add_argument_group(
    "Experiment arguments (run_experiments.py)")
subparser.add_argument("--epochs", type=int, default=2000,
                       help="(default: %(default)s)")

models.add_subparsers(parser)
datasets.add_subparsers(parser)
logger.add_subparser_args(parser)
monitor.add_subparser_args(parser)

def open_pipes(args):
    outpipe = os.fdopen(args._outpipe, 'wb')
    if args._inpipe != 0:
        inpipe = os.fdopen(args._inpipe, 'rb')
    else:
        inpipe = sys.stdin.buffer
    args.objects["inpipe"] = inpipe
    args.objects["outpipe"] = outpipe
    args.objects["communicator"] = PipeCommunicator(inpipe, outpipe)
    print("Child process - setting up pipes")
parser.function_hooks["argparse"].appendleft(open_pipes)

# Prepare configs and data
args = arguments.parse_args(parser)
inpipe = args.objects["inpipe"]
outpipe = args.objects["outpipe"]

# Data preprocessing
for func in args.objects["data_preprocess_callbacks"]:
    func(args)

for func in args.objects["pretrain_callbacks"]:
    func(**args.objects["tensors"])

communicator = args.objects["communicator"]
communicator.signalReady()

try:
    while True:
        command = communicator.recvObject()
        if command == "set_params":
            confDict = communicator.recvObject() #type: dict
            print(f"Set params: {confDict}")
            for key, val in confDict.items():
                if key == "early_stopping":
                    args.objects["early_stopping"] = controller.SlidingMeanEarlyStopping(val)
                else:
                    setattr(args, key, val)
            communicator.signalReady()
        elif command == "update_dataset":
            dataDict = communicator.recvObject() #type: dict
            dataset = args.objects["dataset"]
            dataset.reload_data()
            for key, val in dataDict.items():
                if key == "adj":
                    key = "sparse_adj"
                assert hasattr(dataset, key)
                setattr(dataset, key, val)
            for func in args.objects["data_preprocess_callbacks"]:
                func(args)
            communicator.signalReady()
            print(f"Dataset updated with new {dataDict.keys()}!")
        elif command == "train":
            args.current_epoch = 0
            while args.current_epoch < args.epochs:
                args.current_epoch += 1
                for func in args.objects["pre_epoch_callbacks"]:
                    func(args.current_epoch, args)
                args.objects["epoch_stats"] = dict()
                args.objects["epoch_stats"].update(
                    args.objects["train_step"](**args.objects["tensors"]))
                args.objects["epoch_stats"].update(
                    args.objects["test_step"](**args.objects["tensors"]))
                for func in args.objects["post_epoch_callbacks"]:
                    func(args.current_epoch, args)
                while args.current_epoch >= args.epochs and len(args.objects["post_train_callbacks"]) > 0:
                    func = args.objects["post_train_callbacks"].popleft()
                    func(args)
            communicator.signalReady()
        elif command == "predict":
            predictions = toNumpy(args.objects["predict_step"](**args.objects["tensors"]))
            communicator.sendObject(predictions)
            print(f"New prediction ready!")
        elif command == "debug":
            import ptvsd
            print("Waiting for debugger attach")
            ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
            ptvsd.wait_for_attach()
            breakpoint()
            communicator.signalReady()
        elif command == "exit":
            print("Exiting...")
            break
        else:
            raise ValueError()
except:
    communicator.signalFail() # Prevent parent process for signal waiting
    raise

if args._interactive:
    import IPython
    IPython.embed()
