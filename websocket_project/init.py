import argparse
import sys
import torch.nn.functional as F

n_workers=3
base_port=8777

class Arguments():
    def __init__(self, in_size, out_size, hidden_layers,
                       activation=F.softmax, dim=-1):
        self.batch_size = 5
        self.drop_p = None
        self.epochs = 10
        self.lr = 0.001
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_layers = hidden_layers
        self.precision_fractional=3
        self.activation = activation
        self.dim = dim

model_args = Arguments(42, 2, [30,15,5])
        
def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of the training")
    parser.add_argument(
        "--test_batch_size", type=int, default=5, help="batch size used for the test data"
    )
    parser.add_argument(
        "--training_rounds", type=int, default=40, help="number of federated learning rounds"
    )
    parser.add_argument(
        "--federate_after_n_batches", type=int, default=10,
        help = "number of training steps performed on each remote worker before averaging",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--seed", type=int, default=1, help="seed used for randomization")
    parser.add_argument("--save_model", action="store_true", help="if set, model will be saved")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="if set, websocket client workers will be started in verbose mode",
    )

    args = parser.parse_args(args=args)
    return args