import argparse
import os

# this is needed to prevent a cudnn error on some GPU's
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# this reduces the amount of tensorflow logging messages to only errors.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl  # noqa: E402


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=1,
        help="The number of training rounds to conduct. (1)"
        )
    return parser


strategy = fl.server.strategy.FedAvg(
    min_fit_clients=5, min_available_clients=5, fraction_fit=0.1,
)


def main():
    fl.server.start_server(
        config={"num_rounds": round_count}, strategy=strategy,
    )


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    round_count = args.num_rounds
    main()
