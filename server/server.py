import os

# this is needed to prevent a cudnn error on some GPU's
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# this reduces the amount of tensorflow logging messages to only errors.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl  # noqa: E402


strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2, min_available_clients=2, fraction_fit=0.1,
)


def main():
    fl.server.start_server(
        config={"num_rounds": 10}, strategy=strategy,
    )


if __name__ == "__main__":
    main()
