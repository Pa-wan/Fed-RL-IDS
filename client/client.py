import os

# this is needed to prevent a cudnn error on some GPU's
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# this reduces the amount of tensorflow logging messages to only errors.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import flwr as fl  # noqa: E402



def create_environment():
    print("Creating Environment")


def main():
    print("Starting Client")
    # fl.client.start_numpy_client("[::]:8080", client=PsoClient())


if __name__ == "__main__":
    main()
