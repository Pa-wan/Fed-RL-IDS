import os

import flwr as fl
from client.pso_flower_client import pso_dqn_flower_client
# this is needed to prevent a cudnn error on some GPU's
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# this reduces the amount of tensorflow logging messages to only errors.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from agents.agent_initialiser import AgentInitialiser  # noqa: E402
from agents.ddqn_agent import ddqn_agent   # noqa: E402
from environ.nsl_kdd import nsl_kdd_env  # noqa: E402


# import flwr as fl  # noqa: E402

GLOBAL_REWARD = 1
CLIENT_ID = 1


def create_ddqn_initiliser(env):
    initialiser = AgentInitialiser(
            alpha=0.005,
            type_name="ddqn",
            n_actions=2,
            gamma=0.99,
            epsilon=1.0,
            epsilon_dec=0.995,
            epsilon_min=0.01,
            model_file="models/ddqn_model.h5",
            replace_target=1000,
            batch_size=32,
            memory_size=10000,
            input_dims=[env.get_observation_space()],
            fc1_dims=128,
            fc2_dims=128,
            learning_rate=0.001)
    return initialiser


def main():
    print("Starting Client")
    env = nsl_kdd_env(GLOBAL_REWARD, CLIENT_ID)
    env.setup()
    print([env.get_observation_space()])
    initialiser = create_ddqn_initiliser(env)
    agent = ddqn_agent(initialiser)

    fl.client.start_numpy_client(
        "[::]:8080",
        client=pso_dqn_flower_client(agent, env)
        )


if __name__ == "__main__":
    main()
