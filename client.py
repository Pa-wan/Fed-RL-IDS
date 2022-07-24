import os
import argparse

# this is needed to prevent a cudnn error on some GPU's
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# this reduces the amount of tensorflow logging messages to only errors.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import flwr as fl  # noqa E402

from client.ddqn_flower_client import dqn_flower_client  # noqa E402
from client.pso_flower_client import pso_dqn_flower_client  # noqa E402
from agents.agent_initialiser import AgentInitialiser  # noqa: E402
from agents.ddqn_agent import ddqn_agent   # noqa: E402
from environ.nsl_kdd import nsl_kdd_env  # noqa: E402


GLOBAL_REWARD = 10


def create_ddqn_initiliser(env):
    initialiser = AgentInitialiser(
            alpha=0.005,
            type_name="ddqn",
            n_actions=2,
            gamma=0.99,
            epsilon=1.0,
            epsilon_dec=0.995,
            epsilon_min=0.01,
            model_file="history/models/ddqn_model.h5",
            replace_target=1000,
            batch_size=32,
            memory_size=10000,
            input_dims=[env.get_observation_space()],
            fc1_dims=128,
            fc2_dims=128,
            learning_rate=0.001)
    return initialiser


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--child_id",
        type=int,
        default=1,
        help="The id of this client (1)"
        )

    parser.add_argument(
        "--agent_type",
        type=str,
        default="ddqn",
        help="The type of agent to use (ddqn, duelDQN)"
        )

    parser.add_argument(
        "--flower_client_type",
        type=str,
        default="base",
        help="The type of flower client to use (base, pso)"
        )

    return parser


def create_enviroment():
    env = nsl_kdd_env(GLOBAL_REWARD, child_id)
    env.setup()
    return env


def get_client_type(env, agent):
    if flower_client_type == "base":
        client = dqn_flower_client(agent, env, child_id)
    elif flower_client_type == "pso":
        client = pso_dqn_flower_client(agent, env, child_id)
    return client


def get_agent_type(initialiser):
    if agent_type == "ddqn":
        agent = ddqn_agent(initialiser)
    elif agent_type == "duelDQN":
        raise NotImplementedError("duelDQN is not implemented yet")
    elif agent_type == "pso_ddqn":
        raise NotImplementedError("pso_ddqn is not implemented yet")
    return agent


def main():
    print("Starting Client")
    env = create_enviroment()

    initialiser = create_ddqn_initiliser(env)
    agent = get_agent_type(initialiser)
    client = get_client_type(env, agent)
    # starts the actual learning process.
    fl.client.start_numpy_client("[::]:8080", client=client)
    client.plot_results()

    try:
        client.save_scores()
    except Exception as e:
        print("Write error ", e)


if __name__ == "__main__":

    parser = create_arg_parser()
    args = parser.parse_args()
    # parse the arguments
    child_id = args.child_id
    agent_type = args.agent_type
    flower_client_type = args.flower_client_type

    main()
