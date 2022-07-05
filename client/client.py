import os
from agents.agent_initialiser import AgentInitialiser
from agents.ddqn_agent import ddqn_agent

from environ.nsl_kdd import nsl_kdd_env

# this is needed to prevent a cudnn error on some GPU's
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# this reduces the amount of tensorflow logging messages to only errors.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import flwr as fl  # noqa: E402

GLOBAL_REWARD = 1
CLIENT_ID = 1


def create_ddqn_initiliser(env):
    initialiser = AgentInitialiser()
    initialiser.type_name = "ddqn"
    initialiser.action_space = env.action_space
    initialiser.n_actions = env.n_actions
    initialiser.gamma = 0.99
    initialiser.epsilon = 1.0
    initialiser.epsilon_dec = 0.995
    initialiser.epsilon_min = 0.01
    initialiser.model_file = "models/ddqn_model.h5"
    initialiser.replace_target = 1000
    initialiser.batch_size = 32
    return initialiser


def main():
    print("Starting Client")
    env = nsl_kdd_env(GLOBAL_REWARD, CLIENT_ID)
    initialiser = create_ddqn_initiliser(env)
    agent = ddqn_agent(initialiser)
    

    # fl.client.start_numpy_client("[::]:8080", client=PsoClient())


if __name__ == "__main__":
    main()
