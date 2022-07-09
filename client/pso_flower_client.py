import flwr as fl
import numpy as np
from pyswarms.single import GlobalBestPSO
from agents.agent_initialiser import AgentInitialiser

from agents.ddqn_agent import ddqn_agent
from environ.nsl_kdd import nsl_kdd_env


class pso_dqn_flower_client(fl.client.NumPyClient):

    def __init__(self, agent: ddqn_agent, env: nsl_kdd_env):
        """
            Initialises the flower client

            ParametersL
            agent (RL agent): The agent to use for the client
            env: (Environment): The environment to use for the client
        """
        super(pso_dqn_flower_client, self).__init__()
        self.agent = agent
        self.env = env

    def get_parameters(self):
        """
            Returns the parameters of the agent
        """
        return self.agent.get_weights()

    def fit(self, paramerters, config):
        """
            Trains the agent
        """
        self.agent.set_weights(paramerters)
        self.optimise_agents()
        return self.agent.get_weights(), self.env.get_total_record_count(), {}

    def evaluate(self, parameters, config):
        """
            Evaluates the agent's neural network
        """
        self.agent.set_weights(parameters)
        loss, accuracy = self.evaluate_nn()
        return loss, self.env.get_total_record_count(), {"accuracy": accuracy}

    def optimise_agents(self):
        x_max = 1.0 - np.ones(3)
        x_min = -1.0 * x_max
        bounds = (x_min, x_max)
        options = {'c1': 0.4, 'c2': 0.6, 'w': 0.4}

        optimizer = GlobalBestPSO(
            n_particles=10,
            dimensions=3,
            options=options,
            bounds=bounds
            )
        _, pos = optimizer.optimize(self.evaluate_agent, 20)
        self.agent = ddqn_agent(self.create_initiliser_for_weights(pos))
        self.learn(self.agent, 1)

    def evaluate_agent(self, paramaters):
        """
            Evaluates the agent
        """
        results = []
        for weights in paramaters:
            init = self.create_initiliser_for_weights(weights)
            local_agent = ddqn_agent(init)
            score = self.learn(local_agent, 1)
            results.append(score)
        return results

    def learn(self, _agent, epochs, final=False):
        """
            Trains the agent
        """
        for epoch in range(epochs):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                action = _agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                _agent.remember(state, action, reward, next_state, done)
                state = next_state
                score += reward
            if final:
                print("Epoch: {} Score: {}".format(epoch, score))
        return score

    def evaluate_nn(self):
        """
            Evaluates the agent's neural network
        """
        self.agent.epsilon = self.agent.epsilon_min
        score = 0
        done = False
        state = self.env.reset()
        while not done:
            action = self.agent.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            score += reward

        loss = score / self.env.get_total_record_count()
        accuracy = score
        return loss, accuracy

    def create_initiliser_for_weights(self, weights):
        """
            Creates an initialiser for the agent
        """
        init = AgentInitialiser(
                input_dims=[self.env.get_observation_space()],
                alpha=weights[0],
                gamma=weights[1],
                epsilon=weights[2]
                )
        return init
