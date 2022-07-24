import flwr as fl
import numpy as np
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.plotters import plot_contour
from agents.agent_initialiser import AgentInitialiser

from agents.ddqn_agent import ddqn_agent
from environ.nsl_kdd import nsl_kdd_env
from helpers.file_helpers import save_array_to_file
from helpers.plotter import plot


class pso_dqn_flower_client(fl.client.NumPyClient):

    def __init__(self, agent: ddqn_agent, env: nsl_kdd_env, agent_id: int):
        """
            Initialises the flower client

            ParametersL
            agent (RL agent): The agent to use for the client
            env: (Environment): The environment to use for the client
        """
        super(pso_dqn_flower_client, self).__init__()
        self.agent = agent
        self.env = env
        self.agent_id = agent_id
        self.total_scores = []

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
        _, pos = optimizer.optimize(self.evaluate_agent, 5)
        self.plot_position_history(optimizer)
        self.agent = ddqn_agent(self.create_initiliser_for_weights(pos))
        # train the final agent
        self.learn(self.agent, 1, True)

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
                self.total_scores.append(score)
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
        print(weights)
        init = AgentInitialiser(
                input_dims=[self.env.get_observation_space()],
                alpha=weights[0],
                gamma=weights[1],
                epsilon=weights[2]
                )
        return init

    def plot_position_history(self, optimizer):
        # Obtain pos history from optimizer instance
        pos_history = optimizer.pos_history
        plot_contour(pos_history)

    def plot_results(self):
        plot(
            self.agent_id,
            "Pso_" + self.agent.agent_type,
            self.total_scores,
            self.env.get_total_record_count(),
            self.env.reward
            )

    def save_scores(self):
        scores_file_name = "history/scores/scores_PSO" \
            + self.agent.agent_type + "_" + str(self.agent_id) + ".csv"
        print(self.total_scores)
        # write the scores to a file.
        save_array_to_file(scores_file_name, self.total_scores)
