import flwr as fl

from agents.ddqn_agent import ddqn_agent
from environ.nsl_kdd import nsl_kdd_env
from helpers.plotter import plot
from helpers.file_helpers import save_array_to_file


class dqn_flower_client(fl.client.NumPyClient):

    def __init__(self, agent: ddqn_agent, env: nsl_kdd_env, agent_id: int):
        """
            Initialises the flower client

            Parameters:
            -----------
            agent (RL agent): The agent to use for the client
            env: (Environment): The environment to use for the client
        """
        super(dqn_flower_client, self).__init__()
        self.agent = agent
        self.env = env
        self.agent_id = agent_id
        self.scores = []

    def get_parameters(self):
        """
            Returns the parameters of the agent
        """
        return self.agent.get_weights()

    def fit(self, parameters, config):
        """
            Trains the agent
        """
        self.agent.set_weights(parameters)
        self.learn(1)
        return self.agent.get_weights(), self.env.get_total_record_count(), {}

    def evaluate(self, parameters, config):
        """
            Evaluates the agent's neural network
        """
        self.agent.set_weights(parameters)
        loss, accuracy = self.evaluate_agent()
        return loss, self.env.get_total_record_count(), {"accuracy": accuracy}

    def learn(self, epochs):
        """
            Trains the agent
        """
        print("Start learn")
        eps_history = []
        for epoch in range(epochs):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                score += reward
                self.agent.learn()

            eps_history.append(self.agent.epsilon)
            # print(score)
            self.scores.append(score)
        return score

    def evaluate_agent(self):
        """
            Evaluates the agent
        """
        # make sure it uses the Q Network not random guesses.
        self.agent.epsilon = self.agent.epsilon_min
        score = 0
        done = False
        # change to set it up for testing instead of training.
        state = self.env.reset()
        while not done:
            action = self.agent.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            score += reward
            state = next_state

        loss = score / self.env.get_total_record_count()
        accuracy = score
        return loss, accuracy

    def plot_results(self):
        plot(
            self.agent_id,
            self.agent.agent_type,
            self.scores,
            self.env.get_total_record_count(),
            self.env.reward
            )

    def save_model(self):
        self.agent.save_model()

    def save_scores(self):
        scores_file_name = "history/scores/scores_" \
            + self.agent.agent_type + "_" + str(self.agent_id) + ".csv"
        print(self.scores)
        # write the scores to a file.
        save_array_to_file(scores_file_name, self.scores)
