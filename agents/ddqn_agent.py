import numpy as np
import tensorflow as tf

from agents.agent_initialiser import AgentInitialiser
from memory.replay_memory import ReplayBuffer
from ml_models.deep_q_network import deep_q_model


class ddqn_agent():
    """
        Double Deep Q Network Agent
    """

    def __init__(self, initialiser: AgentInitialiser):
        print("Agent created.")
        self.agent_type = initialiser.type_name
        self.action_space = [0, 1]
        self.n_actions = initialiser.n_actions
        self.gamma = initialiser.gamma
        self.epsilon = initialiser.epsilon
        self.epsilon_dec = initialiser.epsilon_dec
        self.epsilon_min = initialiser.epsilon_min
        self.model_file = initialiser.model_file
        self.replace_target = initialiser.replace_target
        self.batch_size = initialiser.batch_size
        self.memory = ReplayBuffer(
                            initialiser.memory_size,
                            initialiser.input_dims,
                            initialiser.n_actions
                            )
        self.q_eval = self.build_model(initialiser)
        self.q_target = self.build_model(initialiser)

    def get_weights(self):
        """
            Get the weights of the Q neural network.
        """
        return self.q_eval.get_weights()
    
    def set_weights(self, weights):
        """
            Set the weights of the Q neural network.
        """
        temp_weights = self.get_weights()
        try:
            self.q_eval.set_weights(weights)
        except(ValueError):
            print("Error setting weights.")
            self.q_eval.set_weights(temp_weights)

    def remember(self, state, action, reward, next_state, done):
        """
            Remember the current state, action, reward, next_state, done
        """
        self.memory.store_transition(state, action, reward, next_state, done)

    def choose_action(self, state):
        """
            Choose an action based on the current state
        """
        state = state[np.newaxis, :]
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            q_values = self.q_eval.predict(state)
            action = np.argmax(q_values[0])

        return action

    def learn(self):
        """
            Conduct training on the batch of data.
        """
        if self.memory.mem_cntr > self.batch_size:
            states, actions, rewards, next_states, dones = \
                self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(actions, action_values)

            q_next = self.q_target.predict(next_states)
            q_eval = self.q_eval.predict(next_states)
            q_pred = self.q_eval.predict(states)
            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = rewards + \
                self.gamma * q_next[batch_index, max_actions.astype(int)] * \
                dones

            self.q_eval.fit(states, q_target, verbose=0)

            self.epsilon = self.epsilon * self.epsilon_dec \
                if self.epsilon > self.epsilon_min else self.epsilon_min

            if self.memory.mem_cntr % self.replace_target == 0:
                self.q_target = self.q_eval
                print("Target network updated.")

    def build_model(self, initialiser: AgentInitialiser):
        """
            Helper method to build the Q neural network.
        """
        model = deep_q_model(
                initialiser.input_dims,
                initialiser.fc1_dims,
                initialiser.fc2_dims,
                self.n_actions
                )

        opt = tf.keras.optimizers.Adam(lr=initialiser.alpha)

        model.compile(
            optimizer=opt,
            loss='mse'
            )

        return model
