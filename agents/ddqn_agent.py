import numpy as np

from agents.agent_initialiser import AgentInitialiser
from memory.replay_memory import ReplayBuffer


class ddqn_agent():
    """
        Double Deep Q Network Agent
    """

    def __init__(self, initialiser: AgentInitialiser):
        print("Agent created.")
        self.agent_type = initialiser.type_name
        self.action_space = initialiser.action_space
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

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            q_values = self.model.predict(state)
            action = np.argmax(q_values[0])

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            states, actions, rewards, next_states, dones = \
                self.memory.sample_buffer(self.batch_size)
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(actions, action_values)

            

