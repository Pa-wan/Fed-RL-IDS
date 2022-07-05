import numpy as np


class ReplayBuffer():
    """
        Replay Buffer

        based on the code provided by
        https://www.youtube.com/channel/UC58v9cLitc8VaCjrcKyAbrw
    """

    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros(
            (self.mem_size, *input_shape),
            dtype=np.float32
            )  # noqa: E501

        self.new_state_memory = np.zeros(
            (self.mem_size, *input_shape),
            dtype=np.float32)

        self.action_memory = np.zeros(
            (self.mem_size, n_actions),
            dtype=np.bool
            )  # noqa: E501

        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float)

    def store_transition(self, state, action, reward, state_, done):
        """
            Stores the transition in the memory buffer
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.get_action_for_transition(action, index)

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
            Samples a batch of transitions from the memory buffer
        """
        max_mem = self.mem_cntr % self.mem_size
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

    def get_action_for_transition(self, action, index):
        """
            Populates the action space for the given action and index
        """
        actions = np.zeros(self.action_memory.shape[1])
        actions[action] = 1
        self.action_memory[index] = actions
