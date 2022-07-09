from dataclasses import dataclass


@dataclass
class AgentInitialiser:
    input_dims: list
    n_actions: int = 2
    gamma: float = 0.01
    epsilon: float = 1.0
    epsilon_dec: float = 0.995
    epsilon_min: float = 0.01
    model_file: str = "models/ddqn_model.h5"
    alpha: float = 0.005
    replace_target: int = 100
    type_name: str = "Double_Q_Agent"
    batch_size: int = 34
    memory_size: int = 10000
    fc1_dims: int = 128
    fc2_dims: int = 128
    learning_rate: float = 0.001
