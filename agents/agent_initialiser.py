from dataclasses import dataclass


@dataclass
class AgentInitialiser:
    action_space: list
    n_actions: int
    gamma: float
    epsilon: float
    epsilon_dec: float
    epsilon_min: float
    model_file: str
    replace_target: int = 100
    type_name: str = "Double_Q_Agent"
    batch_size: int = 34
