from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer

class ExponentialScheduler(LambdaLR):

    def __init__(self, 
                 optimizer: Optimizer, 
                 max_steps: int, 
                 gamma: float = 10., 
                 power: float = 0.75) -> None:
        
        self.max_steps = max_steps
        self.gamma = gamma
        self.power = power
    
        decay_fnc = lambda step: (1 + gamma * step / max_steps) ** (-power)
        super().__init__(optimizer=optimizer, lr_lambda=decay_fnc)


class ConstantDecayScheduler(LambdaLR):
    def __init__(self, 
                 optimizer: Optimizer, 
                 steps: int,
                 decay: float) -> None:
        
        self.steps = steps
        self.decay = decay
    
        decay_fnc = lambda step: 1.0 if step < self.steps else self.decay
        super().__init__(optimizer=optimizer, lr_lambda=decay_fnc)
        