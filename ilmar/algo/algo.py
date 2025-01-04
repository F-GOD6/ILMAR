from .ppo import PPO, PPOExpert
from .sac import SAC, SACExpert
from .ilmar import ILMAR,ILMARExpert
# all the algorithms
ALGOS = {
    'sac': SAC,
    'ppo': PPO,
    'ilmar':ILMAR,
}

# all the well-trained algorithms
EXP_ALGOS = {
    'sac': SACExpert,
    'ppo': PPOExpert,
    'ilmar':ILMARExpert,
}