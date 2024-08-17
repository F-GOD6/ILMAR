from .ppo import PPO, PPOExpert
from .sac import SAC, SACExpert
from .bc import BC,BCExpert
from .iswbc import ISWBC,ISWBCExpert
from .ilmar import ILMAR,ILMARExpert
from .demodice import DemoDICE,DemoDICEExpert
# all the algorithms
ALGOS = {
    'sac': SAC,
    'ppo': PPO,
    'bc':BC,
    'iswbc': ISWBC,
    'demodice': DemoDICE,
    'ilmar':ILMAR,
}

# all the well-trained algorithms
EXP_ALGOS = {
    'sac': SACExpert,
    'ppo': PPOExpert,
    'bc':BCExpert,
    'iswbc':ISWBCExpert,
    'demodice':DemoDICEExpert,
    'ilmar':ILMARExpert,
}