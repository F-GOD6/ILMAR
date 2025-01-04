import torch
import numpy as np

from torch import nn
from typing import Tuple
from .utils import build_mlp, reparameterize, evaluate_log_pi
from ..learner import Learner
from torch.distributions import Normal, Independent
import math
EPS = np.finfo(np.float32).eps
class StateIndependentPolicy(nn.Module):
    """
    Stochastic policy \pi(a|s)

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    hidden_units: tuple
        hidden units of the policy
    hidden_activation: nn.Module
        hidden activation of the policy
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            hidden_units: tuple = (64, 64),
            hidden_activation: nn.Module = nn.Tanh()
    ):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            init=True
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get the mean of the stochastic policy

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            mean of the stochastic policy
        """
        return torch.tanh(self.net(states))

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions given states

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            actions to take
        log_pi: torch.Tensor
            log_pi of the actions
        """
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log(\pi(a|s)) of the given action

        Parameters
        ----------
        states: torch.Tensor
            states that the actions act in
        actions: torch.Tensor
            actions taken

        Returns
        -------
        log_pi: : torch.Tensor
            log(\pi(a|s))
        """
        return evaluate_log_pi(self.net(states), self.log_stds, actions)



# 定义一个辅助函数实现 atanh，因为在较早的 PyTorch 版本中可能不存在 torch.atanh
def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))

class TanhActor(nn.Module):
    def __init__(self, 
                 state_shape, 
                 action_shape, 
                 hidden_units=[256, 256], 
                 name='TanhNormalPolicy',
                 mean_range=(-7., 7.), 
                 logstd_range=(-5., 2.), 
                 eps=1e-6, 
                 initial_std_scaler=1.0,
                 hidden_activation=nn.ReLU(),
                 kernel_initializer=nn.init.kaiming_normal_):
        """
        初始化 TanhActor 网络。

        Args:
            state_shape (int): 状态向量的维度。
            action_shape (int): 动作向量的维度。
            hidden_units (list, optional): 隐藏层的神经元数量列表。默认值为 [256, 256]。
            name (str, optional): 网络名称。默认值为 'TanhNormalPolicy'。
            mean_range (tuple, optional): 均值的范围。默认值为 (-7., 7.)。
            logstd_range (tuple, optional): 对数标准差的范围。默认值为 (-5., 2.)。
            eps (float, optional): 用于数值稳定性的极小值。默认值为 1e-6。
            initial_std_scaler (float, optional): 初始标准差的缩放因子。默认值为 1.0。
            hidden_activation (nn.Module, optional): 隐藏层的激活函数实例。默认值为 nn.ReLU()。
            kernel_initializer (function, optional): 权重初始化函数。默认值为 nn.init.kaiming_normal_。
        """
        super(TanhActor, self).__init__()

        self.action_shape = action_shape[0]
        self.initial_std_scaler = initial_std_scaler
        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

        # 定义 MLP 层
        layers = []
        input_dim = state_shape[0]
        for hidden_unit in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_unit))
            layers.append(hidden_activation)
            input_dim = hidden_unit
        self.fc_layers = nn.Sequential(*layers)

        # 定义均值和对数标准差的输出层
        self.fc_mean = nn.Linear(hidden_units[-1], action_shape[0])
        self.fc_logstd = nn.Linear(hidden_units[-1], action_shape[0])

        # 初始化权重
        self._initialize_weights(kernel_initializer)

    def _initialize_weights(self, initializer):
        """
        初始化网络中的线性层权重。

        Args:
            initializer (function): 权重初始化函数。
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                initializer(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs, training=True):
        """
        前向传播，生成动作及其对数概率。

        Args:
            inputs (torch.Tensor): 输入的状态张量，形状为 [batch_size, state_shape]。
            training (bool, optional): 是否处于训练模式。默认值为 True。

        Returns:
            tuple: 包含 (deterministic_action, action, log_prob) 和 network_state（None）的元组。
        """
        h = self.fc_layers(inputs)
        
        # 计算均值并限制范围
        mean = self.fc_mean(h)
        mean = torch.clamp(mean, self.mean_min, self.mean_max)
        
        # 计算对数标准差并限制范围
        logstd = self.fc_logstd(h)
        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)
        
        # 计算标准差
        std = torch.exp(logstd) * self.initial_std_scaler

        # 创建独立正态分布
        normal_dist = Normal(mean, std)
        pretanh_action_dist = Independent(normal_dist, 1)

        # 采样预激活动作（使用 rsample 以支持重参数化）
        pretanh_action = pretanh_action_dist.rsample()
        
        # 应用 Tanh 激活函数
        action = torch.tanh(pretanh_action)

        # 计算对数概率
        log_prob = self.log_prob(pretanh_action_dist, pretanh_action)

        # 计算确定性动作（均值经过 Tanh）
        deterministic_action = torch.tanh(mean)

        return deterministic_action  # network_state 为 None
    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，生成动作及其对数概率。

        Args:
            inputs (torch.Tensor): 输入的状态张量，形状为 [batch_size, state_shape]。
            training (bool, optional): 是否处于训练模式。默认值为 True。

        Returns:
            tuple: 包含 (deterministic_action, action, log_prob) 和 network_state（None）的元组。
        """
        h = self.fc_layers(states)
        # 计算均值并限制范围
        mean = self.fc_mean(h)
        mean = torch.clamp(mean, self.mean_min, self.mean_max)
        
        # 计算对数标准差并限制范围
        logstd = self.fc_logstd(h)
        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)
        
        # 计算标准差
        std = torch.exp(logstd) * self.initial_std_scaler

        # 创建独立正态分布
        normal_dist = Normal(mean, std)
        pretanh_action_dist = Independent(normal_dist, 1)

        # 采样预激活动作（使用 rsample 以支持重参数化）
        pretanh_action = pretanh_action_dist.rsample()
        
        # 应用 Tanh 激活函数
        action = torch.tanh(pretanh_action)

        # 计算对数概率
        log_prob = self.log_prob(pretanh_action_dist, pretanh_action)

        # 计算确定性动作（均值经过 Tanh）
        deterministic_action = torch.tanh(mean)

        return  action, log_prob  # network_state 为 None


    def log_prob(self, pretanh_action_dist, pretanh_action):
        """
        计算动作的对数概率。

        Args:
            pretanh_action_dist (Independent): 预激活动作的分布。
            pretanh_action (torch.Tensor): 预激活动作。

        Returns:
            torch.Tensor: 动作的对数概率，形状为 [batch_size]。
        """
        # 通过 Tanh 变换得到最终动作
        action = torch.tanh(pretanh_action)

        # 计算预激活动作的对数概率
        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_action)

        # 计算 Tanh 变换的雅可比行列式的对数值
        log_det_jacobian = torch.sum(torch.log(1 - action ** 2 + self.eps), dim=-1)

        # 总的对数概率
        log_prob = pretanh_log_prob - log_det_jacobian

        return log_prob

    def evaluate_log_pi(self, states, actions):
        """
        根据状态和动作计算对数概率。

        Args:
            states (torch.Tensor): 一批状态，形状为 [batch_size, state_shape]。
            actions (torch.Tensor): 一批动作，形状为 [batch_size, action_shape]。

        Returns:
            torch.Tensor: 动作的对数概率，形状为 [batch_size, 1]。
        """
        h = self.fc_layers(states)

        # 计算均值并限制范围
        mean = self.fc_mean(h)
        mean = torch.clamp(mean, self.mean_min, self.mean_max)

        # 计算对数标准差并限制范围
        logstd = self.fc_logstd(h)
        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)

        # 计算标准差
        std = torch.exp(logstd) * self.initial_std_scaler

        # 创建独立正态分布
        normal_dist = Normal(mean, std)
        pretanh_action_dist = Independent(normal_dist, 1)

        # 限制动作值以避免数值不稳定
        clipped_actions = torch.clamp(actions, -1 + self.eps, 1 - self.eps)

        # 计算预激活动作（atanh）
        # 检查是否支持 torch.atanh
        if hasattr(torch, 'atanh'):
            pretanh_actions = torch.atanh(clipped_actions)
        else:
            # 如果不支持，手动实现 atanh
            pretanh_actions = atanh(clipped_actions)

        # 计算预激活动作的对数概率
        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_actions)

        # 计算 Tanh 变换的雅可比行列式的对数值
        log_det_jacobian = torch.sum(torch.log(1 - actions ** 2 + self.eps), dim=-1)

        # 总的对数概率
        log_probs = pretanh_log_prob - log_det_jacobian

        # 为了避免广播问题，增加一个维度
        log_probs = log_probs.unsqueeze(-1)

        return log_probs







class StateIndependentState(nn.Module):
    """
    Stochastic policy \pi(a|s)

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    hidden_units: tuple
        hidden units of the policy
    hidden_activation: nn.Module
        hidden activation of the policy
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            hidden_units: tuple = (64, 64),
            hidden_activation: nn.Module = nn.Tanh()
    ):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            init=True
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get the mean of the stochastic policy

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            mean of the stochastic policy
        """
        return self.net(states)

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions given states

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            actions to take
        log_pi: torch.Tensor
            log_pi of the actions
        """
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log(\pi(a|s)) of the given action

        Parameters
        ----------
        states: torch.Tensor
            states that the actions act in
        actions: torch.Tensor
            actions taken

        Returns
        -------
        log_pi: : torch.Tensor
            log(\pi(a|s))
        """
        return evaluate_log_pi(self.net(states), self.log_stds, actions)


class MetaStateIndependentPolicy(nn.Module):
    """
    Stochastic policy \pi(a|s)

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    hidden_units: tuple
        hidden units of the policy
    hidden_activation: nn.Module
        hidden activation of the policy
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            hidden_units: tuple = (64, 64),
            hidden_activation: str = "tanh"
    ):
        super().__init__()
        model_config= [
        ('linear', [hidden_units[0], state_shape[0]]),
        (hidden_activation, [True]),
        ('linear', [hidden_units[1], hidden_units[0]]),
        (hidden_activation, [True]),
        ('linear', [action_shape[0], hidden_units[1]]),
    ]
        self.net = Learner(model_config)
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get the mean of the stochastic policy

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            mean of the stochastic policy
        """
        return torch.tanh(self.net(states))

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions given states

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            actions to take
        log_pi: torch.Tensor
            log_pi of the actions
        """
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states: torch.Tensor, actions: torch.Tensor,vars=None) -> torch.Tensor:
        """
        Evaluate the log(\pi(a|s)) of the given action

        Parameters
        ----------
        states: torch.Tensor
            states that the actions act in
        actions: torch.Tensor
            actions taken

        Returns
        -------
        log_pi: : torch.Tensor
            log(\pi(a|s))
        """
        if vars == None:
            return evaluate_log_pi(self.net(states), self.log_stds, actions)
        else:
            return evaluate_log_pi(self.net(states,vars), self.log_stds, actions)


class StateDependentPolicy(nn.Module):
    """
    State dependent policy defined in SAC

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    hidden_units: tuple
        hidden units of the policy
    hidden_activation: nn.Module
        hidden activation of the policy
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            hidden_units: tuple = (256, 256),
            hidden_activation: nn.Module = nn.ReLU(inplace=True)
    ):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get the mean of the stochastic policy

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            mean of the stochastic policy
        """
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions given states

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            actions to take
        log_pi: torch.Tensor
            log_pi of the actions
        """
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp(-20, 2))
    def evaluate_log_pi(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log(\pi(a|s)) of the given action

        Parameters
        ----------
        states: torch.Tensor
            states that the actions act in
        actions: torch.Tensor
            actions taken

        Returns
        -------
        log_pi: : torch.Tensor
            log(\pi(a|s))
        """
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return evaluate_log_pi(means, log_stds, actions)

class MetaStateDependentPolicy(nn.Module):
    """
    State dependent policy defined in SAC

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    hidden_units: tuple
        hidden units of the policy
    hidden_activation: nn.Module
        hidden activation of the policy
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            hidden_units: tuple = (256, 256),
            hidden_activation: str = "tanh",
            mean_range=(-7., 7.), 
            logstd_range=(-5., 2.),
            initial_std_scaler=1.0

    ):
        super().__init__()
        model_config= [
        ('linear', [hidden_units[0], state_shape[0]]),
        (hidden_activation, [True]),
        ('linear', [hidden_units[1], hidden_units[0]]),
        (hidden_activation, [True]),
        ('linear', [2*action_shape[0], hidden_units[1]]),
    ]
        self.net = Learner(model_config)
        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = EPS
        self.initial_std_scaler = initial_std_scaler

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get the mean of the stochastic policy

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            mean of the stochastic policy
        """
        means = self.net(states).chunk(2, dim=-1)[0]
        means = torch.clamp(means, self.mean_min, self.mean_max)
        return torch.tanh(means)

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions given states

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            actions to take
        log_pi: torch.Tensor
            log_pi of the actions
        """
        means, log_stds = self.net(states).chunk(2, dim=-1)
        means = torch.clamp(means, self.mean_min, self.mean_max)
        log_stds = torch.clamp(log_stds, self.logstd_min, self.logstd_max)
        std = torch.exp(log_stds) * self.initial_std_scaler
        normal_dist = Normal(means, std)
        pretanh_action_dist = Independent(normal_dist, 1)
        pretanh_action = pretanh_action_dist.rsample()
        action = torch.tanh(pretanh_action)
        log_prob = self.log_prob(pretanh_action_dist, pretanh_action)
        return action, log_prob

    def log_prob(self, pretanh_action_dist, pretanh_action) -> torch.Tensor:
        """
        计算动作的对数概率。

        Args:
            pretanh_action_dist (Independent): 预激活动作的分布。
            pretanh_action (torch.Tensor): 预激活动作。

        Returns:
            torch.Tensor: 动作的对数概率，形状为 [batch_size]。
        """
        # 通过 Tanh 变换得到最终动作

        action = torch.tanh(pretanh_action)

        # 计算预激活动作的对数概率
        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_action)

        # 计算 Tanh 变换的雅可比行列式的对数值
        log_det_jacobian = torch.sum(torch.log(1 - action ** 2 + self.eps), dim=-1)

        # 总的对数概率
        log_prob = pretanh_log_prob - log_det_jacobian

        return log_prob

    def evaluate_log_pi(self, states: torch.Tensor, actions: torch.Tensor,vars=None) -> torch.Tensor:
        """
        Evaluate the log(\pi(a|s)) of the given action

        Parameters
        ----------
        states: torch.Tensor
            states that the actions act in
        actions: torch.Tensor
            actions taken

        Returns
        -------
        log_pi: : torch.Tensor
            log(\pi(a|s))
        """
        if vars == None:
            means, log_stds = self.net(states).chunk(2, dim=-1)
        else:
            means, log_stds = self.net(states,vars).chunk(2, dim=-1)
        means = torch.clamp(means, self.mean_min, self.mean_max)
        log_stds = torch.clamp(log_stds, self.logstd_min, self.logstd_max)
        std = torch.exp(log_stds) * self.initial_std_scaler
        normal_dist = Normal(means, std)
        pretanh_action_dist = Independent(normal_dist, 1)
        clipped_actions = torch.clamp(actions, -1 + self.eps, 1 - self.eps)
        if hasattr(torch, 'atanh'):
            pretanh_actions = torch.atanh(clipped_actions)
        else:
            # 如果不支持，手动实现 atanh
            pretanh_actions = atanh(clipped_actions)
        # 计算预激活动作的对数概率
        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_actions)

        # 计算 Tanh 变换的雅可比行列式的对数值
        log_det_jacobian = torch.sum(torch.log(1 - actions ** 2 + self.eps), dim=-1)

        # 总的对数概率
        log_probs = pretanh_log_prob - log_det_jacobian

        # 为了避免广播问题，增加一个维度
        log_probs = log_probs.unsqueeze(-1)
        return log_probs



class DeterministicPolicy(nn.Module):
    """
    Deterministic policy

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    hidden_units: tuple
        hidden units of the policy
    hidden_activation: nn.Module
        hidden activation of the policy
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            hidden_units: tuple = (256, 256),
            hidden_activation: nn.Module = nn.ReLU(inplace=True)
    ):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            init=True
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get actions given states

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            actions to take
        """
        return torch.tanh(self.net(states))
