import os
import gym
import torch
import numpy as np

from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from .base import Algorithm, Expert
from ilmar.buffer import Buffer
from ilmar.utils import soft_update, disable_gradient
from ilmar.network import StateDependentPolicy, TwinnedStateActionFunction , StateActionFunction , StateIndependentPolicy,StateActionConcatedFunction
from ilmar.buffer import SerializedBuffer
EPS = np.finfo(np.float32).eps
EPS2 = 1e-3
def minimax_discriminator_loss(expert_cost_val, union_cost_val, label_smoothing=0.):
    """
    Implements the Minimax discriminator loss function.
    
    Args:
        expert_cost_val (torch.Tensor): The discriminator's output for real samples.
        union_cost_val (torch.Tensor): The discriminator's output for generated samples.
        label_smoothing (float, optional): The amount of label smoothing to apply. Defaults to 0.
    
    Returns:
        torch.Tensor: The Minimax discriminator loss.
    """
    expert_loss = -torch.mean(torch.log(torch.clamp(expert_cost_val - label_smoothing, min=1e-12, max=1.0)))
    union_loss = -torch.mean(torch.log(torch.clamp(1. - union_cost_val - label_smoothing, min=1e-12, max=1.0)))
    return expert_loss + union_loss
class ISWBC(Algorithm):
    """
    Implementation of BC



    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    device: torch.device
        cpu or cuda
    seed: int
        random seed
    gamma: float
        discount factor
    batch_size: int
        batch size for sampling in the replay buffer
    rollout_length: int
        rollout length of the buffer
    lr_actor: float
        learning rate of the actor
    lr_critic: float
        learning rate of the critic
    lr_alpha: float
        learning rate of log(alpha)
    units_actor: tuple
        hidden units of the actor
    units_critic: tuple
        hidden units of the critic
    start_steps: int
        start steps. Training starts after collecting these steps in the environment.
    tau: float
        tau coefficient
    """
    def __init__(
            self,
            buffer_exp: SerializedBuffer,
            buffer_union:SerializedBuffer,
            state_shape: np.array,
            action_shape: np.array,
            device: torch.device,
            seed: int,
            gamma: float = 0.99,
            batch_size: int = 256,
            lr_actor: float = 3e-4, #3e-5 for humanoid 
            lr_critic: float = 3e-4,
            units_actor: tuple = (256, 256),
            units_critic: tuple = (256, 256),
            start_steps: int = 10000,
            tau: float = 0.0,
            grad_reg_coef : float=1.0,
    ):
        super().__init__(state_shape, action_shape, device, seed, gamma)
        self.buffer_exp = buffer_exp
        self.buffer_union = buffer_union
        # actor
        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.critic = StateActionConcatedFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(),lr=lr_critic)
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.tau = tau
        self.grad_reg_coef = grad_reg_coef
        self.device = device
    def is_update(self, step: int) -> bool:
        """
        Whether the time is for update

        Parameters
        ----------
        step: int
            current training step

        Returns
        -------
        update: bool
            whether to update. SAC updates when the step is larger
            than the start steps and the batch size
        """
        return step >= max(self.start_steps, self.batch_size)

    def step(self, env: gym.wrappers.TimeLimit, state: np.array, t: int, step: int):
        """
        Sample one step in the environment

        Parameters
        ----------
        env: gym.wrappers.TimeLimit
            environment
        state: np.array
            current state
        t: int
            current time step in the episode
        step: int
            current total steps

        Returns
        -------
        next_state: np.array
            next state
        t: int
            time step
        """
        t += 1

        if step <= self.start_steps:
            action = env.action_space.sample()
        else:
            action = self.explore(state)[0]
        next_state, reward, terminated, truncated,_ = env.step(action)
        mask = terminated or truncated
        self.buffer.append(state, action, reward, mask, next_state)
        if mask :
            t = 0
            next_state = env.reset()[0]

        return next_state, t

    def update(self, writer: SummaryWriter):
        """
        Update the algorithm

        Parameters
        ----------
        writer: SummaryWriter
            writer for logs
        """
        self.learning_steps += 1
        expert_states, expert_actions, rewards, dones, next_states,_ = \
            self.buffer_exp.sample(self.batch_size)
        union_states,union_actions, rewards, dones, next_states,_ = \
            self.buffer_union.sample(self.batch_size)
        expert_inputs = torch.cat([expert_states, expert_actions], -1)
        union_inputs = torch.concat([union_states, union_actions], -1)
        expert_cost_val = self.critic(expert_inputs)
        union_cost_val= self.critic(union_inputs)                            
        unif_rand = torch.rand(size=(expert_states.shape[0], 1)).to(self.device)
        mixed_inputs1 = unif_rand * expert_inputs + (1 - unif_rand) * union_inputs
        indices = torch.randperm(union_inputs.size(0))

        mixed_inputs2 = unif_rand * union_inputs[indices] + (1 - unif_rand) * union_inputs
        mixed_inputs = torch.concat([mixed_inputs1, mixed_inputs2], 0)             
        mixed_inputs.requires_grad_(True)
        cost_output = self.critic(mixed_inputs)
        cost_output = torch.log(1/(torch.nn.Sigmoid()(cost_output)+EPS2)- 1 + EPS2)           
        cost_mixed_grad = torch.autograd.grad(outputs=cost_output,inputs=mixed_inputs ,grad_outputs=torch.ones_like(cost_output),create_graph=True,retain_graph=True,only_inputs=True)[0]+EPS
        cost_grad_penalty = ((cost_mixed_grad.norm(2, dim=1) - 1) ** 2).mean()
        cost_loss = minimax_discriminator_loss(expert_cost_val, union_cost_val, label_smoothing=0.) \
                        + self.grad_reg_coef * cost_grad_penalty
        cost_prob = torch.nn.Sigmoid()(union_cost_val)
        weight = (cost_prob / (1 - cost_prob))
        indices = (weight >= self.tau).float()
        pi_loss = - (indices * weight.detach() * self.actor.evaluate_log_pi(union_states, union_actions)).mean()
        self.update_actor(pi_loss,writer)
        self.update_critic(cost_loss,writer)
    

    def update_actor(self, pi_loss,writer: SummaryWriter):
        """
        Update the actor for one step

        Parameters
        ----------
        states: torch.Tensor
            sampled states
        writer: SummaryWriter
            writer for logs
        """

        self.optim_actor.zero_grad()
        pi_loss.backward(retain_graph=False)
        self.optim_actor.step()

        
    def update_critic(
            self,
            loss,
            writer: SummaryWriter
    ):
        """
        Update the critic for one step

        Parameters
        ----------
        writer: SummaryWriter
            writer for logs
        """
 

        self.optim_critic.zero_grad()
        loss.backward(retain_graph=False)
        self.optim_critic.step()

        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/critic', loss.item(), self.learning_steps)
    def save_models(self, save_dir: str):
        """
        Save the model

        Parameters
        ----------
        save_dir: str
            path to save
        """
        super().save_models(save_dir)
        # we only save actor to reduce workloads
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )


class ISWBCExpert(Expert):
    """
    Well-trained SAC agent

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    device: torch.device
        cpu or cuda
    path: str
        path to the well-trained weights
    units_actor: tuple
        hidden units of the actor
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            device: torch.device,
            path: str,
            units_actor: tuple = (256, 256)
    ):
        super(ISWBCExpert, self).__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )
        self.actor = StateDependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.actor.load_state_dict(torch.load(path, map_location=device))
        disable_gradient(self.actor)
