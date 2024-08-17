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
from ilmar.network import StateIndependentPolicy, TwinnedStateActionFunction
from ilmar.buffer import SerializedBuffer

class BC(Algorithm):
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
            state_shape: np.array,
            action_shape: np.array,
            device: torch.device,
            seed: int,
            gamma: float = 0.99,
            batch_size: int = 256,
            rollout_length: int = 10**6,
            lr_actor: float = 3e-4,
            units_actor: tuple = (256, 256),
            units_critic: tuple = (256, 256),
            start_steps: int = 10000,
            tau: float = 5e-3
    ):
        super().__init__(state_shape, action_shape, device, seed, gamma)
        self.buffer_exp = buffer_exp

        # actor
        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)


        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)

        self.batch_size = batch_size
        self.start_steps = start_steps
        self.tau = tau

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

        next_state, reward, done, _ = env.step(action)
        mask = True if t == env.max_episode_steps else done

        self.buffer.append(state, action, reward, mask, next_state)

        if done or t == env.max_episode_steps:
            t = 0
            next_state = env.reset()

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
        states, actions, rewards, dones, next_states,_ = \
            self.buffer_exp.sample(self.batch_size)

        self.update_actor(states,actions,writer)

    

    def update_actor(self, states: torch.Tensor,actions: torch.Tensor,writer: SummaryWriter):
        """
        Update the actor for one step

        Parameters
        ----------
        states: torch.Tensor
            sampled states
        writer: SummaryWriter
            writer for logs
        """
        loss_actor = - (self.actor.evaluate_log_pi(states, actions)).mean() 
        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

        

        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)



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


class BCExpert(Expert):
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
        super(BCExpert, self).__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )
        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.actor.load_state_dict(torch.load(path, map_location=device))
        disable_gradient(self.actor)
