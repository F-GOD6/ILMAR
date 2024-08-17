import os
import gym
import torch
import numpy as np

from torch import nn
from torch.optim import Adam,SGD
from torch.utils.tensorboard import SummaryWriter
from .base import Algorithm, Expert
from ilmar.buffer import Buffer
from ilmar.utils import soft_update, disable_gradient
from ilmar.network import MetaStateIndependentPolicy,AdvantageStateActionFunction
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
    expert_loss = -torch.mean(torch.log(torch.clamp(expert_cost_val - label_smoothing, min=1e-12, max=1.0-1e-12)))
    union_loss = -torch.mean(torch.log(torch.clamp(1. - union_cost_val - label_smoothing, min=1e-12, max=1.0-1e-12)))
    return expert_loss + union_loss
class ILMAR(Algorithm):
    """
    Implementation of ILMAR
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
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            units_actor: tuple = (256, 256),
            units_critic: tuple = (256, 256),
            start_steps: int = 10000,
            tau: float = 0,
            grad_reg_coef : float= 0.2,
            alpha: float=1,
            beta: float=1,
            phi: int = 3,
    ):
        super().__init__(state_shape, action_shape, device, seed, gamma)
        self.buffer_exp = buffer_exp
        self.buffer_union = buffer_union
        # actor
        self.actor = MetaStateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor
        ).to(device)
        self.critic = AdvantageStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_critic,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(),lr=lr_critic)
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.tau = tau
        self.grad_reg_coef = grad_reg_coef
        self.device = device
        self.alpha=alpha
        self.beta=beta
        self.phi=phi
    def is_update(self, step: int) -> bool:
        return step >= max(self.start_steps, self.batch_size)

    def step(self, env: gym.wrappers.TimeLimit, state: np.array, t: int, step: int):
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
        self.learning_steps += 1
        expert_states, expert_actions, rewards, dones, next_states,_ = \
            self.buffer_exp.sample(self.batch_size)
        union_states,union_actions, rewards, dones, next_states,_ = \
            self.buffer_union.sample(self.batch_size)
        expert_actions_pred,_=self.actor.sample(expert_states)
        union_actions_pred ,_=self.actor.sample(union_states)
        expert_actions_pred=expert_actions_pred.detach()
        union_actions_pred=union_actions_pred.detach()
        random_action_expert=torch.rand_like(expert_actions)
        better_0 = self.critic(expert_states ,expert_actions,expert_actions_pred)               
        better_1= self.critic(expert_states ,expert_actions,random_action_expert)              
        worse_1 = self.critic(expert_states ,random_action_expert,expert_actions)
        better_2 = self.critic(expert_states,expert_actions_pred,random_action_expert)
        worse_2 = self.critic(expert_states,random_action_expert,expert_actions_pred)
        random_action_union=torch.rand_like(union_actions)
        better_3 = self.critic(union_states,union_actions_pred,random_action_union)
        worse_3 = self.critic(union_states,random_action_union,union_actions_pred)
        better_4 = self.critic(union_states,union_actions,random_action_union)
        worse_4 = self.critic(union_states,random_action_union,union_actions)
        union_cost_val= self.critic(union_states,union_actions,union_actions_pred)                       
        unif_rand = torch.rand(size=(expert_states.shape[0], 1)).to(self.device)
        mixed_states = unif_rand * expert_states + (1 - unif_rand) * union_states
        mixed_actions = unif_rand * expert_actions + (1 - unif_rand) * union_actions
        mixed_actions_pred ,_= self.actor.sample(mixed_states)
        mixed_actions_pred=mixed_actions_pred.detach()
        indices = torch.randperm(union_states.size(0))
        mixed_states_2= unif_rand * mixed_states[indices] + (1 - unif_rand) * mixed_states
        mixed_actions_2= unif_rand * mixed_actions[indices] + (1 - unif_rand) * mixed_actions
        mixed_actions_pred_2= unif_rand * mixed_actions_pred[indices] + (1 - unif_rand) *  mixed_actions_pred
        mixed_states_2.requires_grad_(True)
        mixed_actions_2.requires_grad_(True)
        mixed_actions_pred_2.requires_grad_(True)
        cost_output = self.critic(mixed_states_2,mixed_actions_2,mixed_actions_pred_2)
        cost_output = torch.log(1/(torch.nn.Sigmoid()(cost_output)+EPS2)- 1 + EPS2)         
        cost_mixed_grad = torch.autograd.grad(outputs=cost_output,inputs=mixed_states_2 ,grad_outputs=torch.ones_like(cost_output),create_graph=True,retain_graph=True,only_inputs=True)[0]+ EPS
        cost_mixed_grad2= torch.autograd.grad(outputs=cost_output,inputs=mixed_actions_2 ,grad_outputs=torch.ones_like(cost_output),create_graph=True,retain_graph=True,only_inputs=True)[0]
        cost_mixed_grad3= torch.autograd.grad(outputs=cost_output,inputs=mixed_actions_pred_2 ,grad_outputs=torch.ones_like(cost_output),create_graph=True,retain_graph=True,only_inputs=True)[0]+EPS
        cost_grad_penalty = ((cost_mixed_grad.norm(2, dim=1) - 1) ** 2).mean() + ((cost_mixed_grad2.norm(2, dim=1) - 1) ** 2).mean()+((cost_mixed_grad3.norm(2, dim=1) - 1) ** 2).mean()
        cost_loss = -torch.mean(torch.log(torch.clamp(better_0, min=1e-12, max=1e-12)))+minimax_discriminator_loss(better_1, worse_1, label_smoothing=0.) \
                         +minimax_discriminator_loss(better_2, worse_2, label_smoothing=0.)+minimax_discriminator_loss(better_3, worse_3, label_smoothing=0.)\
                              +minimax_discriminator_loss(better_4, worse_4, label_smoothing=0.)+self.grad_reg_coef * cost_grad_penalty
        cost_prob = torch.nn.Sigmoid()(union_cost_val)
        if self.phi==1:
            weight = (cost_prob / (1 - cost_prob))
        elif self.phi==2:
            union_cost = torch.log(1 / (torch.nn.Sigmoid()(union_cost_val) + EPS2) - 1 + EPS2)
            weight = (torch.exp(-union_cost))
        elif self.phi ==3:
            weight = torch.tan(cost_prob * torch.pi/2)
        elif self.phi ==0:
            weight = cost_prob
        indices = (weight >= self.tau).float()
        pi_loss = - (indices * weight * self.actor.evaluate_log_pi(union_states, union_actions)).mean()
        fast_weights = self.actor.net.parameters()
        grad = torch.autograd.grad(pi_loss, fast_weights, create_graph=True,retain_graph=True)
        fast_weights = list(map(lambda p: p[1] - self.lr_actor * p[0], zip(grad, fast_weights)))
        meta_loss = -(self.actor.evaluate_log_pi(expert_states, expert_actions,fast_weights)).mean()
        self.update_critic(cost_loss,meta_loss,writer)
        pi_loss = - (indices * weight.detach() * self.actor.evaluate_log_pi(union_states, union_actions)).mean() 
        self.update_actor(pi_loss,writer)
        

    def update_actor(self, pi_loss,writer: SummaryWriter):
        self.optim_actor.zero_grad()
        pi_loss.backward(retain_graph=True)
        self.optim_actor.step()
        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/actor', pi_loss.item(), self.learning_steps)
    def update_critic(
            self,
            loss,
            meta_loss,
            writer: SummaryWriter
    ):
        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/critic', loss.item(), self.learning_steps)

        self.optim_critic.zero_grad()
        loss = self.alpha * meta_loss + self.beta * loss
        loss.backward(retain_graph=False)
        self.optim_critic.step()

    def save_config(self):
        config =  {
                    "exp_name":"ILMAR",
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "phi": self.phi,
                    "lr_actor": self.lr_actor,
                    "lr_critic": self.lr_critic,
                    "grad_reg_coef":self.grad_reg_coef}
        return config


    def save_models(self, save_dir: str):
        super().save_models(save_dir)
        torch.save(
            self.actor.state_dict(),
            os.path.join(save_dir, 'actor.pth')
        )


class ILMARExpert(Expert):
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            device: torch.device,
            path: str,
            units_actor: tuple = (256, 256)
    ):
        super(ILMARExpert, self).__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )
        self.actor = MetaStateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor
        ).to(device)
        self.actor.load_state_dict(torch.load(path, map_location=device))
        disable_gradient(self.actor)
