import os
import gym
import torch
import numpy as np

from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from .base import Algorithm, Expert
from ilmar.buffer import sample_from_dataset
from ilmar.utils import soft_update, disable_gradient
from ilmar.network import AdvantageStateActionFunction,MetaStateDependentPolicy
import wandb
EPS = np.finfo(np.float32).eps
EPS2 = 1e-3
def minimax_discriminator_loss(expert_cost_val, union_cost_val, label_smoothing=0.):

    real_labels = torch.ones_like(expert_cost_val) * (1.0 - label_smoothing)
    fake_labels = torch.zeros_like(union_cost_val)

    loss_fn = nn.BCEWithLogitsLoss()

    loss_real = loss_fn(expert_cost_val, real_labels)
    loss_fake = loss_fn(union_cost_val, fake_labels)

    total_loss = loss_real + loss_fake

    return total_loss

class ILMAR(Algorithm):
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
            buffer_exp,
            buffer_union,
            state_shape: np.array,
            action_shape: np.array,
            device: torch.device,
            seed: int,
            gamma: float = 0.99,
            batch_size: int = 256,
            lr_actor: float = 3e-5,
            lr_critic: float = 3e-4,
            units_actor: tuple = (256, 256),
            units_critic: tuple = (256, 256),
            start_steps: int = 10000,
            tau: float = 0,
            grad_reg_coef : float= 0.2,
            config = None,
    ):
        super().__init__(state_shape, action_shape, device, seed, gamma)
        self.buffer_exp = buffer_exp
        self.buffer_union = buffer_union
        # actor
        self.actor = MetaStateDependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation="relu"
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
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.phi=config["phi"]
        
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

    def update(self, writer: SummaryWriter,path):
        """
        Update the algorithm

        Parameters
        ----------
        writer: SummaryWriter
            writer for logs
        """
        self.learning_steps += 1
        expert_states, expert_actions, _ , _, rewards,_ = sample_from_dataset(self.buffer_exp,self.batch_size,self.device)
        union_states,union_actions,next_states, _, union_rewards,_ = sample_from_dataset(self.buffer_union,self.batch_size,self.device)
        expert_actions_pred,_=self.actor.sample(expert_states)
        union_actions_pred ,_=self.actor.sample(union_states)
        expert_actions_pred=expert_actions_pred.detach()
        union_actions_pred=union_actions_pred.detach()
        random_action_expert=torch.rand_like(expert_actions) * 2 -1
        better_0 = self.critic(expert_states ,expert_actions,expert_actions_pred)               
        better_1= self.critic(expert_states ,expert_actions,random_action_expert)               
        worse_1 = self.critic(expert_states ,random_action_expert,expert_actions)               
        better_2 = self.critic(expert_states,expert_actions_pred,random_action_expert)          
        worse_2 = self.critic(expert_states,random_action_expert,expert_actions_pred)           
        random_action_union=torch.rand_like(union_actions)  * 2 -1                                    
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
        # cost_loss = minimax_discriminator_loss(expert_cost_val, random_cost_val, label_smoothing=0.)+minimax_discriminator_loss(random_cost_val_r,expert_cost_val_r,label_smoothing=0.) \
        #                  + self.grad_reg_coef * cost_grad_penalty
        cost_loss = -torch.mean(torch.log(torch.clamp(better_0, min=1e-12, max=1)))+minimax_discriminator_loss(better_1, worse_1, label_smoothing=0.) \
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
        # weight = cost_prob
        weight = weight - 0.5
        indices = (weight >= self.tau).float()

        pi_loss = - (indices * weight * self.actor.evaluate_log_pi(union_states, union_actions)).mean()
        fast_weights = self.actor.net.parameters()
        grad = torch.autograd.grad(pi_loss, fast_weights, create_graph=True,retain_graph=True)
        fast_weights = list(map(lambda p: p[1] - self.lr_actor * p[0], zip(grad, fast_weights)))
        meta_loss = -(self.actor.evaluate_log_pi(expert_states, expert_actions,fast_weights)).mean()
        self.update_critic(cost_loss,meta_loss,writer)
        pi_loss = - (indices * weight.detach() * self.actor.evaluate_log_pi(union_states, union_actions)).mean()
        self.update_actor(pi_loss,writer)
        loss = -(self.actor.evaluate_log_pi(expert_states, expert_actions)).mean()


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
        pi_loss.backward(retain_graph=True)
        #for param in self.actor.parameters():
        #     param.data.sub_(self.lr_actor * param.grad.data)
        self.optim_actor.step()
        if self.learning_steps % 1000 == 0:
            writer.add_scalar(
                'loss/actor', pi_loss.item(), self.learning_steps)
            wandb.log({ "actor loss": pi_loss.item()},step=self.learning_steps)
    def update_critic(
            self,
            loss,
            meta_loss,
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
        final_loss = self.alpha * meta_loss + self.beta * loss
        if self.learning_steps % 1000 == 0:
            wandb.log({"meta loss": meta_loss.item(),
                        "vanilla loss": loss.item(),
                        "final loss": final_loss.item()},step=self.learning_steps)
        # loss = alpha * meta_loss
        final_loss.backward(retain_graph=False)
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


class ILMARExpert(Expert):
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
        super(ILMARExpert, self).__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )
        self.actor = MetaStateDependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation="relu"
        ).to(device)
        self.actor.load_state_dict(torch.load(path, map_location=device))
        disable_gradient(self.actor)
