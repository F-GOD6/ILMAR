import os
import argparse
import torch
import numpy as np
from datetime import datetime
from ilmar.env import make_env
from ilmar.buffer import SerializedBuffer
from ilmar.algo.algo import ALGOS
from ilmar.trainer import Trainer
from ilmar.utils import return_range
from ilmar.env import get_dataset
import wandb
import pickle
def get_device():
    # 检查 CUDA_VISIBLE_DEVICES 环境变量
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    

    if visible_devices is not None and torch.cuda.is_available():
        print(f"Using GPU: {visible_devices}")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    
    return device
def run(args):
    """Train Imitation Learning algorithms"""

    expert_dataset = get_dataset(dirname=args.dirname,env_id=args.env_id, dataname="expert",num_trajectories=args.expert_num_trajectories)
    suboptimal_dataset = {
            'init_states': [],
            'states': [],
            'actions': [],
            'next_states': [],
            'dones': [],
            'rewards': []
        }
    union_dataset = {
            'init_states': [],
            'states': [],
            'actions': [],
            'next_states': [],
            'dones': [],
            'rewards': []
        }

    expert_dataset_name = "expert"
    if len(args.suboptimal_dataset_names) > 0:
        for suboptimal_datatype_idx, (suboptimal_dataset_name, suboptimal_num_traj) in enumerate(
                zip(args.suboptimal_dataset_names, args.suboptimal_num_trajs)):
            start_idx = args.expert_num_trajectories if (expert_dataset_name == suboptimal_dataset_name) else 0

            dataset = get_dataset(args.dirname, args.env_id, suboptimal_dataset_name, suboptimal_num_traj, start_idx=start_idx)
            suboptimal_dataset["init_states"].append(dataset["init_states"])
            suboptimal_dataset["states"].append(dataset["states"])
            suboptimal_dataset["actions"].append(dataset["actions"])
            suboptimal_dataset["next_states"].append(dataset["next_states"])
            suboptimal_dataset["dones"].append(dataset["dones"])
            suboptimal_dataset["rewards"].append(dataset["rewards"])

    suboptimal_dataset["init_states"] = np.concatenate(suboptimal_dataset["init_states"]).astype(np.float32)
    suboptimal_dataset["states"] = np.concatenate(suboptimal_dataset["states"]).astype(np.float32)
    suboptimal_dataset["actions"] = np.concatenate(suboptimal_dataset["actions"]).astype(np.float32)
    suboptimal_dataset["next_states"] = np.concatenate(suboptimal_dataset["next_states"]).astype(np.float32)
    suboptimal_dataset["dones"] = np.concatenate(suboptimal_dataset["dones"]).astype(np.float32)
    suboptimal_dataset["rewards"] = np.concatenate(suboptimal_dataset["rewards"]).astype(np.float32)

    union_dataset["init_states"] = np.concatenate([suboptimal_dataset["init_states"], expert_dataset["init_states"]]).astype(np.float32)
    union_dataset["states"] = np.concatenate([suboptimal_dataset["states"], expert_dataset["states"]]).astype(np.float32)
    union_dataset["actions"] = np.concatenate([suboptimal_dataset["actions"], expert_dataset["actions"]]).astype(np.float32)
    union_dataset["next_states"] = np.concatenate([suboptimal_dataset["next_states"], expert_dataset["next_states"]]).astype(np.float32)
    union_dataset["dones"] = np.concatenate([suboptimal_dataset["dones"], expert_dataset["dones"]]).astype(np.float32)
    union_dataset["rewards"] = np.concatenate([suboptimal_dataset["rewards"], expert_dataset["rewards"]]).astype(np.float32)
    print('# of expert demonstraions: {}'.format(expert_dataset["states"].shape[0]))
    print('# of imperfect demonstraions: {}'.format(suboptimal_dataset["states"].shape[0]))
     # normalize
    shift = -np.mean(suboptimal_dataset["states"], 0)
    scale = 1.0 / (np.std(suboptimal_dataset["states"], 0) + 1e-3)
    params = {"shift": shift, "scale": scale}
    file_path = 'shift_scale_params.pkl'  # 文件路径

    with open(file_path, 'wb') as f:
        pickle.dump(params, f)
    print(f"Params saved to {file_path}")
    union_init_states = (union_dataset["init_states"] + shift) * scale
    expert_dataset["states"] = (expert_dataset["states"] + shift) * scale
    expert_dataset["next_states"] = (expert_dataset["next_states"] + shift) * scale
    union_dataset["states"] = (union_dataset["states"] + shift) * scale
    union_dataset["next_states"] = (union_dataset["next_states"] + shift) * scale
    env = make_env(args.env_id,normalize=True, shift=shift, scale=scale)
    env_test = env
    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_shape=env.observation_space.shape 
    if 'Ant' in env.spec.id: 
        state_shape = (27,)
    algo_type=True

    if args.algo == 'ilmar':
        algo_type=False
        if args.use_union:
            algo = ALGOS[args.algo](
                buffer_exp=expert_dataset,
                buffer_union=union_dataset,
                state_shape=state_shape,
                action_shape=env.action_space.shape,
                device=device,
                seed=args.seed,
                batch_size= args.batch_size,
                config = vars(args)
            )
        else:
            algo = ALGOS[args.algo](
                buffer_exp=expert_dataset,
                buffer_union=suboptimal_dataset,
                state_shape=state_shape,
                action_shape=env.action_space.shape,
                device=device,
                seed=args.seed,
                batch_size= args.batch_size,
                config = vars(args)
            )           
    else:
        algo = ALGOS[args.algo](
            buffer_exp=expert_dataset,
            state_shape=state_shape,
            action_shape=env.action_space.shape,
            device=device,
            seed=args.seed,
            rollout_length=args.rollout_length,
        )

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(
        'logs', args.env_id, args.algo, f'seed{args.seed}-{time}')
    total_dir = os.path.join(
        'logs', args.env_id)

    if args.algo == "ilmar":
        algo_type=False
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_epi,
        seed=args.seed,
        algo_type=algo_type,
        total_dir =  total_dir,
        config = vars(args)
                     #启动问题
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # required
    p.add_argument('--buffer_exp', type=str, required=False,
                   help='path to the demonstration buffer')
    p.add_argument('--buffer_union', type=str, required=False,
                   help='path to the union demonstration buffer')
    p.add_argument('--env-id', type=str, required=True,
                   help='name of the environment')
    p.add_argument('--algo', type=str, required=True,
                   help='Imitation Learning algorithm to be trained')

    # custom
    p.add_argument('--rollout-length', type=int, default=10000,
                   help='rollout length of the buffer')
    p.add_argument('--num-steps', type=int, default=10**6,
                   help='number of steps to train')
    p.add_argument('--eval-interval', type=int, default=10**4,
                   help='time interval between evaluations')

    # default
    p.add_argument('--num-eval-epi', type=int, default=10,
                   help='number of episodes for evaluation')
    p.add_argument('--seed', type=int, default=0,
                   help='random seed')
    p.add_argument('--label', type=float, default=0.05,
                   help='ratio of labeled data')
    p.add_argument('--batch_size', type=int, default=512,
                   help='batch_size')
    p.add_argument('--dirname', type=str, default="datasets",
                   help='dirname')
    p.add_argument('--plots_dir', type=str, default="plots",
                   help='plots_dir')
    p.add_argument('--save_best', type=bool, default=False,
                   help='save')                
    p.add_argument('--expert_num_trajectories', type=int, default=1,
                   help='expert_num_trajectories')
    p.add_argument('--suboptimal_dataset_names', type=str, nargs='+', default=["expert","medium1","medium2","medium3","medium4"],
                   help='batch_size')
    p.add_argument('--suboptimal_num_trajs', type=int, nargs='+',default=[400,400,400,400,400],
                   help='batch_size')
    p.add_argument('--use_union', type=bool, default=True,
                   help='use_union')
    p.add_argument('--phi', type=int, default=0,
                   help='phi'),        
    p.add_argument('--alpha', type=float, default=0.0,
                   help='phi')    
    p.add_argument('--beta', type=float, default=1.0,
                   help='phi')      
    args = p.parse_args()
    wandb.init(project="ILMAR", entity="", 
                    name=f"{args.algo}_{args.env_id}_seed_{args.seed}")
    wandb.config.update(vars(args))
    run(args)
    wandb.finish()
