import os
import argparse
import torch
import numpy as np

from datetime import datetime
from ilmar.env import make_env
from ilmar.buffer import SerializedBuffer
from ilmar.algo.algo import ALGOS
from ilmar.trainer import Trainer


def run(args):
    """Train Imitation Learning algorithms"""
    env = make_env(args.env_id)
    env_test = env
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    buffer_exp = SerializedBuffer(
        path=args.buffer_exp,
        device=device,
        label_ratio=args.label,
        use_mean=args.use_transition
    )



    if args.algo == 'iswbc' or args.algo == 'demodice':
        buffer_union = SerializedBuffer(
        path=args.buffer_union,
        device=device,
        label_ratio=args.label,
        use_mean=args.use_transition
    )
        algo = ALGOS[args.algo](
            buffer_exp=buffer_exp,
            buffer_union=buffer_union,
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=device,
            seed=args.seed,
        )
    elif args.algo == 'ilmar':
        buffer_union = SerializedBuffer(
        path=args.buffer_union,
        device=device,
        label_ratio=args.label,
        use_mean=args.use_transition
    )
        algo = ALGOS[args.algo](
            buffer_exp=buffer_exp,
            buffer_union=buffer_union,
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=device,
            seed=args.seed,
            batch_size= args.batch_size,
        )
    else:
        algo = ALGOS[args.algo](
            buffer_exp=buffer_exp,
            state_shape=env.observation_space.shape,
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
    algo_type=True
    if args.algo=="bc" or args.algo=="iswbc" or args.algo == "ilmar" or args.algo =="demodice" or args.algo=="metaiswbc" or args.algo =="metademodice":
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
        total_dir =  total_dir           
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # required
    p.add_argument('--buffer_exp', type=str, required=True,
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

    p.add_argument('--pre-train', type=int, default=20000000,
                   help='pre-train steps for CAIL')
    p.add_argument('--use-transition', action='store_true', default=False,
                   help='use state transition reward for cail')
    # default
    p.add_argument('--num-eval-epi', type=int, default=10,
                   help='number of episodes for evaluation')
    p.add_argument('--seed', type=int, default=0,
                   help='random seed')
    p.add_argument('--label', type=float, default=0.05,
                   help='ratio of labeled data')
    p.add_argument('--batch_size', type=int, default=256,
                   help='batch_size')

    args = p.parse_args()
    run(args)
