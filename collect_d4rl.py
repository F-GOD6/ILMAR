import os
import argparse
import torch

from ilmar.env import make_env
from ilmar.algo.algo import EXP_ALGOS
from ilmar.utils import collect_d4rl
import pickle
def get_device():
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    
    if visible_devices is not None and torch.cuda.is_available():
        print(f"Using GPU: {visible_devices}")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    
    return device
def run(args):
    """Collect demonstrations"""
    env = make_env(args.env_id)
    device = get_device()
    state_shape=env.observation_space.shape 
    if 'Ant' in env.spec.id: 
        state_shape = (111,)
    algo = EXP_ALGOS[args.algo](
        state_shape=state_shape,
        action_shape=env.action_space.shape,
        device=device,
        path=args.weight
    )

    # file_path = 'shift_scale_params.pkl'  # 

    # with open(file_path, 'rb') as f:
    #     params = pickle.load(f)

    # shift = params["shift"]
    # scale = params["scale"]
    mean_return = collect_d4rl(
        env=env,
        algo=algo,
        buffer_size=args.buffer_size,
        device=device,
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed,
        env_id=args.env_id,
        # shift=shift,
        # scale=scale
    )



if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # required
    p.add_argument('--weight', type=str, required=True,
                   help='path to the well-trained weights of the agent')
    p.add_argument('--env-id', type=str, required=True,
                   help='name of the environment')
    p.add_argument('--algo', type=str, required=True,
                   help='name of the well-trained agent')

    # default
    p.add_argument('--buffer-size', type=int, default=40000,
                   help='size of the buffer')
    p.add_argument('--std', type=float, default=0.01,
                   help='standard deviation add to the policy')
    p.add_argument('--p-rand', type=float, default=0.0,
                   help='with probability of p_rand, the policy will act randomly')
    p.add_argument('--seed', type=int, default=0,
                   help='random seed')
    p.add_argument("--save-path",type=str,required=False,default=None,
                    help="save path")

    args = p.parse_args()
    run(args)
