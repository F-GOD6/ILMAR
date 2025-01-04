import gymnasium as gym 
import numpy as np
import os
from urllib import request
import h5py
gym.logger.set_level(40)

KEYS = ['observations', 'actions', 'rewards', 'terminals']
def get_dataset(dirname, env_id, dataname, num_trajectories, start_idx=0, dtype=np.float32):
    MAX_EPISODE_STEPS = 1000
    original_env_id = env_id
    if env_id in ['Hopper-v2', 'Walker2d-v2', 'HalfCheetah-v2', 'Ant-v2']:
        env_id = env_id.split('-v2')[0].lower()

    filename = f'{original_env_id}/{dataname}'
    filepath = os.path.join(dirname, filename + '.hdf5')

    def get_keys(h5file):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys

    dataset_file = h5py.File(filepath, 'r')
    dataset_keys = KEYS
    use_timeouts = False
    use_next_obs = False
    if 'timeouts' in get_keys(dataset_file):
        if 'timeouts' not in dataset_keys:
            dataset_keys.append('timeouts')
        use_timeouts = True
    dataset = {k: dataset_file[k][:] for k in dataset_keys}
    dataset_file.close()
    N = dataset['observations'].shape[0]
    init_obs_, init_action_, obs_, action_, next_obs_, rew_, done_, reward_ = [], [], [], [], [], [], [],[]
    episode_steps = 0
    num_episodes = 0
    for i in range(N - 1):
        if env_id == 'ant':
            obs = dataset['observations'][i][:27]
            if use_next_obs:
                next_obs = dataset['next_observations'][i][:27]
            else:
                next_obs = dataset['observations'][i + 1][:27]
        else:
            obs = dataset['observations'][i]
            if use_next_obs:
                next_obs = dataset['next_observations'][i]
            else:
                next_obs = dataset['observations'][i + 1]
        action = dataset['actions'][i]
        done_bool = bool(dataset['terminals'][i])
        reward = dataset['rewards'][i]
        if use_timeouts:
            is_final_timestep = dataset['timeouts'][i]
        else:
            is_final_timestep = (episode_steps == MAX_EPISODE_STEPS - 1)

        if is_final_timestep:
            episode_steps = 0
            num_episodes += 1
            if num_episodes >= num_trajectories + start_idx:
                break
            continue

        if num_episodes >= start_idx:
            if episode_steps == 0:
                init_obs_.append(obs)
            obs_.append(obs)
            next_obs_.append(next_obs)
            action_.append(action)
            done_.append(done_bool)
            reward_.append(reward)

        episode_steps += 1
        if done_bool:
            episode_steps = 0
            num_episodes += 1
            if num_episodes >= num_trajectories + start_idx:
                break

    env = gym.make(original_env_id)
    if env.action_space.dtype == int:
        action_ = np.eye(env.action_space.n)[np.array(action_, dtype=np.int)]  # integer to one-hot encoding

    print(f'{num_episodes} trajectories are sampled')
    def package_data(init_obs_, obs_, action_, next_obs_, done_, reward_, dtype):
        data_dict = {
            'init_states': np.array(init_obs_, dtype=dtype),
            'states': np.array(obs_, dtype=dtype),
            'actions': np.array(action_, dtype=dtype),
            'next_states': np.array(next_obs_, dtype=dtype),
            'dones': np.array(done_, dtype=dtype),
            'rewards': np.array(reward_, dtype=dtype)
        }
        return data_dict

# 调用函数并返回字典
    expert_dataset = package_data(init_obs_, obs_, action_, next_obs_, done_, reward_, dtype)
    return  expert_dataset

class NormalizedEnv(gym.Wrapper):
    """
    Environment with action space normalized

    Parameters
    ----------
    env: gym.wrappers.TimeLimit
    """

    def __init__(self, env: gym.wrappers.TimeLimit, normalize=False, shift=0, scale=1):
        gym.Wrapper.__init__(self, env)
        self.action_scale = (env.action_space.high - env.action_space.low) / 2.
        self.action_space.high /= self.action_scale
        self.action_space.low /= self.action_scale
        self.normalize = normalize
        if normalize:
            self.state_shift=shift
            self.state_scale=scale


    def step(self, action: np.array):
        # next_state, reward, done, info = self.env.step(action * self.action_scale) 
        next_state, reward, terminated, truncated,info=self.env.step(action * self.action_scale)
        if 'Ant' in self.env.spec.id:
            next_state = next_state[:27]
        # return next_state, reward, done, info
        if self.normalize:
            next_state = (next_state + self.state_shift) * self.state_scale
        return next_state, reward, terminated, truncated,info
    def reset(self, *, seed=0):
        x=self.env.reset(seed=seed)
        if self.normalize:
            if 'Ant' in self.env.spec.id:
                return (x[0][:27] + self.state_shift) * self.state_scale, x[1:]
            return (x[0] + self.state_shift) * self.state_scale, x[1:]
        else:
            if 'Ant' in self.env.spec.id:
                return x[0][:27],x[1:]

            return x
    @property
    def max_episode_steps(self):
        return self.env._max_episode_steps


def make_env(env_id: str, normalize=False, shift=0, scale=1 ) -> NormalizedEnv:
    """
    Make normalized environment

    Parameters
    ----------
    env_id: str
        id of the env

    Returns
    -------
    env: NormalizedEnv
        normalized environment
    """
    return NormalizedEnv(gym.make(env_id),normalize, shift, scale)
    return NormalizedEnv(gym.make(env_id,render_mode="rgb_array"))
#,use_contact_forces=True
