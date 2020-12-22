import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
import datetime

from highway_env.envs.intersection_env import IntersectionEnv

from models import MultiLayerPerceptron, DuelingNetwork


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="intersection-v0",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=0.5e-3,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=100000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="highway-env",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=10000,
                         help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=500,
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=64,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--start-e', type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.02,
                        help="the ending epsilon for exploration")
    parser.add_argument('--exploration-fraction', type=float, default=0.3,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument('--learning-starts', type=int, default=1000,
                        help="timestep to start learning")
    parser.add_argument('--train-frequency', type=int, default=1,
                        help="the frequency of training")
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

def one_hot(a, size):
    b = np.zeros((size))
    b[a] = 1
    return b

class ProcessObsInputEnv(gym.ObservationWrapper):
    """
    This wrapper handles inputs from `Discrete` and `Box` observation space.
    If the `env.observation_space` is of `Discrete` type,
    it returns the one-hot encoding of the state
    """
    def __init__(self, env):
        super().__init__(env)
        self.n = None
        if isinstance(self.env.observation_space, Discrete):
            self.n = self.env.observation_space.n
            self.observation_space = Box(0, 1, (self.n,))

    def observation(self, obs):
        if self.n:
            return one_hot(np.array(obs), self.n)
        return obs


class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space[0].low[0][0],
            env.observation_space[0].high[0][0],
            (np.array(env.observation_space[0].shape).prod(), ))
        self.action_space = env.action_space[0]

    # def observation(self, obs):
    #     # obs is a list of obs vectors for all agents in one env -> concatenate them
    #     new_obs = obs.flatten()
    #     return new_obs


# TRY NOT TO MODIFY: setup the environment
TRAIN = True

yyyymmdd = datetime.datetime.today().strftime("%Y_%m_%d")
exp_name = f"Intersection_{args.exp_name}_{args.seed}_"+str(datetime.datetime.today()).split(' ')[1].split('.')[0]
experiment_name = os.path.join(yyyymmdd, exp_name)
data_path = os.path.join('data', experiment_name)
models_path = f"{data_path}/models"

writer = SummaryWriter(f"{data_path}/logs")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode and TRAIN:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=exp_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{exp_name}")

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
device = 'cpu'

##------------------- configs -------------------------
env_config = {
    "id": "intersection-v0",
    "import_module": "highway_env",
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "order": "shuffled"
    },
    "destination": "o1"
}

# env = gym.make(args.gym_id)
env = IntersectionEnv(env_config)
env.config['offscreen_rendering'] = True if TRAIN else False
# env.config["screen_width"] = 1000
# env.config["screen_height"] = 1000
# env.config["duration"] = 300

# env = ObsWrapper(env)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
# env.action_space.seed(args.seed)
# env.observation_space.seed(args.seed)
# respect the default timelimit
# assert isinstance(env.action_space, Discrete), "only discrete action space is supported"
if args.capture_video:
    env = Monitor(env, f'{data_path}/videos')

# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


rb = ReplayBuffer(args.buffer_size)

input_size = np.array(env.observation_space.shape).prod()
output_size = env.action_space.n
# q_network = QNetwork(input_size, output_size).to(device)
# target_network = QNetwork(input_size, output_size).to(device)
# target_network.load_state_dict(q_network.state_dict())

model_config = {
    "layers": [128, 128],
    "in": input_size,
    "out": output_size
}

q_network = MultiLayerPerceptron(model_config).to(device)
target_network = MultiLayerPerceptron(model_config).to(device)
target_network.load_state_dict(q_network.state_dict())


optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

loss_fn = nn.MSELoss()

print(device.__repr__())
print(q_network)

if not os.path.exists(models_path):
    os.mkdir(models_path)

if TRAIN:
    # TRY NOT TO MODIFY: start the game
    obs = env.reset()
    episode_reward = 0

    best_reward = -np.inf

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        eps_duration = args.exploration_fraction*args.total_timesteps
        # eps_duration = 2000
        epsilon = linear_schedule(args.start_e, args.end_e, eps_duration, global_step)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            logits = q_network.forward(obs.reshape((1,)+obs.shape))
            action = torch.argmax(logits, dim=1).tolist()[0]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, _ = env.step(action)
        # env.render()
        episode_reward += reward

        # ALGO LOGIC: training.
        # put data collected with all agents into one common buffer
        rb.put((obs, action, reward, next_obs, done))

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
            with torch.no_grad():
                target_max = torch.max(target_network.forward(s_next_obses), dim=1)[0]
                td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))
            old_val = q_network.forward(s_obs).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
            loss = loss_fn(td_target, old_val)

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)

            # optimize the midel
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
            optimizer.step()

            # save model each time
            torch.save(q_network.state_dict(), models_path + "/q_network.pt")

            # save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(q_network.state_dict(), models_path + "/q_network_best.pt")

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if done: #if one agent is done - reset env
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            print(f"global_step={global_step}, episode_reward={episode_reward}")
            writer.add_scalar("charts/episode_reward", episode_reward, global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)
            obs, episode_reward = env.reset(), 0

    env.close()
    writer.close()

else: #test
    # TRY NOT TO MODIFY: start the game
    obs = env.reset()
    episode_reward = 0

    load_path = '/u/05/aznarr1/unix/Documents/Projectintersection/highway-env-master/scripts/data/2020_10_20/Intersection_dqn_ml_1_20:08:55/models'
    q_network.load_state_dict(torch.load(load_path + "/q_network.pt"))

    for _ in range(30):
        while True:
            logits = q_network.forward(obs.reshape((1,)+obs.shape))
            action = torch.argmax(logits, dim=1).tolist()[0]

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, _ = env.step(action)
            env.render()
            episode_reward += reward

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            if done: #if one agent is done - reset env
                # TRY NOT TO MODIFY: record rewards for plotting purposes
                print(f"episode_reward={episode_reward}")
                obs, episode_reward = env.reset(), 0
                break

    env.close()
    writer.close()
