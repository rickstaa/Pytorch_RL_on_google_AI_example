"""This is the training script we want to perform on the google ai platform. It was
pulled (not forked) from https://github.com/colinskow/move37 together with the
dqn_play script. I pulled the script instead of forking since I did not need the
full repository.

The move37 repository was originally forked from Forked from https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On
"""

from lib import wrappers
from lib import dqn_model
from lib.utils import mkdir
import os
import re
import subprocess

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from google.cloud import storage
from google.api_core.exceptions import NotFound

try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

# Script settings
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_GOAL = 19.5
MAX_FRAMES = 1e4
MAX_FRAMES = 1.2e6

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
REPLAY_MIN_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000

EPSILON_START = 1.0
EPSILON_FINAL = 0.02
EPSILON_DECAY_FRAMES = 10 ** 5

Experience = collections.namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)


class ExperienceBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_states),
        )


class Agent:
    """DQN Agent"""

    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        new_state = new_state

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    """Loss calculation function"""

    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Enable cuda"
    )
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV_NAME,
        help="Name of the environment, default=" + DEFAULT_ENV_NAME,
    )
    parser.add_argument(
        "--reward",
        type=float,
        default=MEAN_REWARD_GOAL,
        help="Mean reward goal to stop training, default=%.2f" % MEAN_REWARD_GOAL,
    )
    parser.add_argument(
        "--frames",
        type=float,
        default=MAX_FRAMES,
        help="Mean reward goal to stop training, default=%.2f" % MAX_FRAMES,
    )
    parser.add_argument(
        "--model-dir", default=None, help="The directory to store the model"
    )
    parser.add_argument(
        "--batch-size", default=BATCH_SIZE, help="The directory to store the model"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # Create Pong environment
    env = wrappers.make_env(args.env)

    # Create directory name for storing the model
    model_dir_name = os.path.join("pong_dqn-" + "batch_size_" + str(args.batch_size))
    tmp_model_folder = os.path.join(model_dir_name, "checkpoints")
    mkdir(".", tmp_model_folder)

    # # Method2: Store model using google.cloud.storage module
    # #  Create storage bucket writer
    # if args.model_dir:
    #     # Retrieve bucket name
    #     bucket_name = [
    #         item
    #         for item in os.path.split(re.sub("gs://", "", args.model_dir))
    #         if item != ""
    #     ][
    #         0
    #     ]  # The bucket name
    #     bucket_rel_path = os.path.join(
    #         *[
    #             item
    #             for item in os.path.split(re.sub("gs://", "", args.model_dir))
    #             if item != ""
    #         ][1:]
    #     )  # The path relative to the bucket name
    #     storage_client = storage.Client()
    #     bucket = storage_client.bucket(bucket_name)

    # Create DQN main and target Networks
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    # Create tensorboard summary writer
    writer = SummaryWriter(
        comment="-" + args.env,
        log_dir=os.path.join(args.model_dir if args.model_dir else ".", model_dir_name),
    )
    print(net)

    # Create Replay buffer and agent
    buffer = ExperienceBuffer(REPLAY_BUFFER_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    # Setup optimizer and initiate loop variables
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    # Training loop
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_FRAMES)
        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:

            # Calculate mean reward and speed
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print(
                "%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s"
                % (frame_idx, len(total_rewards), mean_reward, epsilon, speed)
            )

            # Write learning variabes to tensorboard
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            # Save model if reward is higher than best reward
            if best_mean_reward is None or best_mean_reward < mean_reward:
                tmp_model_file = os.path.join(tmp_model_folder, args.env + "-best.dat")
                torch.save(net.state_dict(), tmp_model_file)
                if best_mean_reward is not None:
                    print(
                        "Best mean reward updated %.3f -> %.3f, model saved"
                        % (best_mean_reward, mean_reward)
                    )
                best_mean_reward = mean_reward
                if args.model_dir:

                    # Method1: Store model using gsutil
                    subprocess.check_call(
                        [
                            "gsutil",
                            "cp",
                            tmp_model_file,
                            os.path.join(args.model_dir, tmp_model_file),
                        ]
                    )

                    # # Method2: Store model using google.cloud.storage module
                    # # NOTE: Service account key required https://cloud.google.com/docs/authentication/production#command-line
                    # try:
                    #     blob = bucket.blob(
                    #         os.path.join(bucket_rel_path, tmp_model_file)
                    #     )
                    #     blob.upload_from_filename(tmp_model_file)
                    # except Exception as e:
                    #     if isinstance(e, NotFound):
                    #         raise Exception(
                    #             "Could not save model as. Supplied Google cloud "
                    #             "bucket does not exists! Shutting down training."
                    #         )
                    #     else:
                    #         raise Exception(
                    #             "Something went wrong while trying to store the model "
                    #             "inside the Google cloud bucket! Shutting down "
                    #             "training."
                    #         )

            # Break loop if reward is high enough or max frames has been reached
            if mean_reward > args.reward or frame_idx <= args.frames:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_MIN_SIZE:
            continue

        # Update target network
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        # Perform back-propogation
        optimizer.zero_grad()
        batch = buffer.sample(args.batch_size)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()

    # Close tensorboard writer
    writer.close()
