import gymnasium as gym
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
import torch
import numpy as np


class GenesisFlashSACWrapper(VectorEnv):
    """
    Wraps Dexmachina's BaseEnv as a Gymnasium VectorEnv compatible with FlashSAC.
    """

    def __init__(self, env, rl_device="cuda:0"):
        self.env = env
        self.rl_device = rl_device
        self._clip_actions = self.env.action_clip
        self.num_envs = self.env.num_envs

        # Define single and batched spaces
        single_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.obs_dim,)
        )
        single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.action_dim,)
        )

        # Initialize VectorEnv attributes without passing invalid args to super().__init__()
        super().__init__()
        self.num_envs = self.num_envs
        self.single_observation_space = single_obs_space
        self.single_action_space = single_action_space
        self.observation_space = batch_space(single_obs_space, self.num_envs)
        self.action_space = batch_space(single_action_space, self.num_envs)

    def reset(self, seed=None, options=None):
        obs_dict, extras = self.env.reset()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        if extras is None:
            extras = {}

        extras["actor_observation_size"] = (self.env.obs_dim,)
        extras["critic_observation_size"] = (self.env.obs_dim,)
        extras["num_envs"] = self.num_envs

        return obs, extras

    def step(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.rl_device)
        else:
            actions = actions.to(self.rl_device)

        # Handle flattened or 1D actions
        if actions.ndim == 1:
            batch_size = actions.numel() // self.env.action_dim
            actions = actions.view(batch_size, self.env.action_dim)

        # Safety fallback: If FlashSAC passes a single-env action batch (size 1)
        # while num_envs is 1024, repeat it to match Dexmachina's parallel requirement
        if actions.shape[0] == 1 and self.num_envs > 1:
            actions = actions.repeat(self.num_envs, 1)

        assert actions.shape[0] == self.num_envs, f"Expected action batch size {self.num_envs}, got {actions.shape[0]}"

        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        if extras is None:
            extras = {}

        extras["final_obs"] = obs.clone()
        if "log" in extras:
            extras["episode_info"] = extras["log"]

        return obs, rew, terminated, truncated, extras

    def render(self):
        """
        Renders the environment using Dexmachina's natively initialized floating camera.
        """
        if hasattr(self.env, "_floating_camera") and self.env._floating_camera is not None:
            try:
                # Force Genesis to render the frame
                self.env._floating_camera.render()
                image = self.env._floating_camera.rgba

                # Move tensor to CPU numpy array
                if isinstance(image, torch.Tensor):
                    image = image.cpu().numpy()

                # Drop the alpha channel (RGBA -> RGB) for WandB compatibility
                if image.shape[-1] == 4:
                    image = image[..., :3]

                # Add batch dimension for FlashSAC (Batch, Height, Width, Channels)
                if image.ndim == 3:
                    image = image[np.newaxis, ...]

                return image
            except Exception as e:
                print(f"[DEBUG] Camera rendering failed: {e}")
        else:
            # Only print once to avoid spamming the console 200 times
            if not hasattr(self, "_camera_warning_printed"):
                print("[DEBUG] self.env._floating_camera is missing. Did you pass env.record_video=True?")
                self._camera_warning_printed = True

        # Ultimate Fallback (Dexmachina cameras default to 160x160 resolution)
        return np.zeros((1, 160, 160, 3), dtype=np.uint8)

    def close(self):
        pass