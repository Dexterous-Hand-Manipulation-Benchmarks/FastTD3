from dataclasses import dataclass, replace

import torch
import numpy as np

from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.benchmark_task import RunChoices, benchmark_spec
from dexmachina.envs.constructors import get_all_env_cfg
from dexmachina.envs.spec import EnvSpec
from dexmachina.envs.vec_env import DexMachinaVecEnv

ENV_NAME_PREFIX = "dexmachina"
ENV_NAME_SEPARATOR = "|"
# Caps the evaluation envs to avoid CUDA out-of-memory errors next to the training env.
MAX_EVAL_ENVS = 32


def parse_env_name(env_name: str) -> tuple[str, str]:
    """Hand and clip from ``"dexmachina|<hand>|<clip>"``."""
    parts = env_name.split(ENV_NAME_SEPARATOR)
    if len(parts) != 3 or parts[0] != ENV_NAME_PREFIX:
        raise ValueError(
            f"env_name {env_name!r} is not 'dexmachina|<hand>|<clip>', e.g. 'dexmachina|allegro_hand|box-30-230'. "
            "The hand and clip decide the task, so there is no default for them."
        )
    return parts[1], parts[2]


def benchmark_env_spec(env_name: str, seed: int, num_envs: int) -> EnvSpec:
    """The shared benchmark task for this run; FastTD3 chooses only hand, clip, seed and num_envs."""
    hand, clip = parse_env_name(env_name)
    return benchmark_spec(RunChoices(hand=hand, clip=clip, seed=seed, num_envs=num_envs))


def eval_env_kwargs(spec: EnvSpec, device: str) -> dict:
    """BaseEnv kwargs for evaluating on the spec's task: no early resets, randomization, assistance or curriculum."""
    env_kwargs = get_all_env_cfg(spec.to_args(), device=device)
    env_kwargs["env_cfg"]["is_eval"] = True
    env_kwargs["env_cfg"]["early_reset_threshold"] = 0.0
    env_kwargs["rand_cfg"]["randomize"] = False
    env_kwargs["env_cfg"]["rand_init_ratio"] = 0.0
    for cfg in env_kwargs["object_cfgs"].values():
        cfg["actuated"] = False
    env_kwargs.pop("curriculum_cfg", None)
    return env_kwargs


@dataclass(frozen=True)
class DexMachinaEnvs:
    train: "GenesisFastTD3Wrapper"
    evaluation: "GenesisFastTD3Wrapper"
    render: "GenesisFastTD3Wrapper"


def build_dexmachina_envs(spec: EnvSpec, device: str, num_eval_envs: int) -> DexMachinaEnvs:
    """Training env on the spec, plus evaluation and single-env render envs on the same task."""
    train = GenesisFastTD3Wrapper(DexMachinaVecEnv.from_spec(spec, device, reward_overrides={}), rl_device=device)
    eval_spec = replace(spec, num_envs=min(num_eval_envs, MAX_EVAL_ENVS))
    eval_env = GenesisFastTD3Wrapper(
        DexMachinaVecEnv(BaseEnv(**eval_env_kwargs(eval_spec, device))), rl_device=device
    )
    render_spec = replace(spec, num_envs=1, record_video=True)
    render = GenesisFastTD3Wrapper(
        DexMachinaVecEnv(BaseEnv(**eval_env_kwargs(render_spec, device))), rl_device=device
    )
    if render.base_env._floating_camera is None:
        raise RuntimeError(
            "The DexMachina render env was built without a camera, so rendered videos would be black. "
            "It is built with record_video=True; check how get_all_env_cfg sets up the camera."
        )
    return DexMachinaEnvs(train=train, evaluation=eval_env, render=render)


def advance_curriculum(envs: "GenesisFastTD3Wrapper", global_step: int, total_timesteps: int, current_epoch: int) -> int:
    """Spreads the spec's max_epochs evenly over the run; returns the curriculum epoch now in effect."""
    epoch = (global_step * envs.spec.max_epochs) // total_timesteps
    if epoch > current_epoch:
        envs.base_env.set_curriculum(epoch)
        return epoch
    return current_epoch


class GenesisFastTD3Wrapper:
    """
    Wraps a DexMachinaVecEnv to be natively compatible with FastTD3.
    """

    def __init__(self, vec_env: DexMachinaVecEnv, rl_device="cuda:0"):
        self.vec_env = vec_env
        self.base_env = vec_env.env
        self.spec = vec_env.spec
        self.sim_device = rl_device
        self._clip_actions = self.base_env.action_clip

        # FastTD3 explicitly looks for these attributes instead of gym spaces
        self.num_envs = self.base_env.num_envs
        self.num_obs = self.base_env.obs_dim
        self.num_actions = self.base_env.action_dim
        self.asymmetric_obs = False
        self.max_episode_steps = self.base_env.max_episode_length

    def reset(self):
        """
        FastTD3's reset() expects ONLY the observation tensor to be returned.
        """
        obs, _ = self.vec_env.reset()
        return obs.to(self.sim_device, dtype=torch.float32)

    def step(self, actions):
        """
        FastTD3 expects step to return (obs, rewards, dones, infos).
        It strictly relies on infos["time_outs"] and infos["observations"]["raw"]["obs"].
        """
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.sim_device)
        else:
            actions = actions.to(self.sim_device)

        # Handle flattened or 1D actions
        if actions.ndim == 1:
            batch_size = actions.numel() // self.num_actions
            actions = actions.view(batch_size, self.num_actions)

        # Safety fallback
        if actions.shape[0] == 1 and self.num_envs > 1:
            actions = actions.repeat(self.num_envs, 1)

        assert actions.shape[0] == self.num_envs, f"Expected action batch size {self.num_envs}, got {actions.shape[0]}"

        # Clamp actions based on environment limits
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)

        obs, rew, terminated, truncated, info = self.vec_env.step(actions)

        # NaN safeguards against simulation divergence
        if torch.isnan(obs).any():
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(rew).any():
            rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)

        dones = terminated | truncated

        infos = {}
        # The vec env reports only chunk time-outs as truncated; every other episode end is terminated.
        infos["time_outs"] = truncated
        # The observation an ended episode had before the same-step reset.
        infos["observations"] = {"raw": {"obs": info["final_obs"]}}
        infos["episode_info"] = info["log"]

        # Success and Tracking Error Metrics for Online Validation
        rew_dict = self.base_env.rew_dict
        pos_dist = rew_dict.get("pos_dist", None)
        rot_dist = rew_dict.get("rot_dist", None)
        arti_dist = rew_dict.get("arti_dist", None)
        if pos_dist is not None and rot_dist is not None:
            # 2cm threshold: pos error < 0.02m and rot error < 0.20 rad (~11.5 deg)
            succ_2cm = (pos_dist < 0.02) & (rot_dist < 0.20)
            # 5cm threshold: pos error < 0.05m and rot error < 0.35 rad (~20 deg)
            succ_5cm = (pos_dist < 0.05) & (rot_dist < 0.35)
            if arti_dist is not None:
                succ_2cm = succ_2cm & (arti_dist < 0.10)
                succ_5cm = succ_5cm & (arti_dist < 0.15)
            infos["success_2cm"] = succ_2cm
            infos["success_5cm"] = succ_5cm
            infos["pos_dist"] = pos_dist
            infos["rot_dist"] = rot_dist
            if arti_dist is not None:
                infos["arti_dist"] = arti_dist

        return obs, rew, dones, infos

    def render(self):
        """
        Renders the environment using Dexmachina's natively initialized floating camera.
        """
        if self.base_env._floating_camera is not None:
            try:
                # Force Genesis to render the frame
                render_out = self.base_env._floating_camera.render()
                image = render_out[0] if isinstance(render_out, tuple) else render_out

                # Move tensor to CPU numpy array
                if isinstance(image, torch.Tensor):
                    image = image.cpu().numpy()

                # Drop the alpha channel (RGBA -> RGB) for WandB compatibility
                if image.shape[-1] == 4:
                    image = image[..., :3]

                # Ensure uint8
                if image.dtype != np.uint8:
                    if image.max() <= 1.01:
                        image = image * 255.0
                    image = np.clip(image, 0, 255).astype(np.uint8)

                # Add batch dimension (Batch, Height, Width, Channels)
                if image.ndim == 3:
                    image = image[np.newaxis, ...]

                return image
            except Exception as e:
                print(f"[DEBUG] Camera rendering failed: {e}")
        else:
            # Only print once to avoid spamming the console
            if not hasattr(self, "_camera_warning_printed"):
                print("[DEBUG] self.base_env._floating_camera is missing: the env was built without record_video.")
                self._camera_warning_printed = True

        # Ultimate Fallback (Dexmachina cameras default to 160x160 resolution)
        return np.zeros((1, 160, 160, 3), dtype=np.uint8)

    def close(self):
        self.vec_env.close()
