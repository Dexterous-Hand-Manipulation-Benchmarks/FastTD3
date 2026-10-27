import torch
import numpy as np


def apply_dexmachina_env_name(args):
    """Parse ``args.env_name`` ("dexmachina|<hand>|<clip>") into ``args.hand`` / ``args.clip``.

    Also maps tyro-friendly defaults back to the ``None`` values dexmachina expects.
    Returns ``(hand_name, clip_name)``.
    """
    parts = args.env_name.split("|")
    hand_name = parts[1] if len(parts) > 1 else "allegro_hand"
    clip_name = parts[2] if len(parts) > 2 else "box-30-230"
    args.hand = hand_name
    args.clip = clip_name

    # Handle 'None' defaults that tyro doesn't parse cleanly
    if args.n_envs_per_row == 0:
        args.n_envs_per_row = None
    if args.color_object == "":
        args.color_object = None
    return hand_name, clip_name


class GenesisFastTD3Wrapper:
    """
    Wraps Dexmachina's BaseEnv to be natively compatible with FastTD3.
    """

    def __init__(self, env, rl_device="cuda:0"):
        self.env = env
        self.sim_device = rl_device
        self._clip_actions = self.env.action_clip

        # FastTD3 explicitly looks for these attributes instead of gym spaces
        self.num_envs = self.env.num_envs
        self.num_obs = self.env.obs_dim
        self.num_actions = self.env.action_dim
        self.asymmetric_obs = False

        # Try to pull max_episode_length if Dexmachina defines it, otherwise default to 1000
        self.max_episode_steps = getattr(self.env, "max_episode_length", 1000)

    def reset(self):
        """
        FastTD3's reset() expects ONLY the observation tensor to be returned.
        """
        obs_dict, extras = self.env.reset()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        # Ensure we are returning a float tensor on the correct device
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, device=self.sim_device, dtype=torch.float32)
        else:
            obs = obs.to(self.sim_device, dtype=torch.float32)

        return obs

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

        # Step the Dexmachina simulator
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        # Cast outputs to standard tensors for FastTD3
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, device=self.sim_device, dtype=torch.float32)
        if not isinstance(rew, torch.Tensor):
            rew = torch.tensor(rew, device=self.sim_device, dtype=torch.float32)
        if not isinstance(terminated, torch.Tensor):
            terminated = torch.tensor(terminated, device=self.sim_device, dtype=torch.bool)
        if not isinstance(truncated, torch.Tensor):
            truncated = torch.tensor(truncated, device=self.sim_device, dtype=torch.bool)

        # NaN safeguards against simulation divergence
        if torch.isnan(obs).any():
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(rew).any():
            rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)

        # FastTD3 expects a unified `dones` tensor
        dones = terminated | truncated

        # --- FastTD3 Info Formatting ---
        infos = {}

        # 1. Truncated timeouts
        infos["time_outs"] = truncated.clone()

        # 2. Terminal Observations.
        # Since GPU simulators typically auto-reset and return the new first-step obs,
        # FastTD3 needs the actual terminal observation to compute accurate bootstrapping.
        # If Dexmachina provides it in `extras`, use it; otherwise fallback to current obs.
        terminal_obs = extras.get("final_obs", extras.get("terminal_observation", obs)).clone()

        infos["observations"] = {
            "raw": {
                "obs": terminal_obs
            }
        }

        # Carry over any logging info
        if "log" in extras:
            infos["episode_info"] = extras["log"]

        # 3. Success and Tracking Error Metrics for Online Validation
        if hasattr(self.env, "rew_dict") and isinstance(self.env.rew_dict, dict):
            pos_dist = self.env.rew_dict.get("pos_dist", None)
            rot_dist = self.env.rew_dict.get("rot_dist", None)
            arti_dist = self.env.rew_dict.get("arti_dist", None)
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
        if hasattr(self.env, "_floating_camera") and self.env._floating_camera is not None:
            try:
                # Force Genesis to render the frame
                render_out = self.env._floating_camera.render()
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
                print("[DEBUG] self.env._floating_camera is missing. Did you pass env.record_video=True?")
                self._camera_warning_printed = True

        # Ultimate Fallback (Dexmachina cameras default to 160x160 resolution)
        return np.zeros((1, 160, 160, 3), dtype=np.uint8)

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()