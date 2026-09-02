import os
import sys

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import random
import time
import math

# Changed: Imported specific columns to customize the progress bar
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
import wandb
import numpy as np

try:
    # Required for avoiding IsaacGym import error
    import isaacgym
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

from tensordict import TensorDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
LOGGING_DIR = os.path.join(PROJECT_ROOT, "logging")
ENVS_FASTTD3_DIR = os.path.join(PROJECT_ROOT, "envs", "FastTD3")
for p in [LOGGING_DIR, PROJECT_ROOT, ENVS_FASTTD3_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from wandb_logger import UniversalLogger

from fast_td3_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    PerTaskRewardNormalizer,
    SimpleReplayBuffer,
    save_params,
    mark_step,
)
from hyperparams import get_args

torch.set_float32_matmul_precision("high")

try:
    import jax.numpy as jnp
except ImportError:
    pass


def main():
    args = get_args()
    print(args)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"
    checkpoint_dir = f"models/FastTD3/{args.exp_name}/{timestamp}"

    # --- DYNAMIC LOGGER INSTALLATION ---
    tokens = args.env_name.split("|")
    hand_name = getattr(args, "hand", tokens[1] if len(tokens) > 1 else "unknown_hand")
    clip_name = getattr(args, "clip", tokens[2] if len(tokens) > 2 else "unknown_clip")

    UniversalLogger.install(
        algo="fasttd3",
        hand=str(hand_name),
        clip=str(clip_name),
        seed=int(args.seed),
        num_envs=int(args.num_envs),
    )
    # -----------------------------------

    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "mps" if args.cuda and torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    if args.use_wandb:
        wandb_kwargs = {
            "project": args.project,
            "name": run_name,
            "config": vars(args),
            "save_code": True,
        }
        if hasattr(args, "entity") and args.entity:
            wandb_kwargs["entity"] = args.entity
        if hasattr(args, "group_name") and args.group_name:
            wandb_kwargs["group"] = args.group_name
        wandb.init(**wandb_kwargs)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{args.device_rank}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    if args.env_name.startswith("h1hand-") or args.env_name.startswith("h1-"):
        from environments.humanoid_bench_env import HumanoidBenchEnv

        env_type = "humanoid_bench"
        envs = HumanoidBenchEnv(args.env_name, args.num_envs, device=device)
        eval_envs = envs
        render_env = HumanoidBenchEnv(
            args.env_name, 1, render_mode="rgb_array", device=device
        )
    elif args.env_name.startswith("Isaac-"):
        from environments.isaaclab_env import IsaacLabEnv

        env_type = "isaaclab"
        envs = IsaacLabEnv(
            args.env_name,
            device.type,
            args.num_envs,
            args.seed,
            action_bounds=args.action_bounds,
        )
        eval_envs = envs
        render_env = envs
    elif args.env_name.startswith("MTBench-"):
        from environments.mtbench_env import MTBenchEnv

        env_name = "-".join(args.env_name.split("-")[1:])
        env_type = "mtbench"
        envs = MTBenchEnv(env_name, args.device_rank, args.num_envs, args.seed)
        eval_envs = envs
        render_env = envs
        # ... existing HumanoidBench/IsaacLab env setup ...

    elif args.env_name.startswith("dexmachina"):
        import sys
        import os

        # 1. Add specific package paths to avoid folder name confusion
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        genesis_path = os.path.join(repo_root, 'genesis')
        dexmachina_path = os.path.join(repo_root, 'dexmachina')

        # Add root so it can find the 'envs' folder
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # Add specific paths so it finds the real 'genesis' and 'dexmachina' modules
        if genesis_path not in sys.path:
            sys.path.insert(0, genesis_path)
        if dexmachina_path not in sys.path:
            sys.path.insert(0, dexmachina_path)

        # 2. Now initialize Genesis correctly
        import genesis as gs
        gs.init(backend=gs.gpu, logging_level="warning")

        # 3. Import Dexmachina and Wrapper
        from envs.FastTD3.dexmachina_wrapper import GenesisFastTD3Wrapper
        from dexmachina.envs.constructors import get_all_env_cfg
        from dexmachina.envs import BaseEnv

        # Parse the string (e.g., "dexmachina|allegro_hand|box-30-230")
        parts = args.env_name.split("|")
        hand_name = parts[1] if len(parts) > 1 else "allegro_hand"
        clip_name = parts[2] if len(parts) > 2 else "box-30-230"

        # FastTD3 uses device_rank instead of device
        sim_device = f"cuda:{getattr(args, 'device_rank', 0)}"

        # Set values from env_name
        args.hand = hand_name
        args.clip = clip_name

        # Handle 'None' defaults that tyro doesn't parse cleanly
        if args.n_envs_per_row == 0:
            args.n_envs_per_row = None
        if args.color_object == "":
            args.color_object = None

        dex_args = args

        print(f"[INFO] Building Dexmachina env: {hand_name} / {clip_name}")

        # FastTD3 uses device_rank instead of device
        sim_device = f"cuda:{getattr(args, 'device_rank', 0)}"

        env_kwargs = get_all_env_cfg(dex_args, device=sim_device)
        base_env = BaseEnv(**env_kwargs)

        # Wrap it for FastTD3
        envs = GenesisFastTD3Wrapper(base_env, rl_device=sim_device)
        env_type = "dexmachina"

        print(f"[INFO] Building Dexmachina EVALUATION env (Unassisted True Physics)...")
        original_num_envs = args.num_envs
        # Cap the evaluation environments to 64 to prevent CUDA OOM errors
        eval_num = min(getattr(args, "num_eval_envs", 32), 32)
        dex_args.num_envs = eval_num

        eval_env_kwargs = get_all_env_cfg(dex_args, device=sim_device)
        eval_env_kwargs['env_cfg']['is_eval'] = True
        eval_env_kwargs['env_cfg']['early_reset_threshold'] = 0.0
        eval_env_kwargs['rand_cfg']['randomize'] = False
        eval_env_kwargs['env_cfg']['rand_init_ratio'] = 0.0
        for name, cfg in eval_env_kwargs['object_cfgs'].items():
            cfg['actuated'] = False
        eval_env_kwargs.pop("curriculum_cfg", None)

        eval_base_env = BaseEnv(**eval_env_kwargs)
        eval_envs = GenesisFastTD3Wrapper(eval_base_env, rl_device=sim_device)

        print(f"[INFO] Building Dexmachina RENDER env (Unassisted True Physics)...")
        dex_args.num_envs = 1  # We only need 1 environment to record a video!

        render_env_kwargs = get_all_env_cfg(dex_args, device=sim_device)
        render_env_kwargs['env_cfg']['is_eval'] = True
        render_env_kwargs['env_cfg']['early_reset_threshold'] = 0.0
        render_env_kwargs['rand_cfg']['randomize'] = False
        render_env_kwargs['env_cfg']['rand_init_ratio'] = 0.0
        for name, cfg in render_env_kwargs['object_cfgs'].items():
            cfg['actuated'] = False
        render_env_kwargs.pop("curriculum_cfg", None)

        render_base_env = BaseEnv(**render_env_kwargs)
        render_env = GenesisFastTD3Wrapper(render_base_env, rl_device=sim_device)
        
        args.num_envs = original_num_envs

        try:
            final_reward_config = envs.env.reward_module.cfg
            # Log it to WandB
            wandb.config.update({"dex_rewards": final_reward_config}, allow_val_change=True)
            print("[INFO] Successfully logged Dexmachina reward config to WandB.")

        except AttributeError as e:
            print(f"[WARNING] Could not find reward config to log: {e}")

    else:
        from environments.mujoco_playground_env import make_env

        # TODO: Check if re-using same envs for eval could reduce memory usage
        env_type = "mujoco_playground"
        envs, eval_envs, render_env = make_env(
            args.env_name,
            args.seed,
            args.num_envs,
            args.num_eval_envs,
            args.device_rank,
            use_tuned_reward=args.use_tuned_reward,
            use_domain_randomization=args.use_domain_randomization,
            use_push_randomization=args.use_push_randomization,
        )

    n_act = envs.num_actions
    n_obs = envs.num_obs if type(envs.num_obs) == int else envs.num_obs[0]
    if envs.asymmetric_obs:
        n_critic_obs = (
            envs.num_privileged_obs
            if type(envs.num_privileged_obs) == int
            else envs.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs
    action_low, action_high = -1.0, 1.0

    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(
            shape=n_critic_obs, device=device
        )
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()

    if args.reward_normalization:
        if env_type in ["mtbench"]:
            reward_normalizer = PerTaskRewardNormalizer(
                num_tasks=envs.num_tasks,
                gamma=args.gamma,
                device=device,
                g_max=min(abs(args.v_min), abs(args.v_max)),
            )
        else:
            reward_normalizer = RewardNormalizer(
                gamma=args.gamma,
                device=device,
                g_max=min(abs(args.v_min), abs(args.v_max)),
            )
    else:
        reward_normalizer = nn.Identity()

    actor_kwargs = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_envs": args.num_envs,
        "device": device,
        "init_scale": args.init_scale,
        "hidden_dim": args.actor_hidden_dim,
        "std_min": args.std_min,
        "std_max": args.std_max,
    }
    critic_kwargs = {
        "n_obs": n_critic_obs,
        "n_act": n_act,
        "num_atoms": args.num_atoms,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "hidden_dim": args.critic_hidden_dim,
        "device": device,
    }

    if env_type == "mtbench":
        actor_kwargs["n_obs"] = n_obs - envs.num_tasks + args.task_embedding_dim
        critic_kwargs["n_obs"] = n_critic_obs - envs.num_tasks + args.task_embedding_dim
        actor_kwargs["num_tasks"] = envs.num_tasks
        actor_kwargs["task_embedding_dim"] = args.task_embedding_dim
        critic_kwargs["num_tasks"] = envs.num_tasks
        critic_kwargs["task_embedding_dim"] = args.task_embedding_dim

    if args.agent == "fasttd3":
        if env_type in ["mtbench"]:
            from fast_td3 import MultiTaskActor, MultiTaskCritic

            actor_cls = MultiTaskActor
            critic_cls = MultiTaskCritic
        else:
            from fast_td3 import Actor, Critic

            actor_cls = Actor
            critic_cls = Critic

        actor_kwargs.update(
            {
                "sim_type": args.sim_type,
                "sim_dimension": args.sim_dimension,
                "seq_len": args.actor_seq_len,
            }
        )
        critic_kwargs.update(
            {
                "sim_type": args.sim_type,
                "sim_dimension": args.sim_dimension,
                "seq_len": args.critic_seq_len,
            }
        )

        print("Using FastTD3")
    elif args.agent == "fasttd3_simbav2":
        if args.sim_type:
            raise ValueError("SimNorm options are only supported with agent='fasttd3'")

        if env_type in ["mtbench"]:
            from fast_td3_simbav2 import MultiTaskActor, MultiTaskCritic

            actor_cls = MultiTaskActor
            critic_cls = MultiTaskCritic
        else:
            from fast_td3_simbav2 import Actor, Critic

            actor_cls = Actor
            critic_cls = Critic

        print("Using FastTD3 + SimbaV2")
        actor_kwargs.pop("init_scale")
        actor_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.actor_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.actor_hidden_dim),
                "alpha_init": 1.0 / (args.actor_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.actor_hidden_dim),
                "expansion": 4,
                "c_shift": 3.0,
                "num_blocks": args.actor_num_blocks,
            }
        )
        critic_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.critic_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.critic_hidden_dim),
                "alpha_init": 1.0 / (args.critic_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.critic_hidden_dim),
                "num_blocks": args.critic_num_blocks,
                "expansion": 4,
                "c_shift": 3.0,
            }
        )
    else:
        raise ValueError(f"Agent {args.agent} not supported")

    actor = actor_cls(**actor_kwargs)

    if env_type in ["mtbench"]:
        # Python 3.8 doesn't support 'from_module' in tensordict
        policy = actor.explore
    else:
        from tensordict import from_module

        actor_detach = actor_cls(**actor_kwargs)
        # Copy params to actor_detach without grad
        from_module(actor).data.to_module(actor_detach)
        policy = actor_detach.explore

    qnet = critic_cls(**critic_kwargs)
    qnet_target = critic_cls(**critic_kwargs)
    qnet_target.load_state_dict(qnet.state_dict())

    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(args.critic_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(args.actor_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )

    # Add learning rate schedulers
    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.critic_learning_rate_end, device=device),
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.actor_learning_rate_end, device=device),
    )

    rb = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        asymmetric_obs=envs.asymmetric_obs,
        playground_mode=env_type == "mujoco_playground",
        n_steps=args.num_steps,
        gamma=args.gamma,
        device=device,
    )

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip

    def evaluate():
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=device)
        episode_lengths = torch.zeros(num_eval_envs, device=device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

        succ_2cm_list = []
        succ_5cm_list = []
        pos_dist_list = []
        rot_dist_list = []

        if env_type == "isaaclab":
            obs = eval_envs.reset(random_start_init=False)
        else:
            obs = eval_envs.reset()

        # Run for a fixed number of steps
        for i in range(eval_envs.max_episode_steps):
            with torch.no_grad(), autocast(
                    device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                obs = normalize_obs(obs, update=False)
                actions = actor(obs)

            next_obs, rewards, dones, infos = eval_envs.step(actions.float())

            if env_type == "mtbench":
                # We only report success rate in MTBench evaluation
                rewards = (
                    infos["episode"]["success"].float() if "episode" in infos else 0.0
                )
            elif env_type == "dexmachina":
                if "success_2cm" in infos and isinstance(infos["success_2cm"], torch.Tensor):
                    succ_2cm_list.append(infos["success_2cm"].float())
                if "success_5cm" in infos and isinstance(infos["success_5cm"], torch.Tensor):
                    succ_5cm_list.append(infos["success_5cm"].float())
                if "pos_dist" in infos and isinstance(infos["pos_dist"], torch.Tensor):
                    pos_dist_list.append(infos["pos_dist"].float())
                if "rot_dist" in infos and isinstance(infos["rot_dist"], torch.Tensor):
                    rot_dist_list.append(infos["rot_dist"].float())

            episode_returns = torch.where(
                ~done_masks, episode_returns + rewards, episode_returns
            )
            episode_lengths = torch.where(
                ~done_masks, episode_lengths + 1, episode_lengths
            )
            if env_type == "mtbench" and "episode" in infos:
                dones = dones | infos["episode"]["success"]
            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs

        eval_out = {
            "eval_avg_return": episode_returns.mean().item(),
            "eval_avg_length": episode_lengths.mean().item(),
        }
        if succ_2cm_list:
            eval_out["eval/success_2cm"] = torch.stack(succ_2cm_list).mean().item()
        if succ_5cm_list:
            eval_out["eval/success_5cm"] = torch.stack(succ_5cm_list).mean().item()
        if pos_dist_list:
            eval_out["eval/pos_dist"] = torch.stack(pos_dist_list).mean().item()
        if rot_dist_list:
            eval_out["eval/rot_dist"] = torch.stack(rot_dist_list).mean().item()

        return eval_out

    def render_with_rollout():
        # Quick rollout for rendering
        if env_type == "humanoid_bench":
            obs = render_env.reset()
            renders = [render_env.render()]
        elif env_type in ["isaaclab", "mtbench"]:
            raise NotImplementedError(
                "We don't support rendering for IsaacLab and MTBench environments"
            )
        elif env_type == "dexmachina":
            # 1. Reset the environment and safely capture the initial observation (obs)
            reset_out = render_env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            # 2. Grab the first RGB frame from the floating camera
            base = getattr(render_env, "env", getattr(render_env, "base_env", render_env))
            render_out = base._floating_camera.render()
            frame = render_out[0] if isinstance(render_out, tuple) else render_out
            renders = [frame]
        else:
            obs = render_env.reset()
            render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            renders = [render_env.state]

        # 3. Rollout loop
        for i in range(render_env.max_episode_steps):
            with torch.no_grad(), autocast(
                    device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                # Now `obs` exists and can be normalized successfully!
                obs = normalize_obs(obs, update=False)
                actions = actor(obs)

            next_obs, _, done, _ = render_env.step(actions.float())

            if env_type == "mujoco_playground":
                render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])

            if i % 2 == 0:
                if env_type == "humanoid_bench":
                    renders.append(render_env.render())
                elif env_type == "dexmachina":
                    # Capture subsequent frames during the rollout
                    base = getattr(render_env, "env", getattr(render_env, "base_env", render_env))
                    render_out = base._floating_camera.render()
                    frame = render_out[0] if isinstance(render_out, tuple) else render_out
                    renders.append(frame)
                else:
                    renders.append(render_env.state)

            if done.any():
                break
            # Advance to the next observation
            obs = next_obs
        if env_type == "mujoco_playground":
            renders = render_env.render_trajectory(renders)

        return renders

    def update_main(data, logs_dict):
        with autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            if envs.asymmetric_obs:
                critic_observations = data["critic_observations"]
                next_critic_observations = data["next"]["critic_observations"]
            else:
                critic_observations = observations
                next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            if args.disable_bootstrap:
                bootstrap = (~dones).float()
            else:
                bootstrap = (truncations | ~dones).float()

            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(policy_noise).clamp(
                -noise_clip, noise_clip
            )

            next_state_actions = (actor(next_observations) + clipped_noise).clamp(
                action_low, action_high
            )
            discount = args.gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
                qf1_next_target_projected, qf2_next_target_projected = (
                    qnet_target.projection(
                        next_critic_observations,
                        next_state_actions,
                        rewards,
                        bootstrap,
                        discount,
                    )
                )
                qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
                qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)
                if args.use_cdq:
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1)
                        < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
                else:
                    qf1_next_target_dist, qf2_next_target_dist = (
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )

            qf1, qf2 = qnet(critic_observations, actions)
            qf1_loss = -torch.sum(
                qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
            ).mean()
            qf2_loss = -torch.sum(
                qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
            ).mean()
            qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if args.use_grad_norm_clipping:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(q_optimizer)
        scaler.update()

        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = qf1_next_target_value.max().detach()
        logs_dict["qf_min"] = qf1_next_target_value.min().detach()
        return logs_dict

    def update_pol(data, logs_dict):
        with autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            critic_observations = (
                data["critic_observations"]
                if envs.asymmetric_obs
                else data["observations"]
            )

            qf1, qf2 = qnet(critic_observations, actor(data["observations"]))
            qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
            qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
            if args.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        if args.use_grad_norm_clipping:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(actor_optimizer)
        scaler.update()
        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["actor_loss"] = actor_loss.detach()
        return logs_dict

    @torch.no_grad()
    def soft_update(src, tgt, tau: float):
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]

        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    if args.compile:
        compile_mode = args.compile_mode
        update_main = torch.compile(update_main, mode=compile_mode)
        update_pol = torch.compile(update_pol, mode=compile_mode)
        policy = torch.compile(policy, mode=None)
        normalize_obs = torch.compile(obs_normalizer.forward, mode=None)
        normalize_critic_obs = torch.compile(critic_obs_normalizer.forward, mode=None)
        if args.reward_normalization:
            update_stats = torch.compile(reward_normalizer.update_stats, mode=None)
        normalize_reward = torch.compile(reward_normalizer.forward, mode=None)
    else:
        normalize_obs = obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward
        if args.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    if envs.asymmetric_obs:
        obs, critic_obs = envs.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs = envs.reset()
    if args.checkpoint_path:
        # Load checkpoint if specified
        torch_checkpoint = torch.load(
            f"{args.checkpoint_path}", map_location=device, weights_only=False
        )
        actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        critic_obs_normalizer.load_state_dict(
            torch_checkpoint["critic_obs_normalizer_state"]
        )
        qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        global_step = torch_checkpoint["global_step"]
    else:
        global_step = 0

    dones = None

    # Changed: Added explicit columns to `Progress` to mimic tqdm (including MofNCompleteColumn for X/Y steps)
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )
    progress.start()
    pbar = progress.add_task("", total=args.total_timesteps, completed=global_step)

    start_time = None
    desc = ""
    current_epoch = -1
    latest_episode_info = {}

    while global_step < args.total_timesteps:
        if hasattr(envs, "env") and hasattr(envs.env, "set_curriculum"):
            max_epochs = getattr(args, "max_epochs", 5000)
            epoch = (global_step * max_epochs) // args.total_timesteps
            if epoch > current_epoch:
                envs.env.set_curriculum(epoch)
                current_epoch = epoch

        mark_step()
        logs_dict = TensorDict()
        if (
                start_time is None
                and global_step >= args.measure_burnin + args.learning_starts
        ):
            start_time = time.time()
            measure_burnin = global_step

        with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            norm_obs = normalize_obs(obs)
            actions = policy(obs=norm_obs, dones=dones)

        next_obs, rewards, dones, infos = envs.step(actions.float())
        truncations = infos["time_outs"]
        
        if "episode_info" in infos:
            latest_episode_info = infos["episode_info"]

        if args.reward_normalization:
            if env_type == "mtbench":
                task_ids_one_hot = obs[..., -envs.num_tasks:]
                task_indices = torch.argmax(task_ids_one_hot, dim=1)
                update_stats(rewards, dones.float(), task_ids=task_indices)
            else:
                update_stats(rewards, dones.float())

        if envs.asymmetric_obs:
            next_critic_obs = infos["observations"]["critic"]
        # Compute 'true' next_obs and next_critic_obs for saving
        true_next_obs = torch.where(
            dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
        )
        if envs.asymmetric_obs:
            true_next_critic_obs = torch.where(
                dones[:, None] > 0,
                infos["observations"]["raw"]["critic_obs"],
                next_critic_obs,
            )

        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "rewards": torch.as_tensor(
                        rewards, device=device, dtype=torch.float
                    ),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )
        if envs.asymmetric_obs:
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = true_next_critic_obs
        rb.extend(transition)

        obs = next_obs
        if envs.asymmetric_obs:
            critic_obs = next_critic_obs

        if global_step > args.learning_starts:
            for i in range(args.num_updates):
                data = rb.sample(max(1, args.batch_size // args.num_envs))
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(
                    data["next"]["observations"]
                )
                if envs.asymmetric_obs:
                    data["critic_observations"] = normalize_critic_obs(
                        data["critic_observations"]
                    )
                    data["next"]["critic_observations"] = normalize_critic_obs(
                        data["next"]["critic_observations"]
                    )
                raw_rewards = data["next"]["rewards"]
                if env_type in ["mtbench"] and args.reward_normalization:
                    # Multi-task reward normalization
                    task_ids_one_hot = data["observations"][..., -envs.num_tasks:]
                    task_indices = torch.argmax(task_ids_one_hot, dim=1)
                    data["next"]["rewards"] = normalize_reward(
                        raw_rewards, task_ids=task_indices
                    )
                else:
                    data["next"]["rewards"] = normalize_reward(raw_rewards)

                logs_dict = update_main(data, logs_dict)
                if args.num_updates > 1:
                    if i % args.policy_frequency == 1:
                        logs_dict = update_pol(data, logs_dict)
                else:
                    if global_step % args.policy_frequency == 0:
                        logs_dict = update_pol(data, logs_dict)

                soft_update(qnet, qnet_target, args.tau)

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                progress.update(pbar, description=f"[cyan]{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "actor_loss": logs_dict["actor_loss"].mean(),
                        "qf_loss": logs_dict["qf_loss"].mean(),
                        "qf_max": logs_dict["qf_max"].mean(),
                        "qf_min": logs_dict["qf_min"].mean(),
                        "actor_grad_norm": logs_dict["actor_grad_norm"].mean(),
                        "critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                        "env_rewards": rewards.mean(),
                        "buffer_rewards": raw_rewards.mean(),
                    }

                    for k, v in latest_episode_info.items():
                        if isinstance(v, (int, float)):
                            logs[f"env/{k}"] = v
                        elif hasattr(v, "item"):
                            if hasattr(v, "numel") and v.numel() > 1:
                                logs[f"env/{k}"] = v.float().mean().item()
                            else:
                                logs[f"env/{k}"] = v.item()

                    if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                        print(f"Evaluating at global step {global_step}")
                        eval_metrics = evaluate()
                        if env_type in ["humanoid_bench", "isaaclab", "mtbench"]:
                            # NOTE: Hacky way of evaluating performance, but just works
                            obs = envs.reset()
                        logs.update(eval_metrics)

                    if (
                            args.render_interval > 0
                            and global_step % args.render_interval == 0
                    ):
                        renders = render_with_rollout()
                        render_video = wandb.Video(
                            np.array(renders).transpose(
                                0, 3, 1, 2
                            ),  # Convert to (T, C, H, W) format
                            fps=30,
                            format="gif",
                        )
                        logs["eval/video"] = render_video
                if args.use_wandb:
                    env_step = global_step * args.num_envs
                    wandb.log(
                        {
                            "global/env_step": env_step,
                            "speed": speed,
                            "frame": env_step,
                            "critic_lr": q_scheduler.get_last_lr()[0],
                            "actor_lr": actor_scheduler.get_last_lr()[0],
                            **logs,
                        },
                        step=env_step,
                    )

            if (
                    args.save_interval > 0
                    and global_step > 0
                    and global_step % args.save_interval == 0
            ):
                ckpt_path = f"{checkpoint_dir}/step{global_step}.pt"
                print(f"[INFO] Saving checkpoint at global step {global_step} to: {ckpt_path}")
                save_params(
                    global_step,
                    actor,
                    qnet,
                    qnet_target,
                    obs_normalizer,
                    critic_obs_normalizer,
                    args,
                    ckpt_path,
                )

        global_step += 1
        actor_scheduler.step()
        q_scheduler.step()
        progress.update(pbar, advance=1)

    progress.stop()

    final_ckpt_path = f"{checkpoint_dir}/final.pt"
    print(f"[INFO] Saving final checkpoint at global step {global_step} to: {final_ckpt_path}")
    save_params(
        global_step,
        actor,
        qnet,
        qnet_target,
        obs_normalizer,
        critic_obs_normalizer,
        args,
        final_ckpt_path,
    )


if __name__ == "__main__":
    main()