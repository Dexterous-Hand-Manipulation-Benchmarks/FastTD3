import os
import sys
import argparse
from collections import defaultdict
import json
import yaml
import numpy as np
import torch
import wandb
import genesis as gs

# Ensure FastTD3, logging, and dexmachina packages can be imported
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
fast_td3_dir = os.path.join(base_dir, "FastTD3")
dexmachina_dir = os.path.join(base_dir, "dexmachina")
logging_dir = os.path.join(base_dir, "logging")
for p in [base_dir, fast_td3_dir, dexmachina_dir, logging_dir, os.path.dirname(__file__)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import math
from wandb_logger import UniversalLogger
from fast_td3.hyperparams import DexMachinaArgs
from fast_td3.fast_td3 import Actor as FastTD3Actor
from fast_td3.fast_td3_simbav2 import Actor as SimbaActor
from fast_td3.fast_td3_utils import EmpiricalNormalization
from dexmachina_wrapper import GenesisFastTD3Wrapper
from dexmachina.envs.constructors import get_all_env_cfg
from dexmachina.envs.base_env import BaseEnv


def gather_object_state_tensor(demo_data):
    obj_pos = demo_data["obj_pos"]
    obj_quat = demo_data["obj_quat"]
    obj_arti = demo_data["obj_arti"]
    if len(obj_arti.shape) == 1:
        obj_arti = obj_arti[:, None]
    arr = np.concatenate([obj_pos, obj_quat, obj_arti], axis=1)
    return torch.tensor(arr, dtype=torch.float32)


def sanitize_frame(frame):
    """
    Ensures a frame is a 3D numpy array (H, W, 3) of dtype uint8 in range [0, 255].
    Handles PyTorch tensors, leading batch dims, alpha/seg channels, and float ranges.
    """
    if frame is None:
        return None
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if not isinstance(frame, np.ndarray):
        return None

    # Squeeze leading singleton/batch dimensions: e.g. (1, H, W, C) -> (H, W, C)
    while frame.ndim > 3 and frame.shape[0] == 1:
        frame = frame[0]

    # Handle grayscale / 2D
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)

    # Drop alpha channel if 4-channel RGBA or RGB+seg
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]

    # Convert floats in [0, 1] to uint8 [0, 255]
    if frame.dtype != np.uint8:
        if frame.max() <= 1.01 and frame.max() > 0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Validate final dimensions
    if frame.ndim == 3 and frame.shape[-1] == 3 and frame.shape[0] > 10 and frame.shape[1] > 10:
        return frame
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate FastTD3 on Dexmachina environment and generate video.")
    parser.add_argument('--checkpoint', '-ck', type=str, required=True, help='Path to FastTD3 .pt checkpoint')
    parser.add_argument('--num_envs', '-ne', type=int, default=20, help='Number of parallel evaluation environments')
    parser.add_argument('--env_name', type=str, default=None, help='Dexmachina env name e.g. dexmachina|allegro_hand|box-30-230')
    parser.add_argument('--eval_episodes', type=int, default=1, help='Number of evaluation episodes')
    parser.add_argument('--record_video', action='store_true', help='Record a video of the evaluation')
    parser.add_argument('--video_fname', type=str, default="video.gif", help='Filename of the output video')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for simulation and policy')
    parser.add_argument('--wandb_run_id', type=str, default=None, help='WandB run ID to attach evaluation videos and metrics to')
    parser.add_argument('--step', type=int, default=None, help='The exact checkpoint step')
    parser.add_argument('--project', type=str, default='dexmachina', help='WandB project name')
    parser.add_argument('--entity', type=str, default='schrapsi-kit', help='WandB entity/team name')
    args = parser.parse_args()

    # Load checkpoint
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    saved_args = checkpoint.get("args", {})
    if isinstance(saved_args, dict):
        pass
    elif hasattr(saved_args, "__dict__"):
        saved_args = vars(saved_args)
    else:
        saved_args = {}

    env_name = args.env_name or saved_args.get("env_name", "dexmachina|allegro_hand|box-30-230")
    print(f"[INFO] Environment: {env_name} | Eval Envs: {args.num_envs}")

    # Initialize environment config
    dex_args = DexMachinaArgs(env_name=env_name, num_envs=args.num_envs)
    parts = dex_args.env_name.split("|")
    dex_args.hand = parts[1] if len(parts) > 1 else "allegro_hand"
    dex_args.clip = parts[2] if len(parts) > 2 else "box-30-230"
    if hasattr(dex_args, "n_envs_per_row") and dex_args.n_envs_per_row == 0:
        dex_args.n_envs_per_row = None
    if hasattr(dex_args, "color_object") and dex_args.color_object == "":
        dex_args.color_object = None
    
    gs.init(backend=gs.gpu, logging_level='warning')
    env_kwargs = get_all_env_cfg(dex_args, device=str(device))
    
    # Force eval settings
    env_kwargs['env_cfg']['early_reset_threshold'] = 0.0
    env_kwargs['env_cfg']['is_eval'] = True
    env_kwargs['rand_cfg']['randomize'] = False
    env_kwargs['env_cfg']['num_envs'] = args.num_envs
    env_kwargs['env_cfg']['rand_init_ratio'] = 0.0
    
    if args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True  
        env_kwargs['env_cfg']['scene_kwargs']['show_viewer'] = False
        env_kwargs['env_cfg']['record_video'] = True 
        env_kwargs['env_cfg']['render_segmentation'] = False 
        if 'camera_kwargs' not in env_kwargs['env_cfg']:
            env_kwargs['env_cfg']['camera_kwargs'] = {}
        env_kwargs['env_cfg']['camera_kwargs']['front'] = dict(
            res=(512, 512),
            fov=40,
            pos=(0.5, -1.5, 1.2),
            lookat=(0.0, -1.58, 2.0),
        )
    
    for name, cfg in env_kwargs['object_cfgs'].items():
        cfg['actuated'] = False
        
    env_kwargs.pop("curriculum_cfg", None)
    
    # Create env
    base_env = BaseEnv(**env_kwargs)
    env = GenesisFastTD3Wrapper(base_env, rl_device=str(device))
    uenv = env.env
    
    demo_data = env_kwargs['demo_data']
    obj_state_tensor = gather_object_state_tensor(demo_data).to(device)
    
    n_act = env.num_actions
    n_obs = env.num_obs if isinstance(env.num_obs, int) else env.num_obs[0]
    
    actor_hidden_dim = saved_args.get("actor_hidden_dim", dex_args.actor_hidden_dim)
    init_scale = saved_args.get("init_scale", dex_args.init_scale)
    std_min = saved_args.get("std_min", dex_args.std_min)
    std_max = saved_args.get("std_max", dex_args.std_max)
    sim_type = saved_args.get("sim_type", "")
    sim_dimension = saved_args.get("sim_dimension", 64)
    seq_len = saved_args.get("seq_len", 8)

    # Initialize models based on saved agent type
    agent_type = saved_args.get("agent", "fasttd3")
    if agent_type == "fasttd3_simbav2":
        actor_num_blocks = saved_args.get("actor_num_blocks", 1)
        actor = SimbaActor(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=args.num_envs,
            hidden_dim=actor_hidden_dim,
            scaler_init=math.sqrt(2.0 / actor_hidden_dim),
            scaler_scale=math.sqrt(2.0 / actor_hidden_dim),
            alpha_init=1.0 / (actor_num_blocks + 1),
            alpha_scale=1.0 / math.sqrt(actor_hidden_dim),
            expansion=4,
            c_shift=3.0,
            num_blocks=actor_num_blocks,
            std_min=std_min,
            std_max=std_max,
            device=device,
        ).to(device)
    else:
        actor = FastTD3Actor(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=args.num_envs,
            init_scale=init_scale,
            hidden_dim=actor_hidden_dim,
            std_min=std_min,
            std_max=std_max,
            sim_type=sim_type,
            sim_dimension=sim_dimension,
            seq_len=seq_len,
            device=device,
        ).to(device)
    obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    
    # Manually remove noise_scales before load_state_dict to support changing num_envs during eval
    actor_state_dict = checkpoint["actor_state_dict"].copy()
    if "noise_scales" in actor_state_dict:
        del actor_state_dict["noise_scales"]
    actor.load_state_dict(actor_state_dict, strict=False)

    if checkpoint.get("obs_normalizer_state") is not None:
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
        
    actor.eval()
    obs_normalizer.eval()
    
    ep_len = uenv.max_episode_length
    obj = uenv.objects[uenv.object_names[0]]
    if obj.actuated:
        obj.set_joint_gains(0.0, 0.0, force_range=0.0)

    # Setup output directory
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    ckpt_name = os.path.basename(args.checkpoint).replace('.pt', '')
    out_dir = args.out_dir or os.path.join(ckpt_dir, f"{ckpt_name}_eval")
    os.makedirs(out_dir, exist_ok=True)

    # Save config.yaml so dexmachina evaluation scripts (compute_add.py, group_results.py) can parse metadata
    config_yaml_path = os.path.join(out_dir, "config.yaml")
    config_payload = {
        "clip": dex_args.clip,
        "hand": dex_args.hand,
        "env_name": env_name,
        "num_envs": args.num_envs,
    }
    with open(config_yaml_path, "w") as f:
        yaml.dump(config_payload, f, default_flow_style=False)
    print(f"[INFO] Saved config metadata to: {config_yaml_path}")

    last_video_path = None
    ep_return = 0.0

    # Evaluation Loop
    for ep in range(args.eval_episodes):
        print(f"[INFO] Running evaluation rollout (Episode {ep+1}/{args.eval_episodes}, length: {ep_len} steps)...")
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]

        if args.record_video:
            uenv.start_recording()
            uenv.max_video_frames = int(ep_len * 2)

        manual_renders = []
        if args.record_video and uenv._floating_camera is not None:
            try:
                render_out = uenv._floating_camera.render()
                frame = render_out[0] if isinstance(render_out, tuple) else render_out
                sanitized = sanitize_frame(frame)
                if sanitized is not None:
                    manual_renders.append(sanitized)
            except Exception as e:
                pass

        eval_data = defaultdict(list)
        accumulated_rewards = torch.zeros(args.num_envs, device=device)
        
        for i in range(ep_len):
            with torch.inference_mode():
                env_step = uenv.episode_length_buf.cpu().numpy()[0]
                
                # Normalize observation and act deterministically
                norm_obs = obs_normalizer(obs, update=False)
                actions = actor(norm_obs)
                
                demo_state = obj_state_tensor[env_step]
                
                obs, rew, dones, infos = env.step(actions)
                accumulated_rewards += rew
                if isinstance(obs, dict):
                    obs = obs["obs"]

                obj_pos, obj_quat, obj_arti = obj.root_pos, obj.root_quat, obj.dof_pos
                obj_state = torch.cat([obj_pos, obj_quat, obj_arti], dim=-1)
                
                eval_data["obj_state"].append(obj_state.cpu().numpy())
                eval_data["demo_state"].append(demo_state.cpu().numpy())
                
                rew_dict = uenv.rew_dict
                for key in ['pos_dist', 'rot_dist', 'arti_dist']:
                    if key in rew_dict:
                        eval_data[key].append(rew_dict[key].cpu().numpy())
                
                if args.record_video and i % 2 == 0 and uenv._floating_camera is not None:
                    try:
                        render_out = uenv._floating_camera.render()
                        frame = render_out[0] if isinstance(render_out, tuple) else render_out
                        sanitized = sanitize_frame(frame)
                        if sanitized is not None:
                            manual_renders.append(sanitized)
                    except Exception as e:
                        pass
                        
        ep_return = accumulated_rewards.mean().item()
        eval_data = {k: np.stack(v) for k, v in eval_data.items()}
        out_file = os.path.join(out_dir, f"eval_ep{ep}.npy")
        np.save(out_file, eval_data)
        print(f"[SUCCESS] Saved FastTD3 eval data to {out_file}")
        
        if args.record_video:
            # First try retrieving recorded frames from BaseEnv
            recorded_frames = uenv.get_recorded_frames(wait_for_max=False)
            source_frames = recorded_frames if (recorded_frames is not None and len(recorded_frames) > 0) else manual_renders

            if source_frames and len(source_frames) > 0:
                video_name = args.video_fname if args.eval_episodes == 1 else f"video_ep{ep}.gif"
                if video_name.endswith(".mp4"):
                    video_name = video_name[:-4] + ".gif"
                if not video_name.endswith(".gif"):
                    video_name += ".gif"
                video_path = os.path.join(out_dir, video_name)
                
                clean_renders = []
                for r in source_frames:
                    clean = sanitize_frame(r)
                    if clean is not None:
                        clean_renders.append(clean)
                
                if len(clean_renders) > 0:
                    fps = int(1.0 / uenv.dt / 2) if hasattr(uenv, "dt") and uenv.dt > 0 else 30
                    try:
                        import imageio
                        imageio.mimsave(video_path, clean_renders, fps=fps, loop=0)
                    except Exception:
                        from moviepy.editor import ImageSequenceClip
                        clip = ImageSequenceClip(clean_renders, fps=fps)
                        clip.write_gif(video_path, fps=fps, logger=None)
                    last_video_path = video_path
                    print(f"[SUCCESS] Saved GIF ({len(clean_renders)} frames, {clean_renders[0].shape}) to {video_path}")
                else:
                    print("[WARNING] Could not format any valid RGB frames for GIF encoding.")
            else:
                print("[WARNING] No video frames were captured during rollout.")

    # Compute ADD and AUC-ADD metrics
    stats = None
    try:
        from dexmachina.eval.compute_add import compute_for_eval
        obj_name = dex_args.clip.split("-")[0]
        eval_file_path = os.path.join(out_dir, "eval_ep0.npy")
        stats_path = os.path.join(out_dir, "add_stats.json")
        compute_for_eval(
            eval_path=eval_file_path,
            obj_name=obj_name,
            out_name="add.npy",
            stats_name="add_stats.json",
            overwrite=True,
            output_cache=False,
            cache_dir="dexmachina/eval/sub_verts",
        )
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            print(f"[RESULTS] ADD-AUC: {stats.get('auc', {}).get('overall')}, ADD: {stats.get('mean_add', {}).get('overall')}")
    except Exception as e:
        print(f"[WARNING] Could not compute ADD metrics directly: {e}")

    # Sync metrics and video to WandB if wandb_run_id is provided
    if args.wandb_run_id:
        print(f"[INFO] Connecting to WandB run {args.wandb_run_id} ({args.entity}/{args.project})...")
        tokens = env_name.split("|")
        hand_name = dex_args.hand if hasattr(dex_args, "hand") else (tokens[1] if len(tokens) > 1 else "unknown_hand")
        clip_name = dex_args.clip if hasattr(dex_args, "clip") else (tokens[2] if len(tokens) > 2 else "unknown_clip")
        training_num_envs = int(saved_args.get("num_envs", 768))
        UniversalLogger.install(
            algo="fast_td3",
            hand=str(hand_name),
            clip=str(clip_name),
            seed=int(saved_args.get("seed", 0)),
            num_train_envs=training_num_envs,
            num_eval_envs=int(args.num_envs),
        )

        wandb.init(id=args.wandb_run_id, project=args.project, entity=args.entity, resume="allow")

        log_payload = {
            "eval/return": ep_return,
            "eval/ep_length": float(ep_len),
        }
        if stats is not None:
            if "auc" in stats and "overall" in stats["auc"]:
                log_payload["eval/auc_add"] = stats["auc"]["overall"]
            if "mean_add" in stats and "overall" in stats["mean_add"]:
                log_payload["eval/add"] = stats["mean_add"]["overall"]

        add_npy_path = os.path.join(out_dir, "add.npy")
        if os.path.exists(add_npy_path):
            try:
                add_data = np.load(add_npy_path, allow_pickle=True).item()
                succ_2cm_list = [np.mean(add_data[part] < 0.02) for part in add_data.keys()]
                succ_5cm_list = [np.mean(add_data[part] < 0.05) for part in add_data.keys()]
                log_payload["eval/success_2cm"] = float(np.mean(succ_2cm_list))
                log_payload["eval/success_5cm"] = float(np.mean(succ_5cm_list))
                print(f"[RESULTS] Success (2cm): {log_payload['eval/success_2cm'] * 100:.2f}% | Success (5cm): {log_payload['eval/success_5cm'] * 100:.2f}%")
            except Exception as e:
                print(f"[WARNING] Could not compute threshold success rates: {e}")

        eval_ep_path = os.path.join(out_dir, "eval_ep0.npy")
        if os.path.exists(eval_ep_path):
            try:
                ep_data = np.load(eval_ep_path, allow_pickle=True).item()
                if "pos_dist" in ep_data:
                    log_payload["eval/pos_dist"] = float(np.mean(ep_data["pos_dist"]))
                if "rot_dist" in ep_data:
                    log_payload["eval/rot_dist"] = float(np.mean(ep_data["rot_dist"]))
            except Exception as e:
                pass

        # Determine effective step from checkpoint metadata if not explicitly provided
        saved_step = checkpoint.get("global_step", 0)
        training_num_envs = int(saved_args.get("num_envs", 768))
        ckpt_env_step = saved_step * training_num_envs
        effective_step = args.step if args.step is not None else ckpt_env_step

        is_best = (ckpt_name == "best") or os.path.basename(args.checkpoint).startswith("best")
        fps = int(1.0 / uenv.dt / 2) if hasattr(uenv, "dt") and uenv.dt > 0 else 30

        if last_video_path and os.path.exists(last_video_path):
            cap_text = f"Checkpoint Step: {effective_step}" if effective_step is not None else "Eval Video"
            log_payload["media/eval_video"] = wandb.Video(last_video_path, fps=fps, format="gif", caption=cap_text)
            print(f"[INFO] Attached GIF {last_video_path} to WandB payload.")

        # Log as best/* metrics and summary if evaluating the best model
        if is_best:
            log_payload["best/return"] = ep_return
            log_payload["best/step"] = effective_step
            log_payload["best/global_step"] = saved_step
            if "eval/auc_add" in log_payload:
                log_payload["best/auc_add"] = log_payload["eval/auc_add"]
            if "eval/add" in log_payload:
                log_payload["best/add"] = log_payload["eval/add"]
            if "eval/success_2cm" in log_payload:
                log_payload["best/success_2cm"] = log_payload["eval/success_2cm"]
            if "eval/success_5cm" in log_payload:
                log_payload["best/success_5cm"] = log_payload["eval/success_5cm"]
            if last_video_path and os.path.exists(last_video_path):
                log_payload["media/best_video"] = wandb.Video(
                    last_video_path, fps=fps, format="gif", caption=f"Best Model (Env Step: {effective_step})"
                )

            # Update WandB run summary for clear table visualization
            for k in ["best/return", "best/step", "best/global_step", "best/auc_add", "best/add", "best/success_2cm", "best/success_5cm"]:
                if k in log_payload:
                    wandb.run.summary[k] = log_payload[k]
            print(f"[INFO] Updated WandB run summary with best/* metrics: step {effective_step}")

        if effective_step is not None:
            log_payload["global/step"] = effective_step
            log_payload["global/env_step"] = effective_step

        if log_payload:
            wandb.log(log_payload)
            print(f"[SUCCESS] Logged evaluation payload to WandB (Step {effective_step}): {list(log_payload.keys())}")

        wandb.finish()

    print(f"[SUCCESS] Evaluation complete! Results saved in {out_dir}")


if __name__ == '__main__':
    main()
