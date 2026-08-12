import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../FastTD3')))
import numpy as np
import torch
import genesis as gs
from collections import defaultdict

from fast_td3.hyperparams import DexMachinaArgs
from fast_td3.fast_td3 import Actor
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
    return torch.tensor(arr).float()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-ck', type=str, required=True, help='Path to FastTD3 .pt checkpoint')
    parser.add_argument('--num_envs', '-ne', type=int, default=20)
    parser.add_argument('--env_name', type=str, default="dexmachina|allegro_hand|box-30-230")
    parser.add_argument('--record_video', action='store_true', help='Record a video of the evaluation')
    args = parser.parse_args()

    # Load checkpoint
    device = torch.device('cuda:0')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Initialize environment config
    dex_args = DexMachinaArgs(env_name=args.env_name, num_envs=args.num_envs)
    parts = dex_args.env_name.split("|")
    dex_args.hand = parts[1] if len(parts) > 1 else "allegro_hand"
    dex_args.clip = parts[2] if len(parts) > 2 else "box-30-230"
    
    gs.init(backend=gs.gpu, logging_level='warning')
    env_kwargs = get_all_env_cfg(dex_args, device=str(device))
    
    # Force eval settings
    env_kwargs['env_cfg']['early_reset_threshold'] = 0.0
    env_kwargs['env_cfg']['is_eval'] = True
    env_kwargs['rand_cfg']['randomize'] = False
    
    if args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True  
        env_kwargs['env_cfg']['record_video'] = True 
        env_kwargs['env_cfg']['render_segmentation'] = True # Required for export_video to fill background with white
        print("Setting render resolution to 512") 
        if 'camera_kwargs' not in env_kwargs['env_cfg']:
            env_kwargs['env_cfg']['camera_kwargs'] = {}
        env_kwargs['env_cfg']['camera_kwargs']['front'] = dict(
            res=(512, 512),
            fov=40,
            pos=(0.5, -1.5, 1.2),
            lookat=(0.0, -0.08, 0.90),
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
    n_obs = env.num_obs if type(env.num_obs) == int else env.num_obs[0]
    
    # Initialize models
    actor = Actor(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=args.num_envs,
        device=device,
        init_scale=dex_args.init_scale,
        hidden_dim=dex_args.actor_hidden_dim,
        std_min=dex_args.std_min,
        std_max=dex_args.std_max
    ).to(device)
    obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    
    # strict=False does NOT ignore shape mismatches for existing keys! 
    # We must manually delete noise_scales from the state dict before loading.
    actor_state_dict = checkpoint["actor_state_dict"].copy()
    if "noise_scales" in actor_state_dict:
        del actor_state_dict["noise_scales"]
    actor.load_state_dict(actor_state_dict, strict=False)
    if "obs_normalizer_state" in checkpoint:
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])
        
    actor.eval()
    obs_normalizer.eval()
    
    # Evaluation Loop
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
        
    ep_len = uenv.max_episode_length
    obj = uenv.objects[uenv.object_names[0]]
    if obj.actuated:
        obj.set_joint_gains(0.0, 0.0, force_range=0.0)
        
    if args.record_video:
        uenv.start_recording()
        uenv.max_video_frames = int(uenv.max_episode_length)
        
    eval_data = defaultdict(list)
    
    for i in range(ep_len):
        with torch.inference_mode():
            env_step = uenv.episode_length_buf.cpu().numpy()[0]
            
            # Normalize and act deterministically
            norm_obs = obs_normalizer(obs, update=False)
            actions = actor(norm_obs) # FastTD3 Actor outputs deterministic actions when eval() is set, or we just use the mean. 
            # In FastTD3, the actor outputs the exact action if we don't add noise manually outside.
            
            demo_state = obj_state_tensor[env_step]
            
            obs, rew, dones, infos = env.step(actions)
            obj_pos, obj_quat, obj_arti = obj.root_pos, obj.root_quat, obj.dof_pos
            obj_state = torch.cat([obj_pos, obj_quat, obj_arti], dim=-1)
            
            eval_data["obj_state"].append(obj_state.cpu().numpy())
            eval_data["demo_state"].append(demo_state.cpu().numpy())
            
            rew_dict = uenv.rew_dict
            for key in ['pos_dist', 'rot_dist', 'arti_dist']:
                if key in rew_dict:
                    eval_data[key].append(rew_dict[key].cpu().numpy())
                    
    eval_data = {k: np.stack(v) for k, v in eval_data.items()}
    
    ckpt_dir = os.path.dirname(args.checkpoint)
    ckpt_name = os.path.basename(args.checkpoint).replace('.pt', '')
    out_dir = os.path.join(ckpt_dir, f"{ckpt_name}_eval")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "eval_ep0.npy")
    
    np.save(out_file, eval_data)
    print(f"Saved FastTD3 eval data to {out_file}")
    
    if args.record_video:
        frames = uenv.get_recorded_frames(wait_for_max=False)
        if frames is not None and len(frames) > 0:
            print(f"DEBUG: Recorded {len(frames)} frames.")
            print(f"DEBUG: First frame shape: {frames[0].shape}, max_val: {np.max(frames[0])}, min_val: {np.min(frames[0])}")
            
            # Save the very first frame as an image to debug what the camera sees!
            from PIL import Image
            img_frame = frames[0][:,:,:3]
            if img_frame.dtype != np.uint8:
                img_frame = (np.clip(img_frame, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img_frame).save(os.path.join(out_dir, "debug_frame0.png"))
            
            # Write video
            from moviepy.editor import ImageSequenceClip
            video_fname = os.path.join(out_dir, "video.mp4")
            clip = ImageSequenceClip([f[:,:,:3] for f in frames], fps=int(1/uenv.dt/2))
            clip.write_videofile(video_fname)
            print(f"Saved video to {video_fname}")
        else:
            print("DEBUG: No frames were recorded by the environment!")

if __name__ == '__main__':
    main()
