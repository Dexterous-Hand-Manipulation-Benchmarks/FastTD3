"""Networks built from the run's args, shared by train.py and eval_dexmachina.py so both build the same ones."""

import math
from types import ModuleType
from typing import Any, Optional

import torch.nn as nn

import fast_td3
import fast_td3_simbav2
from fast_td3_utils import EmpiricalNormalization

AGENT_MODULES = {"fasttd3": fast_td3, "fasttd3_simbav2": fast_td3_simbav2}


def _agent_module(args) -> ModuleType:
    if args.agent not in AGENT_MODULES:
        raise ValueError(f"Agent {args.agent} not supported")
    if args.agent == "fasttd3_simbav2" and args.sim_type:
        raise ValueError("SimNorm options are only supported with agent='fasttd3'")
    return AGENT_MODULES[args.agent]


def _simbav2_kwargs(hidden_dim: int, num_blocks: int) -> dict[str, Any]:
    return {
        "scaler_init": math.sqrt(2.0 / hidden_dim),
        "scaler_scale": math.sqrt(2.0 / hidden_dim),
        "alpha_init": 1.0 / (num_blocks + 1),
        "alpha_scale": 1.0 / math.sqrt(hidden_dim),
        "expansion": 4,
        "c_shift": 3.0,
        "num_blocks": num_blocks,
    }


def actor_class_and_kwargs(
    args, n_obs: int, n_act: int, num_envs: int, device, num_tasks: Optional[int] = None
) -> tuple[type[nn.Module], dict[str, Any]]:
    """The actor class and its kwargs; `num_tasks` selects the multi-task actor (MTBench)."""
    module = _agent_module(args)
    kwargs: dict[str, Any] = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_envs": num_envs,
        "device": device,
        "hidden_dim": args.actor_hidden_dim,
        "std_min": args.std_min,
        "std_max": args.std_max,
    }
    if num_tasks is not None:
        kwargs.update(
            n_obs=n_obs - num_tasks + args.task_embedding_dim,
            num_tasks=num_tasks,
            task_embedding_dim=args.task_embedding_dim,
        )
    if module is fast_td3:
        kwargs.update(
            init_scale=args.init_scale,
            sim_type=args.sim_type,
            sim_dimension=args.sim_dimension,
            seq_len=args.actor_seq_len,
        )
    else:
        kwargs.update(_simbav2_kwargs(args.actor_hidden_dim, args.actor_num_blocks))
    actor_cls = module.Actor if num_tasks is None else module.MultiTaskActor
    return actor_cls, kwargs


def critic_class_and_kwargs(
    args, n_critic_obs: int, n_act: int, device, num_tasks: Optional[int] = None
) -> tuple[type[nn.Module], dict[str, Any]]:
    """The critic class and its kwargs; `num_tasks` selects the multi-task critic (MTBench)."""
    module = _agent_module(args)
    kwargs: dict[str, Any] = {
        "n_obs": n_critic_obs,
        "n_act": n_act,
        "num_atoms": args.num_atoms,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "hidden_dim": args.critic_hidden_dim,
        "device": device,
    }
    if num_tasks is not None:
        kwargs.update(
            n_obs=n_critic_obs - num_tasks + args.task_embedding_dim,
            num_tasks=num_tasks,
            task_embedding_dim=args.task_embedding_dim,
        )
    if module is fast_td3:
        kwargs.update(sim_type=args.sim_type, sim_dimension=args.sim_dimension, seq_len=args.critic_seq_len)
    else:
        kwargs.update(_simbav2_kwargs(args.critic_hidden_dim, args.critic_num_blocks))
    critic_cls = module.Critic if num_tasks is None else module.MultiTaskCritic
    return critic_cls, kwargs


def build_actor(args, n_obs: int, n_act: int, num_envs: int, device) -> nn.Module:
    actor_cls, kwargs = actor_class_and_kwargs(args, n_obs, n_act, num_envs, device)
    return actor_cls(**kwargs)


def build_obs_normalizer(obs_normalization: bool, n_obs: int, device) -> nn.Module:
    return EmpiricalNormalization(shape=n_obs, device=device) if obs_normalization else nn.Identity()
