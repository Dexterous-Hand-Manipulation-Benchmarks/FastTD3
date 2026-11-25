"""Scores one FastTD3 checkpoint on the shared DexMachina benchmark protocol.

Run from algorithms/FastTD3: `python fast_td3/eval_dexmachina.py --checkpoint <run dir>/final.pt [--wandb]`.
Everything comes from what the run saved next to its checkpoints: the env from env_spec.json, the actor and the
observation normaliser from the checkpoint and its saved args, and with --wandb the W&B run from wandb_run.json.
The actor acts with its deterministic mean (no exploration noise) and a frozen normaliser. The zero-action floor runs
on the same env before the policy's episode. The trajectory is saved next to the checkpoint; an existing one is never
replaced.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import genesis as gs
import torch
import torch.nn as nn
import wandb
from agent_builders import build_actor, build_obs_normalizer
from dexmachina.envs.evaluation import (
    BENCHMARK_PREFIX,
    EVAL_SEED,
    BenchmarkEvaluation,
    checked_dexmachina_state,
    rollout_episode,
    save_trajectory,
    score,
    zero_action_floor,
)
from dexmachina.envs.factory import make_eval_env, task_hash
from dexmachina.envs.run_spec import RUN_SPEC_FILENAME, load_run_spec
from environments.dexmachina_env import benchmark_env_spec
from fast_td3_utils import WANDB_RUN_FILENAME
from genesis.utils import set_random_seed

SHORT_SHA_LENGTH = 12
# What a run chooses of its task; the checkpoint's saved args and the run's env_spec.json must agree on all of them.
RUN_CHOICES = ("hand", "clip", "seed", "num_envs")
TRAJECTORY_FILENAME = "benchmark_trajectory_{checkpoint}_{sha}.npz"


class SavedArgs(SimpleNamespace):
    """The args a checkpoint was trained with; a missing one is an error, never a default."""

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"The checkpoint's saved args have no {name!r}, so the network it was trained with cannot be rebuilt. "
            "It was saved by an older FastTD3; retrain with the current one."
        )


def trajectory_path(checkpoint: Path, dexmachina_sha: str) -> Path:
    return checkpoint.parent / TRAJECTORY_FILENAME.format(
        checkpoint=checkpoint.stem, sha=dexmachina_sha[:SHORT_SHA_LENGTH]
    )


def load_obs_normalizer(saved: SavedArgs, state: dict | None, n_obs: int, device: str) -> nn.Module:
    """The run's observation normaliser, frozen; its saved state must match the run's obs_normalization."""
    if saved.obs_normalization and not state:
        raise ValueError(
            "The run normalised observations (obs_normalization=True) but the checkpoint has no normaliser state, "
            "so the actor would see unnormalised observations. The checkpoint is incomplete."
        )
    if not saved.obs_normalization and state:
        raise ValueError(
            "The checkpoint has a normaliser state although the run did not normalise observations "
            "(obs_normalization=False); its args and weights do not belong together."
        )
    normalizer = build_obs_normalizer(saved.obs_normalization, n_obs, device)
    normalizer.load_state_dict(state or {})
    return normalizer.eval()


def check_checkpoint_belongs_to_run(saved: SavedArgs, spec: Any) -> None:
    """The checkpoint was trained on this run's hand, clip, seed and num_envs: its score and W&B run are this run's."""
    trained = benchmark_env_spec(saved.env_name, saved.seed, saved.num_envs)
    differing = [name for name in RUN_CHOICES if getattr(trained, name) != getattr(spec, name)]
    if differing:
        raise ValueError(
            "The checkpoint was trained with "
            + ", ".join(f"{name}={getattr(trained, name)!r}" for name in differing)
            + " but env_spec.json next to it has "
            + ", ".join(f"{name}={getattr(spec, name)!r}" for name in differing)
            + ": the checkpoint does not belong to this run directory."
        )


def load_networks(
    saved: SavedArgs, saved_checkpoint: dict, n_obs: int, n_act: int, device: str
) -> tuple[nn.Module, nn.Module]:
    """The checkpoint's actor and observation normaliser, built from its saved args; every weight must load."""
    actor = build_actor(saved, n_obs, n_act, saved.num_envs, device)
    actor.load_state_dict(saved_checkpoint["actor_state_dict"])
    obs_normalizer = load_obs_normalizer(saved, saved_checkpoint["obs_normalizer_state"], n_obs, device)
    return actor.eval(), obs_normalizer


def deterministic_policy(
    actor: nn.Module, obs_normalizer: nn.Module, saved: SavedArgs, device: str
) -> Callable[[torch.Tensor], torch.Tensor]:
    """The actor's mean action on normalised observations, with training's precision; nothing is updated."""
    device_type = torch.device(device).type
    autocast_enabled = saved.amp and saved.cuda and device_type == "cuda" and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if saved.amp_dtype == "bf16" else torch.float16

    def act(observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=autocast_enabled):
            if saved.obs_normalization:
                observations = obs_normalizer(observations, update=False)
            return actor(observations).float()

    return act


def evaluate(
    checkpoint: Path, device: str, *, eval_seed: int = EVAL_SEED, allow_dirty_dexmachina: bool = False
) -> BenchmarkEvaluation:
    """Scores `checkpoint` on the shared protocol, from what its run saved. Genesis must be initialised for `device`."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{checkpoint} does not exist.")
    run_dir = checkpoint.parent
    if not (run_dir / RUN_SPEC_FILENAME).is_file():
        raise FileNotFoundError(
            f"{run_dir / RUN_SPEC_FILENAME} does not exist. FastTD3 writes it next to its checkpoints when a run "
            "starts, so this run predates it or the path is wrong; such runs cannot be scored on the benchmark."
        )
    run_spec = load_run_spec(run_dir)
    spec = run_spec.spec
    evaluated_state = checked_dexmachina_state(run_spec.dexmachina_state, allow_dirty_dexmachina)
    trajectory_file = trajectory_path(checkpoint, evaluated_state.sha)
    if trajectory_file.exists():
        raise FileExistsError(
            f"{trajectory_file} exists already: this checkpoint was evaluated at this dexmachina commit. Delete the "
            "file to evaluate it again."
        )

    saved_checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
    saved = SavedArgs(**saved_checkpoint["args"])
    check_checkpoint_belongs_to_run(saved, spec)
    checkpoint_step = int(saved_checkpoint["global_step"])

    torch.set_float32_matmul_precision("high")
    env: Any = make_eval_env(spec, device)
    actor, obs_normalizer = load_networks(saved, saved_checkpoint, int(env.obs_dim), int(env.action_dim), device)
    set_random_seed(eval_seed)

    floor = zero_action_floor(env, spec.hand, spec.clip)
    task = task_hash(spec)
    trajectory = rollout_episode(
        env, deterministic_policy(actor, obs_normalizer, saved, device), spec.hand, spec.clip, task,
        evaluated_state.sha,
    )
    save_trajectory(trajectory_file, trajectory)
    return BenchmarkEvaluation(
        eval_score=score(trajectory), task_hash=task, saved_state=run_spec.dexmachina_state,
        evaluated_state=evaluated_state, floor=floor, checkpoint_step=checkpoint_step,
        env_step=checkpoint_step * spec.num_envs, eval_seed=eval_seed,
    )


def recorded_wandb_run(run_dir: Path) -> dict[str, str | None]:
    """The W&B run the checkpoints in `run_dir` were trained in (id, project, entity)."""
    path = run_dir / WANDB_RUN_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist: the run trained without W&B, offline, or before FastTD3 recorded its W&B run. "
            "Evaluate without --wandb."
        )
    return json.loads(path.read_text())


def resume_wandb_run(identity: dict[str, str | None]) -> Any:
    """The training run, resumed, with the benchmark keys plotted against benchmark/env_step."""
    run = wandb.init(id=identity["id"], project=identity["project"], entity=identity["entity"], resume="must")
    wandb.define_metric(f"{BENCHMARK_PREFIX}/*", step_metric=f"{BENCHMARK_PREFIX}/env_step")
    return run


def log_evaluation(run: Any, evaluation: BenchmarkEvaluation) -> None:
    wandb.log(evaluation.metrics())
    run.summary.update(evaluation.provenance())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="a step*.pt or final.pt of a FastTD3 run")
    parser.add_argument("--device", type=str, default="cuda:0", help="device of the env and the actor")
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED,
                        help="seed set before the episodes; the default is shared by all methods")
    parser.add_argument("--allow-dirty-dexmachina", action="store_true",
                        help="evaluate although dexmachina had uncommitted changes at training or now; for "
                             "debugging, not for reported numbers")
    parser.add_argument("--wandb", action="store_true",
                        help="log into the W&B run the checkpoint was trained in (from wandb_run.json)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = args.checkpoint.resolve()
    identity = recorded_wandb_run(checkpoint.parent) if args.wandb else None
    gs.init(backend=gs.cpu if args.device.startswith("cpu") else gs.gpu, logging_level="warning")
    # Resumed before the evaluation, so a W&B failure stops it before it writes its trajectory.
    run = resume_wandb_run(identity) if identity is not None else None
    try:
        evaluation = evaluate(
            checkpoint, args.device, eval_seed=args.eval_seed, allow_dirty_dexmachina=args.allow_dirty_dexmachina
        )
        for name, value in {**evaluation.metrics(), **evaluation.provenance()}.items():
            print(f"{name}: {value}")
        if run is not None:
            log_evaluation(run, evaluation)
    finally:
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
