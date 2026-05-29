"""
Custom VERL PPO entry point for Time-Series GRPO training.

Registers TS extensions in the MAIN process AND inside the TaskRunner Ray actor
by monkey-patching VERL's TaskRunner with a subclass that re-imports the
registrations inside the actor process.

  rollout.py    → registers "ts_vllm"  in RolloutReplicaRegistry + _ROLLOUT_REGISTRY
  agent_loop.py → registers "ts_single_turn_agent" in the agent-loop registry

Usage:
    python3 -m teleprism.rl.training.main \\
        actor_rollout_ref.rollout.name=ts_vllm \\
        actor_rollout_ref.rollout.agent.default_agent_loop=ts_single_turn_agent \\
        [... all other VERL hydra args ...]
"""

# ── TS extension registrations (main process side-effect imports) ─────────────
import teleprism.rl.training.rollout  # noqa: F401  registers "ts_vllm"
import teleprism.rl.training.agent_loop  # noqa: F401  registers "ts_single_turn_agent"
# ─────────────────────────────────────────────────────────────────────────────

# Subclass TaskRunner so that when it runs as a Ray remote actor (a separate
# Python process), it re-imports the registrations before init_workers() tries
# to look up "ts_vllm" in RolloutReplicaRegistry.
import verl.trainer.main_ppo as _verl_main_ppo
from verl.trainer.main_ppo import TaskRunner


class TSTaskRunner(TaskRunner):
    """TaskRunner that ensures TS registries are populated inside the Ray actor."""

    def run(self, config):
        # Re-import in this Ray actor process to register ts_vllm and ts_single_turn_agent
        import teleprism.rl.training.rollout  # noqa: F401
        import teleprism.rl.training.agent_loop  # noqa: F401

        # Patch compute_grpo_outcome_advantage to log group variance stats to wandb
        self._patch_grpo_group_stats()

        super().run(config)

    @staticmethod
    def _patch_grpo_group_stats():
        """Wrap compute_grpo_outcome_advantage to compute and log GRPO group stats."""
        import numpy as np
        from collections import defaultdict
        from verl.trainer.ppo import core_algos, ray_trainer

        _orig = core_algos.compute_grpo_outcome_advantage

        def _patched(token_level_rewards, response_mask, index, **kwargs):
            advantages, returns = _orig(token_level_rewards, response_mask, index, **kwargs)

            try:
                scores = token_level_rewards.sum(dim=-1)
                id2scores = defaultdict(list)
                for i in range(scores.shape[0]):
                    id2scores[index[i]].append(scores[i].item())

                group_stds = []
                zero_var = 0
                for sc in id2scores.values():
                    std = np.std(sc)
                    group_stds.append(std)
                    if std < 0.01:
                        zero_var += 1

                n_groups = len(id2scores)
                all_scores = scores.tolist()
                n = len(all_scores)

                # Store on the function object so ray_trainer can pick it up
                _patched._grpo_stats = {
                    "grpo/group_mean_std": float(np.mean(group_stds)),
                    "grpo/group_median_std": float(np.median(group_stds)),
                    "grpo/zero_var_group_pct": float(zero_var / n_groups) if n_groups else 0,
                    "grpo/score_pct_correct": sum(1 for s in all_scores if s >= 0.99) / n,
                    "grpo/score_pct_wrong": sum(1 for s in all_scores if 0.09 < s < 0.15) / n,
                }
            except Exception as e:
                print(f"[main] GRPO stats error: {e}")

            return advantages, returns

        _patched._grpo_stats = {}
        core_algos.compute_grpo_outcome_advantage = _patched

        # Patch RayPPOTrainer._validate and fit to log grpo stats after compute_advantage
        _orig_compute_adv = ray_trainer.compute_advantage

        def _patched_compute_adv(data, *args, **kwargs):
            result = _orig_compute_adv(data, *args, **kwargs)
            # Log GRPO group stats directly to wandb
            if _patched._grpo_stats:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(_patched._grpo_stats, commit=False)
                except Exception:
                    pass
            return result

        ray_trainer.compute_advantage = _patched_compute_adv
        print("[main] Patched GRPO advantage for group variance wandb logging ✓")


# Monkey-patch VERL's module-level name so that run_ppo() uses TSTaskRunner
# when it does: task_runner_class = ray.remote(num_cpus=1)(TaskRunner)
_verl_main_ppo.TaskRunner = TSTaskRunner

from verl.trainer.main_ppo import main  # inherits @hydra.main decorator

if __name__ == "__main__":
    main()
