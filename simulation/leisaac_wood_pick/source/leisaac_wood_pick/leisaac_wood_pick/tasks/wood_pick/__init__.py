"""Gymnasium registration for the SO-101 wood-pick scene."""

import gymnasium as gym

TASK_ID = "Joyand-SO101-WoodPick-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "leisaac_wood_pick.tasks.wood_pick.wood_pick_env_cfg:WoodPickEnvCfg"
        ),
    },
)

__all__ = ["TASK_ID"]
