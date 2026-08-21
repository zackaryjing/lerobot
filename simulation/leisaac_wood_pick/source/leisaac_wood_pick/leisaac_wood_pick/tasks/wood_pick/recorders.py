"""Recorder terms that keep scripted IK control compatible with real SO-101 data."""

from __future__ import annotations

from isaaclab.managers.manager_term_cfg import RecorderTermCfg
from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.utils import configclass


class PostStepJointPositionTargetsRecorder(RecorderTerm):
    """Record the six joint targets produced by IK as the training action.

    The state machine consumes an internal 8D Cartesian command.  Recording
    that command would create a policy that cannot drive the real LeRobot
    six-joint interface.  At post-step time the IK action has already written
    its actual target, so this term records the deployable action instead.
    """

    def record_post_step(self):
        return "actions", self._env.scene["robot"].data.joint_pos_target.clone()


@configclass
class PostStepJointPositionTargetsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PostStepJointPositionTargetsRecorder
