"""SocialNav agent for the ABotN-PointBench point-goal evaluator (ABot-Navigation).

Run from the SocialNav root, with both repositories on PYTHONPATH:

    export PYTHONPATH="$(pwd):$(pwd)/ABot-Navigation:${PYTHONPATH}"
    python -m abotn_evaluator.point_goal.runner \
        --agent-module utils.abotn_agent:SocialNavPointGoalAgent \
        --agent-config configs/abotn_socialnav.yaml \
        --data-dir ... --map-dir ... --render-url http://127.0.0.1:7036/render_gs

Frames. SocialNav works in the current robot frame with +X right and +Y forward
(paper, Fig. 7), with all coordinates divided by `step_scale`. The evaluator's
`get_pred_poses` reads waypoints in exactly that frame (in metres), and gives the
goal as `target_position = [front, left]`.
"""

import numpy as np
import torch
from PIL import Image

from abotn_evaluator.interface.point_goal import (
    BasePointGoalAgent,
    Observation,
    WaypointPrediction,
)
from utils.citywalker import SocialNavModel

# 4 history frames + the current one, matching <input_pos1>..<input_pos5>.
NUM_FRAMES = 5

# The trajectory-branch prompt of the released checkpoints is not published; this
# follows the scene-information block of the paper's prompt template (Fig. 7).
PROMPT = (
    "Scene information: [[<input_pos1>], [<input_pos2>], [<input_pos3>], [<input_pos4>], "
    "[<input_pos5>], [<input_target>]] are the robot's four history coordinates, its current "
    "coordinate and the goal coordinate, together with the corresponding observation images "
    "in the same order. The positive Y direction corresponds to forward, and the positive X "
    "direction corresponds to right. Predict the next waypoints toward the goal."
)


def front_left_to_model(xy):
    """[front, left] -> SocialNav [right, front]."""
    xy = np.asarray(xy, dtype=np.float32)
    return np.stack([-xy[..., 1], xy[..., 0]], axis=-1)


class SocialNavPointGoalAgent(BasePointGoalAgent):
    """Wraps a SocialNav checkpoint as a `BasePointGoalAgent`.

    Args:
        model_path: SocialNav checkpoint directory.
        device: torch device for the model.
        flow_steps: Euler steps of the flow-matching action expert.
        step_scale: metres per normalised unit (inputs are divided by it, outputs
            multiplied by it).
        execute_first_waypoint: the evaluator moves to waypoint index 1; prepending
            the current position makes it execute the model's first waypoint, so the
            history the model sees stays ~1 unit apart as in training.
        max_target_dist: if set, the goal is clipped to this distance (metres) before
            being encoded, keeping its direction.
        arrive_radius: declare arrival when closer than this to the goal (metres).
        image_size: optional (width, height) to resize the observations to.
    """

    def __init__(
        self,
        model_path,
        device="cuda:0",
        flow_steps=5,
        step_scale=1.0,
        execute_first_waypoint=True,
        max_target_dist=None,
        arrive_radius=0.5,
        image_size=None,
        view="front",
        verbose=True,
        **kwargs,
    ):
        self.net = SocialNavModel(model_path, device=device, flow_steps=flow_steps)
        self.step_scale = float(step_scale)
        self.execute_first_waypoint = execute_first_waypoint
        self.max_target_dist = max_target_dist
        self.arrive_radius = arrive_radius
        self.image_size = tuple(image_size) if image_size else None
        self.view = view
        self.verbose = verbose
        self.reset()

    def reset(self):
        self._frames = []  # [(camera-to-world 4x4, PIL image)]

    def _to_pil(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img, dtype=np.uint8))
        img = img.convert("RGB")
        if self.image_size:
            img = img.resize(self.image_size, Image.BILINEAR)
        return img

    def _input_waypoints(self, frames, observation):
        cur_pose = observation.rotation
        # History positions in the current body frame ([front, left, up]).
        hist_world = np.stack([pose[:3, 3] for pose, _ in frames])
        hist_local = (cur_pose[:3, :3].T @ (hist_world - cur_pose[:3, 3]).T).T
        hist = front_left_to_model(hist_local[:, :2])

        target = front_left_to_model(observation.target_position)
        dist = float(np.linalg.norm(target))
        if self.max_target_dist and dist > self.max_target_dist:
            target = target * (self.max_target_dist / dist)

        return np.concatenate([hist, target[None]], axis=0) / self.step_scale

    @torch.no_grad()
    def _infer(self, images, input_waypoints):
        net = self.net
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": PROMPT})
        text = net.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        ) + "<|im_end|>"
        inputs = net.processor(text=text, images=images, padding=True, return_tensors="pt").to(net.device)
        inputs["input_waypoints"] = torch.tensor(input_waypoints, dtype=torch.float32)[None].to(net.device)

        outputs = net.model(
            **inputs,
            train=False,
            train_branch="fm",
            num_samples=1,
            special_token2id=net.special_token2id,
        )
        return outputs[0].squeeze(0).float().cpu().numpy()

    def predict(self, observation: Observation) -> WaypointPrediction:
        self._frames.append((observation.rotation.copy(), self._to_pil(observation.images[self.view])))
        self._frames = self._frames[-NUM_FRAMES:]
        # Pad the start of an episode by repeating the oldest frame.
        frames = [self._frames[0]] * (NUM_FRAMES - len(self._frames)) + self._frames

        input_waypoints = self._input_waypoints(frames, observation)
        wp = self._infer([img for _, img in frames], input_waypoints) * self.step_scale

        waypoint = wp
        if self.execute_first_waypoint:
            waypoint = np.concatenate([np.zeros((1, 2), dtype=wp.dtype), wp], axis=0)

        arrive = observation.distance_to_goal < self.arrive_radius
        if self.verbose:
            print(
                f"[SocialNav] step={observation.step_count} d2g={observation.distance_to_goal:.2f}m "
                f"target(model)={np.round(input_waypoints[-1], 2).tolist()} "
                f"wp[m]={np.round(wp, 2).tolist()}"
            )
        return WaypointPrediction(
            waypoint=waypoint,
            arrive=arrive,
            extra={"input_waypoints": input_waypoints.tolist(), "pred_waypoints_m": wp.tolist()},
        )
