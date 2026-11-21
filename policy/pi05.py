"""LeRobot PI0.5 adapter for the Reality Check schema-v2 HTTP contract.

This is a real checkpoint-backed adapter, not a fake action policy. It expects
an embodiment-specific PI0.5 checkpoint at /weights. In particular, the public
pi05_base weights are a fine-tuning starting point, not a UR12e policy.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from rc_sdk.policy import Policy

WEIGHTS_DIR = Path("/weights")
MAX_JOINT_DELTA = 0.05
ARM_DIM = 6

# Reality Check's three_view images are wrist/left/right. PI0.5 checkpoints
# use fixed-base/left-wrist/right-wrist slots. The benchmark has no fixed-base
# view in three_view, so this mapping is only a mechanical adapter; fine-tune
# on the same camera mapping before expecting useful actions.
CAMERA_MAP = {
    "observation.images.base_0_rgb": "left",
    "observation.images.left_wrist_0_rgb": "wrist",
    "observation.images.right_wrist_0_rgb": "right",
}


def _decode_image(encoded: Any, name: str) -> torch.Tensor:
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"required camera image is missing: {name}")
    try:
        raw = base64.b64decode(encoded, validate=True)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        array = np.asarray(image, dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"camera image cannot be decoded: {name}") from exc
    # LeRobot's image pipeline expects CHW float RGB in [0, 1]. The policy
    # model handles resize/pad to its configured 224x224 input resolution.
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float() / 255.0


def _joint_state(obs: dict[str, Any]) -> tuple[torch.Tensor, list[float]]:
    proprio = obs.get("proprio")
    joints = proprio.get("joint_pos") if isinstance(proprio, dict) else None
    if not isinstance(joints, (list, tuple)) or len(joints) < ARM_DIM:
        raise ValueError("PI0.5 requires six joint positions in proprio.joint_pos")
    values = [float(x) for x in joints[:ARM_DIM]]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("joint positions must be finite")
    return torch.tensor(values, dtype=torch.float32), values


def _stable_seed(episode_id: Any, step: Any) -> int:
    payload = f"{episode_id!s}:{step!s}".encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big")


class Pi05Policy(Policy):
    def load(self) -> None:
        if not WEIGHTS_DIR.is_dir() or not any(WEIGHTS_DIR.iterdir()):
            raise RuntimeError("a pinned PI0.5 checkpoint must be mounted at /weights")

        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05 import PI05Policy as LeRobotPI05Policy
        from lerobot.configs.policies import PreTrainedConfig

        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = PreTrainedConfig.from_pretrained(str(WEIGHTS_DIR))
        if config.type != "pi05":
            raise RuntimeError(f"/weights contains policy type {config.type!r}, expected 'pi05'")
        config.device = device
        # LeRobot 0.7 deserializes config.dtype as a torch.dtype, not always a
        # string. Normalize both forms before applying the optional override.
        dtype_map = {
            "float32": torch.float32,
            "torch.float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "torch.bfloat16": torch.bfloat16,
        }
        saved_dtype = str(config.dtype)
        dtype_name = __import__("os").environ.get("RC_PI05_DTYPE", saved_dtype.removeprefix("torch."))
        if dtype_name not in {"float32", "bfloat16"}:
            raise RuntimeError("RC_PI05_DTYPE must be float32 or bfloat16")
        config.dtype = dtype_map[dtype_name]

        print(f"[pi05] loading /weights on {device} ({dtype_name})", flush=True)
        self.policy = LeRobotPI05Policy.from_pretrained(str(WEIGHTS_DIR), config=config)
        self.policy.to(device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(WEIGHTS_DIR),
            preprocessor_overrides=self._preprocessor_overrides(device),
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.device = device
        self._memo: tuple[Any, Any, dict[str, Any]] | None = None
        self._episode: Any = None

    @staticmethod
    def _preprocessor_overrides(device: str) -> dict[str, dict[str, Any]]:
        overrides: dict[str, dict[str, Any]] = {"device_processor": {"device": device}}
        tokenizer_path = __import__("os").environ.get("RC_PI05_TOKENIZER_PATH")
        if tokenizer_path:
            local_tokenizer = Path(tokenizer_path)
            if not local_tokenizer.is_dir():
                raise RuntimeError("RC_PI05_TOKENIZER_PATH is set but is not a directory")
            overrides["tokenizer_processor"] = {"tokenizer_name": str(local_tokenizer)}
        return overrides

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        episode_id = obs.get("episode_id")
        step = obs.get("step")
        if self._memo is not None and self._memo[0] == episode_id and self._memo[1] == step:
            return self._memo[2]

        current_state, joint_positions = _joint_state(obs)
        images = obs.get("images")
        if not isinstance(images, dict):
            raise ValueError("PI0.5 requires an images object")
        batch: dict[str, Any] = {
            "observation.state": current_state,
            "task": obs.get("instruction") or "manipulate the object as instructed",
        }
        for feature, camera in CAMERA_MAP.items():
            batch[feature] = _decode_image(images.get(camera), camera)

        batch = self.preprocess(batch)
        # The HTTP contract can replay an identical request. Cache the response
        # below, and seed the stochastic flow sampler from the request identity.
        # PI0.5's released base config has no temporal memory; select_action's
        # short action queue is reset only at episode boundaries, as intended.
        if episode_id != self._episode or step == 0:
            self.policy.reset()
            self._episode = episode_id
            seed = _stable_seed(episode_id, step)
            with torch.random.fork_rng(devices=[0] if self.device == "cuda" else []):
                torch.manual_seed(seed)
                if self.device == "cuda":
                    torch.cuda.manual_seed_all(seed)
                with torch.inference_mode():
                    action = self.postprocess(self.policy.select_action(batch))
        else:
            with torch.inference_mode():
                action = self.postprocess(self.policy.select_action(batch))

        if isinstance(action, dict):
            action = action.get("action")
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        values = action.detach().to(dtype=torch.float32, device="cpu").reshape(-1).tolist()
        if len(values) < 7 or not all(math.isfinite(float(x)) for x in values[:7]):
            raise ValueError("PI0.5 output must contain six arm targets and one gripper target")

        # PI0.5 predicts absolute actions by default. Translate its first six
        # dimensions to the incremental joint_delta contract; keep commands
        # inside the legal envelope before the evaluator sees them.
        deltas = []
        for target, current in zip(values[:ARM_DIM], joint_positions):
            delta = max(-MAX_JOINT_DELTA, min(MAX_JOINT_DELTA, float(target) - current))
            deltas.append(delta)
        gripper = max(0.0, min(1.0, float(values[6])))
        result = {"type": "joint_delta", "values": deltas, "gripper": gripper}
        self._memo = (episode_id, step, result)
        return result
