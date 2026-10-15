import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# State / action layout: [left eef_9d, left gripper, right eef_9d, right gripper] (eef_9d = xyz + first two rotation-matrix
# columns; gripper = finger joint).
PIPER_ACTION_DIM = 20


def make_pine_piper_example() -> dict:
    """Creates a random input example for the Pine Piper (dual-arm) policy."""
    return {
        "observation/eef_9d": np.random.rand(18),
        "observation/joints": np.random.rand(14),  # 2 x (6 joints + gripper)
        "observation/ego_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _merge(eef_9d, joints) -> np.ndarray:
    """Interleaves the two arms' eef_9d (18-D) with their grippers (entries 6 and 13 of the 14-D joint vector)."""
    eef_9d, joints = np.asarray(eef_9d), np.asarray(joints)
    return np.concatenate([eef_9d[..., :9], joints[..., 6:7], eef_9d[..., 9:18], joints[..., 13:14]], axis=-1)


@dataclasses.dataclass(frozen=True)
class PinePiperInputs(transforms.DataTransformFn):
    """Converts Pine Piper dual-arm samples to the model input format, for both training and inference."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": _merge(data["observation/eef_9d"], data["observation/joints"]),
            "image": {
                "base_0_rgb": _parse_image(data["observation/ego_image"]),
                "left_wrist_0_rgb": _parse_image(data["observation/left_wrist_image"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_wrist_image"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training.
        if "actions/eef_9d" in data:
            inputs["actions"] = _merge(data["actions/eef_9d"], data["actions/joints"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PinePiperOutputs(transforms.DataTransformFn):
    """Returns the 20-D dual-arm actions, dropping the model's padding dimensions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :PIPER_ACTION_DIM])}
