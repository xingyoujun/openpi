import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# State / action layout: eef_9d (xyz + first two rotation-matrix columns) followed by the gripper (finger joint, rad).
PINE_ACTION_DIM = 10


def make_pine_example() -> dict:
    """Creates a random input example for the Pine policy."""
    return {
        "observation/eef_9d": np.random.rand(9),
        "observation/gripper": np.random.rand(7),  # joint vector, the last entry is the gripper
        "observation/ego_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_a_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_b_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PineInputs(transforms.DataTransformFn):
    """Converts Pine (UR7e + Robotiq) samples to the model input format, for both training and inference.

    The LeRobot dataset stores the eef pose and the gripper in separate columns (`*.eef_9d` and the 7-D joint
    vector whose last entry is the gripper), so they are concatenated here into a 10-D state / action.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": np.concatenate(
                [np.asarray(data["observation/eef_9d"]), np.asarray(data["observation/gripper"])[..., 6:7]]
            ),
            "image": {
                "base_0_rgb": _parse_image(data["observation/ego_image"]),
                "left_wrist_0_rgb": _parse_image(data["observation/wrist_a_image"]),
                "right_wrist_0_rgb": _parse_image(data["observation/wrist_b_image"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training.
        if "actions/eef_9d" in data:
            inputs["actions"] = np.concatenate(
                [np.asarray(data["actions/eef_9d"]), np.asarray(data["actions/gripper"])[..., 6:7]], axis=-1
            )

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PineOutputs(transforms.DataTransformFn):
    """Returns the 10-D (eef_9d + gripper) actions, dropping the model's padding dimensions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :PINE_ACTION_DIM])}
