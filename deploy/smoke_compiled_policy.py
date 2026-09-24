#!/usr/bin/env python3
"""Exercise a compiled policy gateway without a simulator or robot."""

import argparse
import json

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--expected-policy", required=True)
    parser.add_argument("--prompt", default="move the object to the bowl")
    args = parser.parse_args()

    client = WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    if metadata.get("policy_id") != args.expected_policy:
        raise RuntimeError(
            f"Expected {args.expected_policy!r}; server reports {metadata.get('policy_id')!r}."
        )
    response = client.infer(
        {
            "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
            "prompt": args.prompt,
            "_robolab": {"episode": "smoke", "env_id": 0, "chunk_index": 0},
        }
    )
    actions = np.asarray(response["actions"])
    if actions.shape != (15, 8) or not np.isfinite(actions).all():
        raise RuntimeError(f"Invalid compiled policy actions: shape={actions.shape}")
    print(
        json.dumps(
            {
                "policy_id": metadata["policy_id"],
                "identity": metadata.get("identity_sha256"),
                "action_shape": list(actions.shape),
                "finite": True,
                "server_timing": response.get("server_timing"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
