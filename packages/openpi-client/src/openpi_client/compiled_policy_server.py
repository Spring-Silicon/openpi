"""Serve the compiled PI0.5 B580 artifacts through the OpenPI WebSocket protocol."""

import argparse
import importlib.util
import logging
import pathlib
import signal
import subprocess
import sys
import threading
from typing import Any

import numpy as np

from openpi_client import base_policy
from openpi_client.websocket_policy_server import WebsocketPolicyServer

_VARIANTS = ("pi05_compiled_regular", "pi05_compiled_optimized")
logger = logging.getLogger(__name__)


def _load_unit(artifact: pathlib.Path, filename: str, module_name: str):
    unit_dir = artifact / "unit"
    sys.path.insert(0, str(unit_dir / "src"))
    path = unit_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Compiled policy adapter not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load compiled policy adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class CompiledPolicy(base_policy.BasePolicy):
    """Translate OpenPI DROID observations to a compiled artifact policy."""

    def __init__(self, variant: str, backend: Any, observation_type: type) -> None:
        self._variant = variant
        self._backend = backend
        self._observation_type = observation_type
        self._default_chunks: dict[str, int] = {}
        self._counter_lock = threading.Lock()
        self.metadata = {
            **backend.metadata,
            "policy_id": variant,
            "action_horizon": int(backend.spec.horizon),
            "action_dim": int(backend.spec.action_dim),
            "gateway": "openpi_client.compiled_policy_server",
            "identity_sha256": backend.spec.identity_sha256,
        }

    def infer(self, obs: dict) -> dict:
        orchestration = obs.get("_robolab") or {}
        env_id = int(orchestration.get("env_id", 0))
        if orchestration:
            episode_id = f"robolab-{orchestration.get('episode', 0)}-env-{env_id}"
            chunk_index = int(orchestration.get("chunk_index", 0))
        else:
            episode_id = "openpi-default"
            with self._counter_lock:
                chunk_index = self._default_chunks.get(episode_id, 0)
                self._default_chunks[episode_id] = chunk_index + 1

        observation = self._observation_type(
            episode_id=episode_id,
            env_id=env_id,
            step=chunk_index * int(self._backend.spec.horizon),
            chunk_index=chunk_index,
            instruction=str(obs.get("prompt", "")),
            seed=0,
            images={
                "exterior_image_1_left": np.asarray(
                    obs["observation/exterior_image_1_left"], dtype=np.uint8
                ),
                "wrist_image_left": np.asarray(obs["observation/wrist_image_left"], dtype=np.uint8),
            },
            joint_position=np.asarray(obs["observation/joint_position"], dtype=np.float64),
            gripper_position=np.asarray(obs["observation/gripper_position"], dtype=np.float64),
        )
        actions = np.asarray(self._backend.infer([observation])[0], dtype=np.float32)
        expected = (int(self._backend.spec.horizon), int(self._backend.spec.action_dim))
        if actions.shape != expected:
            raise RuntimeError(f"Compiled backend returned {actions.shape}; expected {expected}.")
        if not np.isfinite(actions).all():
            raise RuntimeError("Compiled backend returned non-finite actions.")
        return {"actions": actions}

    def close(self) -> None:
        self._backend.close()


def _regular_backend(artifact: pathlib.Path, gpu: int, work_dir: pathlib.Path):
    module = _load_unit(artifact, "unit_ov.py", "spring_compiled_regular")
    backend = module.OvPolicy(
        "pi05_compiled_regular",
        module.Assets(artifact / "unit" / "assets_ckptA", artifact / "serve_ov_a.sh", "A"),
        gpu,
        1,
        "pi05-compiled-regular/seed0",
        work_dir / "case",
        work_dir / "openvino.log",
        action_decode="native_joint_position",
    )
    return backend, module.Observation


def _clean_optimized_containers() -> None:
    listed = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}", "--filter", "name=^pi05-mux-"],
        check=True,
        capture_output=True,
        text=True,
    )
    for name in listed.stdout.split():
        logger.warning("Stopping stale optimized policy container %s", name)
        subprocess.run(["docker", "kill", name], check=True, capture_output=True, text=True)


def _optimized_backend(artifact: pathlib.Path, gpu: int, work_dir: pathlib.Path):
    _clean_optimized_containers()
    module = _load_unit(artifact, "unit_b580.py", "spring_compiled_optimized")
    backend = module.B580Policy(
        "pi05_compiled_optimized",
        module.Assets(
            artifact / "unit" / "assets",
            artifact / "campaign13" / "share" / "weights-A",
            artifact / "run_mux_serve.sh",
        ),
        gpu,
        1,
        "pi05-compiled-optimized/seed0",
        work_dir / "case",
        work_dir / "fused.log",
    )
    return backend, module.Observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=_VARIANTS)
    parser.add_argument("--artifact-dir", required=True, type=pathlib.Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--work-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    artifact = args.artifact_dir.resolve()
    work_dir = args.work_dir or pathlib.Path(f"/dev/shm/{args.variant}")
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.variant == "pi05_compiled_regular":
        backend, observation_type = _regular_backend(artifact, args.gpu, work_dir)
    else:
        backend, observation_type = _optimized_backend(artifact, args.gpu, work_dir)

    policy = CompiledPolicy(args.variant, backend, observation_type)
    try:
        WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()
    finally:
        policy.close()


if __name__ == "__main__":
    main()
