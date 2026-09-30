"""TensorFold runtime plugin for sparkrun.

TensorFold (https://github.com/ashhart/TensorFold, MIT) is an OpenAI-compatible
serving runtime for MLX- and CUDA-hosted checkpoints (EXL3 / MLX tensor
formats) with native speculative (MTP) drafting. This plugin launches it in
SOLO mode inside a CUDA container (nvcr pytorch base or a prebuilt TF image),
serving an EXL3 checkpoint from a local HF-cache mount.

Executor: docker (solo = `sleep infinity` container + exec of the serve
command) — same shape as the vllm runtimes. Recipes must pin max_nodes: 1
(TensorFold has no distributed path).

Serve command shape (generated from recipe defaults):
    tensorfold serve <model> --host 0.0.0.0 --port 8000 --parallel 4

`--parallel` = concurrent decode streams (TensorFold-native knob; maps from
recipe key `parallel`). All other flags via TF_FLAG_MAP.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

# Recipe default key -> tensorfold CLI flag
_TF_FLAG_MAP = {
    "host": "--host",
    "port": "--port",
    "parallel": "--parallel",
    "context": "--context",
    "context_length": "--context",
    "vision": "--vision",
    "draft": "--draft",
    "mtp_drafts": "--mtp-drafts",
    "mtp_confidence": "--mtp-confidence",
}

_TF_BOOL_FLAGS = frozenset({"vision"})

_TF_CONFIG_DEFAULTS: dict[str, Any] = {
    "host": "0.0.0.0",
    "port": 8000,
    "parallel": 5,
    "context": 262144,
}


class TensorFoldRuntime(RuntimePlugin):
    """TensorFold serving runtime (solo, Docker executor)."""

    runtime_name = "tensorfold"
    # default image is supplied by the recipe's container: field; leave empty
    default_image_prefix = ""

    def cluster_strategy(self) -> str:
        return "native"

    def default_executor(self) -> str | None:
        return "docker"

    def get_family(self) -> str:
        return "tensorfold"

    def get_extra_volumes(self) -> dict[str, str]:
        """Mount the local models dir so EXL3 packs resolve without HF pulls.

        Recipes may override with their own volume via `-o`; this default
        covers the standard ~/models/hf layout on a Spark host.
        """
        return {}

    def get_common_env(self) -> dict[str, str]:
        # Weights come from the local mount; do not hit the network.
        env = {}
        try:
            from sparkrun.runtimes._util import default_env_hf_offline
            env.update(default_env_hf_offline())
        except Exception:
            pass
        return env

    def generate_command(
        self,
        recipe: "Recipe",
        overrides: dict[str, Any],
        is_cluster: bool,
        num_nodes: int = 1,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        if is_cluster or num_nodes > 1:
            raise ValueError(
                "tensorfold supports single-node serving only; "
                "set max_nodes: 1"
            )
        config = recipe.build_config_chain(overrides)
        # apply built-in defaults for unset keys
        for key, value in _TF_CONFIG_DEFAULTS.items():
            if config.get(key) is None:
                try:
                    config.set(key, value)
                except Exception:
                    pass
        rendered = recipe.render_command(config)
        if rendered:
            rendered = self._augment_served_model_name(rendered, config, "--served-model-name", skip_keys)
            rendered = self.strip_flags_from_command(rendered, skip_keys, _TF_FLAG_MAP, _TF_BOOL_FLAGS)
            return rendered
        parts = ["tensorfold", "serve", str(recipe.model)]
        skip = set(skip_keys) or set()
        parts.extend(
            self.build_flags_from_map(
                config,
                _TF_FLAG_MAP,
                bool_keys=_TF_BOOL_FLAGS,
                skip_keys=skip,
            )
        )
        return " ".join(parts)

    # NOTE: _augment_served_model_name is inherited from RuntimePlugin in the
    # installed sparkrun (see tokenary/llama_cpp precedent). If missing, no-op.
    def _augment_served_model_name(self, rendered, config, flag, skip_keys):
        method = getattr(super(), "_augment_served_model_name", None)
        if callable(method):
            return method(rendered, config, flag, skip_keys)
        served = config.get("served_model_name")
        if served and flag in rendered:
            return rendered
        return rendered

    def version_commands(self) -> dict[str, str]:
        cmds = super().version_commands()
        cmds["tensorfold"] = "tensorfold --version 2>/dev/null | head -1 || echo unknown"
        return cmds

    def validate_recipe(self, recipe: "Recipe") -> list[str]:
        issues = super().validate_recipe(recipe)
        if recipe.max_nodes and recipe.max_nodes > 1:
            issues.append(
                "[tensorfold] multi-node unsupported; max_nodes=%d ignored."
                % recipe.max_nodes
            )
        if recipe.executor and recipe.executor not in ("docker", None):
            issues.append(
                "[tensorfold] only executor=docker supported; got %r (using docker)."
                % recipe.executor
            )
        return issues