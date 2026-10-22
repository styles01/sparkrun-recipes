"""sparkrun-flare — out-of-tree sparkrun runtime plugin for NVIDIA Flare (flare-local-inference).

Serves a PREPARED Flare runtime package (ignite export, docs/model-preparation.md) with the native
flare executable. Per-node distribution uses flare's manual-rank path (docs/multi-host.md
"Start ranks manually"): rank 0 rendezvous 0.0.0.0:<rank-port>, peers rendezvous <head_ip>:<rank-port>
--gpus 0. The package file path comes from recipe key `flare_package`; the unattended dev-publisher
acceptance flags come from `flare_key_fp` (override in-window) + the hardcoded `local-development`
namespace. Runs inside the flare container (docs/container.md; root Dockerfile, ARM64, no CUDA).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sparkrun.runtimes.base import RuntimePlugin

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe


class FlareRuntime(RuntimePlugin):
    """NVIDIA Flare serving runtime (docker executor, native multi-host manual ranks)."""

    runtime_name = "flare"
    default_image_prefix = ""

    def get_family(self) -> str:
        return "flare"

    def cluster_strategy(self) -> str:
        # native multi-node: one container per host, per-node --rank/--rendezvous flags
        return "native"

    def default_executor(self) -> str | None:
        return "docker"

    def prepare(self, recipe: "Recipe", hosts: list[str],
                config: "Any | None" = None, dry_run: bool = False,
                transfer_mode: str = "auto", overrides: dict[str, Any] | None = None) -> None:
        # Flare distributes its own prepared package (ignite export → rank payloads / orbit agents).
        # Do NOT add the HF checkpoint to sparkrun's distribution config: the package, not the
        # checkpoint, is what the runtime reads. Keep the default (no draft-model handling).
        return None

    def _base(self, recipe, overrides, skip_keys):
        config = recipe.build_config_chain(overrides)
        rendered = recipe.render_command(config)
        if not rendered:
            raise ValueError(
                "[flare] recipe must carry an explicit command: template"
                " (flare run {flare_package} --publisher-keys ...)")
        served = config.get("served_model_name")
        if served:
            rendered = self._augment_served_model_name(rendered, config, "--served-model-name", skip_keys)
        # Dev-publisher acceptance pair only when the fingerprint is supplied; an empty
        # --dangerously-approve-key is a guaranteed arg-parse death.
        fp = config.get("flare_key_fp")
        if not fp:
            lines = [ln for ln in rendered.splitlines()
                     if "--dangerously-approve-key" not in ln
                     and "--dangerously-accept-publisher" not in ln]
            rendered = "\n".join(lines)
        return rendered, config

    def generate_command(
        self,
        recipe: "Recipe",
        overrides: dict[str, Any],
        is_cluster: bool,
        num_nodes: int = 1,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
        """Solo or TP1: rank 0 with a single-host rendezvous."""
        rendered, config = self._base(recipe, overrides, set(skip_keys or set()))
        rank_port = config.get("flare_rank_port") or 29800
        return rendered + " --rank 0 --rendezvous 0.0.0.0:%s --gpus 0" % rank_port

    def generate_node_command(
        self,
        recipe: "Recipe",
        overrides: dict[str, Any],
        head_ip: str,
        num_nodes: int,
        node_rank: int,
        init_port: int = 25000,
        skip_keys: set[str] | frozenset[str] = frozenset(),
        hosts: list[str] | None = None,
        placement=None,
    ) -> str:
        """Per-node flare run command: --rank N --rendezvous <head>:<rank-port> --gpus 0."""
        rendered, config = self._base(recipe, overrides, set(skip_keys or set()))
        rank_port = config.get("flare_rank_port") or 29800
        if node_rank == 0:
            rend = "0.0.0.0:%s" % rank_port
        else:
            rend = "%s:%s" % (head_ip, rank_port)
        parts = [rendered]
        parts.extend(["--rank %d" % node_rank,
                      "--rendezvous %s" % rend,
                      "--gpus 0"])
        return " ".join(parts)

    def version_commands(self) -> dict[str, str]:
        cmds = super().version_commands()
        cmds["flare"] = "flare --version 2>/dev/null | head -1 || echo unknown"
        return cmds

    def validate_recipe(self, recipe: "Recipe") -> list[str]:
        issues = super().validate_recipe(recipe)
        cfg = recipe.defaults or {}
        if not cfg.get("flare_package"):
            issues.append("[flare] defaults.flare_package must point at the prepared package path")
        if recipe.executor and recipe.executor not in ("docker", None, ""):
            issues.append(
                "[flare] only executor=docker supported; got %r (using docker)." % recipe.executor)
        return issues