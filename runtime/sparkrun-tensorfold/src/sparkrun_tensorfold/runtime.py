"""TensorFold runtime plugin for sparkrun.

TensorFold (https://github.com/ashhart/TensorFold, MIT) is an OpenAI-compatible
serving runtime for MLX- and CUDA-hosted checkpoints (EXL3 / MLX tensor
formats) with native speculative (MTP) drafting. This plugin launches it in
SOLO mode inside a CUDA container (nvcr pytorch base or a prebuilt TF image),
serving an EXL3 checkpoint from a local HF-cache mount.

Executor: docker (solo = `sleep infinity` container + exec of the serve
command) — same shape as the vllm runtimes. Cluster mode (2026-10-27,
spark-lanes muse window): the muspark-2dgx TP2 topology is mapped 1:1 onto
sparkrun's ranked native-cluster containers — ONE container per host running
`sleep infinity`; the per-rank exec is the author's own launch shape:

  rank 0 (head = hosts[0]): tensorfold-native serve /model --port ... (HTTP API)
  rank 1 (worker):          tf-muse follow /model /drafter

TF rendezvous is the engine's own one-shot RoCE mailbox (TF_TP_MASTER /
TF_TP_PORT env, pinned by the recipe, NOT sparkrun's init_port math), so the
vllm-distributed --nnodes/--node-rank flags are never used. The worker is
exec'd BEFORE the head (the author's start.sh ordering), the head's mailbox
port is awaited after its exec, and the API port is awaited last.

Volume mounts (host paths are estate-staged on both ranks by the launch that
runs this plugin; see spark-lanes muse staging receipts):
  <model_dir>   -> /model    (EXL3 pack tree, e.g. Muse-Spark-Open-exl3-1.8s)
  <drafter_dir> -> /drafter  (DFlash2 drafter pack)
  <zig_dir>     -> /zig      (zig-out: bin/tf-muse + native/bin/tensorfold-native)
Paths resolve from env overrides (TF_MUSE_MODEL_DIR / TF_MUSE_DRAFTER_DIR /
TF_MUSE_ZIG_DIR) with estate defaults, so different packs can be fired without
re-patching the plugin.
"""

from __future__ import annotations

import logging
import os
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

# Estate defaults for the muspark-2dgx TP2 layout (spark-lanes law: the pack
# sits at the same paths on both ranks).
_MUSE_MODEL_DIR = os.environ.get(
    "TF_MUSE_MODEL_DIR", "/home/jaita/models/muse-spark/Muse-Spark-Open-exl3-1.8s"
)
_MUSE_DRAFTER_DIR = os.environ.get(
    "TF_MUSE_DRAFTER_DIR", "/home/jaita/models/muse-spark/Muse-Spark-Open-DFlash2-exl3-4.0bpw"
)
_MUSE_ZIG_DIR = os.environ.get(
    "TF_MUSE_ZIG_DIR", "/home/jaita/muse/tensorfold-zig/zig-out"
)

# The Zig engine's runtime container flags copied from muspark-2dgx's
# start.sh zig_args(): infiniband device for the RoCE mailbox, IPC lock, and
# ulimits the GB10 unified-memory engine requires.
_MUSE_DOCKER_OPTS = [
    "--device",
    "/dev/infiniband",
    "--cap-add",
    "IPC_LOCK",
    "--ulimit",
    "memlock=-1",
    "--ulimit",
    "stack=67108864",
]


def _env_prefix(values: dict[str, str]) -> str:
    return " ".join("%s=%s" % (k, v) for k, v in sorted(values.items()))


class TensorFoldRuntime(RuntimePlugin):
    """TensorFold serving runtime (solo + muspark TP2 cluster, Docker executor)."""

    runtime_name = "tensorfold"
    # default image is supplied by the recipe's container: field; leave empty
    default_image_prefix = ""

    def cluster_strategy(self) -> str:
        return "native"

    def default_executor(self) -> str | None:
        return "docker"

    def get_family(self) -> str:
        return "tensorfold"

    def world_size(
        self,
        parallelism,
        *,
        recipe: "Recipe",
        cluster,
    ) -> int | None:
        # The muse TP2 topology is exactly num_nodes ranks: 1 rank per host.
        # Recipe tensor_parallel: 2 already yields this from tp*pp*dp; nothing
        # to override — return None to keep the default math.
        return None

    def get_extra_volumes(self) -> dict[str, str]:
        """Bind-mount the muse pack, drafter and zig build read-only at the
        author's fixed container paths (/model, /drafter, /zig)."""
        return {
            _MUSE_MODEL_DIR: "/model",
            _MUSE_DRAFTER_DIR: "/drafter",
            _MUSE_ZIG_DIR: "/zig",
        }

    def get_extra_docker_opts(self) -> list[str]:
        # Only add host-existence-guarded device flags at exec script level;
        # extra_docker_opts here are static for both ranks. The infiniband
        # node exists on both DGX Sparks (CX7/RoCE — the muse TP link).
        return list(_MUSE_DOCKER_OPTS)

    def get_common_env(self) -> dict[str, str]:
        # Weights come from the local mounts; do not hit the network.
        env = {}
        try:
            from sparkrun.runtimes._util import default_env_hf_offline
            env.update(default_env_hf_offline())
        except Exception:
            pass
        return env

    def get_extra_env(self) -> dict[str, str]:
        # Per-rank TF_TP_* identity is applied inline in generate_node_command;
        # the recipe env (NCCL_*, engine levers) rides in ctx.all_env already.
        env: dict[str, str] = {}
        for key in ("TF_MUSE_MODEL_DIR", "TF_MUSE_DRAFTER_DIR", "TF_MUSE_ZIG_DIR"):
            val = os.environ.get(key)
            if val:
                env[key] = val
        return env

    # ------------------------------------------------------------------
    # Command generation
    # ------------------------------------------------------------------

    def generate_command(
        self,
        recipe: "Recipe",
        overrides: dict[str, Any],
        is_cluster: bool,
        num_nodes: int = 1,
        head_ip: str | None = None,
        skip_keys: set[str] | frozenset[str] = frozenset(),
    ) -> str:
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
        """Per-rank exec command for the muspark TP2 topology.

        Rank 0 (head): tensorfold-native serve /model ... (binds the API
        port and the TF_TP_PORT mailbox).  Rank 1: tf-muse follow.
        """
        from sparkrun.core.recipe import extract_served_model_name_from_command

        config = recipe.build_config_chain(overrides)
        port = int(config.get("port") or _TF_CONFIG_DEFAULTS["port"])
        parallel = int(config.get("parallel") or 1)
        context = int(config.get("context") or config.get("context_length") or 0)
        max_tokens = int(config.get("max_tokens") or 32768)
        served = config.get("served_model_name") or getattr(recipe, "model", None)
        tp_master = config.get("tp_master_ip") or head_ip
        tf_tp_port = int(config.get("tf_tp_port") or 29681)
        comm = str(config.get("tp_comm") or "roce")
        roce_hca = config.get("tp_roce_hca") or "rocep1s0f0,roceP2p1s0f0"

        id_env = {
            "TF_TP_WORLD": str(num_nodes),
            "TF_TP_RANK": str(node_rank),
            "TF_TP_MASTER": str(tp_master),
            "TF_TP_PORT": str(tf_tp_port),
            "TF_COMM_BACKEND": comm,
            "TF_TP_ROCE_HCA": str(roce_hca),
            "TF_TP_ROCE_FAST": "1",
            "HOME": "/tmp",
        }
        rank_env = _env_prefix(id_env)

        drafter_arg = config.get("drafter_arg")  # container path, e.g. /drafter
        if node_rank == 0:
            serve = [
                "tensorfold-native",
                "serve",
                "/model",
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
                "--name",
                str(served),
                "--parallel",
                str(parallel),
                "--max-tokens",
                str(max_tokens),
            ]
            if context:
                serve.extend(["--context", str(context)])
            for fkey, flag in (("temperature", "--temperature"), ("top_p", "--top-p"), ("top_k", "--top-k")):
                val = config.get(fkey)
                if val not in (None, ""):
                    serve.extend([flag, str(val)])
            if drafter_arg:
                serve.extend(["--drafter", str(drafter_arg)])
            return "%s %s" % (rank_env, " ".join(serve))
        follow = ["tf-muse", "follow", "/model"]
        if drafter_arg:
            follow.append(str(drafter_arg))
        return "%s %s" % (rank_env, " ".join(follow))

    # ------------------------------------------------------------------
    # Cluster orchestration (muse TP2: worker-first exec, then head, then API)
    # ------------------------------------------------------------------

    def _cluster_rank_commands(
        self,
        recipe,
        overrides,
        head_ip: str,
        hosts: list[str],
        init_port: int,
        skip_keys,
    ) -> list[tuple[int, str, str]]:
        """(rank, host, command) for every rank, in exec order (rank 1 first)."""
        num_nodes = len(hosts)
        out: list[tuple[int, str, str]] = []
        for rank in range(num_nodes - 1, -1, -1):
            host = hosts[rank]
            cmd = self.generate_node_command(
                recipe=recipe,
                overrides=overrides,
                head_ip=head_ip,
                num_nodes=num_nodes,
                node_rank=rank,
                init_port=init_port,
                skip_keys=skip_keys,
                hosts=hosts,
            )
            out.append((rank, host, cmd))
        return out

    def _run_cluster(
        self,
        hosts: list[str],
        image: str,
        serve_command: str = "",
        recipe=None,
        overrides=None,
        *,
        cluster_id: str = "sparkrun0",
        env: dict[str, str] | None = None,
        cache_dir: str | None = None,
        config=None,
        dry_run: bool = False,
        detached: bool = True,
        comm_env=None,
        ib_ip_map=None,
        ib_iface_map=None,
        init_port: int = 25000,
        skip_keys: set[str] | frozenset[str] = frozenset(),
        **kwargs,
    ) -> int:
        """Muse TP2 cluster: one ranked container per host + per-rank exec.

        Mirrors run_native_cluster but execs the worker (rank 1, tf-muse
        follow) BEFORE the head (the author's start.sh ordering) and waits on
        the head's API /v1/models rather than an init port.
        """
        from sparkrun.runtimes._cluster_ops import (
            ClusterContext,
            cleanup_ranked_containers,
            exec_serve_on_container,
            dump_serve_log,
            launch_containers_parallel,
            run_pre_serve_hooks,
        )
        from sparkrun.orchestration.primitives import wait_for_port, detect_host_ip
        from sparkrun.runtimes._init_network import (
            InitNetworkCandidates,
            select_init_network,
        )

        executor = self._resolve_executor()
        ctx = ClusterContext.build(
            runtime=self,
            hosts=hosts,
            image=image,
            cluster_id=cluster_id,
            env=env,
            cache_dir=cache_dir,
            config=config,
            dry_run=dry_run,
            cluster=kwargs.pop("cluster", None),
            recipe=recipe,
            placement=kwargs.pop("placement", None),
            runtime_cache=kwargs.pop("runtime_cache", None),
        )
        num_nodes = ctx.num_nodes
        if num_nodes != 2:
            logger.error("[tensorfold] muse TP2 orchestration supports exactly 2 nodes, got %d", num_nodes)
            if dry_run:
                return 0
            return 1

        try:
            head_ip = detect_host_ip(ctx.head_host, ssh_kwargs=ctx.ssh_kwargs, dry_run=ctx.dry_run)
        except RuntimeError as e:
            logger.error("%s", e)
            return 1
        resolved_hosts = list(hosts)
        try:
            init_selection = select_init_network(
                ctx,
                InitNetworkCandidates.from_resolved(head_ip, resolved_hosts, ib_ip_map or {}),
            )
            head_ip = init_selection.head_ip
            resolved_hosts = list(init_selection.hosts)
        except Exception:
            logger.debug("init network selection failed; keeping raw hosts", exc_info=True)

        config_chain = recipe.build_config_chain(overrides) if recipe is not None else None
        ranks = self._cluster_rank_commands(recipe, overrides, head_ip, resolved_hosts, init_port, skip_keys)

        self._print_cluster_banner(
            "TensorFold Muse TP2 Cluster",
            ctx.hosts,
            ctx.image,
            ctx.cluster_id,
            {"API Port": int(config_chain.get("port") or 8888) if config_chain else 8888},
            ctx.dry_run,
        )
        for rank, host, cmd in ranks:
            logger.info("Rank %d on %s:" % (rank, host))
            for line in cmd.strip().splitlines():
                logger.info("  %s", line)
        if ctx.dry_run:
            return 0

        # Step 1: cleanup ranked containers
        cleanup_ranked_containers(ctx, executor)

        # Step 2: launch containers (sleep infinity), no comm-env injection —
        # the muse TP link is the engine's own RoCE mailbox; NCCL settings
        # come from the recipe env verbatim.
        from sparkrun.orchestration.ssh import run_remote_script

        all_nodes = [(ctx.hosts[rank], rank, executor.node_container_name(ctx.cluster_id, rank)) for rank in range(num_nodes)]
        rc = launch_containers_parallel(
            ctx,
            [(host, cname) for host, _rank, cname in all_nodes],
            executor,
            None,
            extra_docker_opts=(self.get_extra_docker_opts() or None),
            runtime=self,
            recipe=recipe,
            container_ranks={cname: rank for _h, rank, cname in all_nodes},
        )
        if rc != 0:
            return rc

        # Step 3: pre-serve hooks
        run_pre_serve_hooks(self, ctx, [(host, cname) for host, _rank, cname in all_nodes], recipe, overrides, trust=kwargs.get("trust", False), cache_dir=cache_dir)

        # Step 4: exec rank 1 (worker follow) first, then rank 0 (head serve)
        for rank, host, cmd in ranks:
            cname = executor.node_container_name(ctx.cluster_id, rank)
            rc = exec_serve_on_container(ctx, executor, host, cname, cmd, detached=True)
            if rc != 0:
                logger.error("rank %d serve exec failed on %s", rank, host)
                from sparkrun.runtimes._cluster_ops import cleanup_after_failure
                cleanup_after_failure(ctx, executor, reason=f"rank {rank} serve exec failed")
                return 1

        # Step 5: wait for the API on the head (weights load takes minutes)
        api_port = int(config_chain.get("port") or 8888) if config_chain else 8888
        logger.info("Waiting for API http://%s:%d/v1/models (load takes minutes)...", ctx.head_host, api_port)
        ready = wait_for_port(
            ctx.head_host,
            api_port,
            max_retries=360,
            retry_interval=5,
            ssh_kwargs=ctx.ssh_kwargs,
            dry_run=ctx.dry_run,
            container_name=executor.node_container_name(ctx.cluster_id, 0),
        )
        if not ready:
            logger.error("Head API %s:%d never came up.", ctx.head_host, api_port)
            dump_serve_log(ctx.head_host, executor.node_container_name(ctx.cluster_id, 0), ctx.ssh_kwargs, dry_run=ctx.dry_run)
            from sparkrun.runtimes._cluster_ops import cleanup_after_failure
            cleanup_after_failure(ctx, executor, reason="head API never became ready")
            return 1

        self._print_connection_info(ctx.hosts, ctx.cluster_id, per_node_logs=True)
        return 0

    def _stop_cluster(
        self,
        hosts: list[str],
        cluster_id: str,
        config=None,
        dry_run: bool = False,
    ) -> int:
        """Stop the muse TP2 cluster: ranked containers on both ranks."""
        return self._stop_native_cluster(hosts, cluster_id, config=config, dry_run=dry_run)

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
        if recipe.executor and recipe.executor not in ("docker", None):
            issues.append(
                "[tensorfold] only executor=docker supported; got %r (using docker)."
                % recipe.executor
            )
        return issues