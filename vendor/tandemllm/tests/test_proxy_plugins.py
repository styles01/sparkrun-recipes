"""The plugins inside vLLM and SGLang (plugins/, P6), against stand-ins shaped as the releases define them.

Neither engine installs on a laptop without a GPU, so the modules the plugins import are built here
with the names, fields and call shapes of the tagged sources:

    vLLM v0.30.0 (tag ced6857)   vllm/v1/metrics/loggers.py: StatLoggerBase (abstract __init__,
        record(scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0),
        log_engine_initialized); vllm/v1/metrics/stats.py: FinishedRequestStats, SchedulerStats,
        PrefixCacheStats, IterationStats; vllm/v1/spec_decode/metrics.py: SpecDecodingStats;
        vllm/v1/engine/__init__.py: FinishReason (an IntEnum whose str is the reason);
        vllm/plugins/__init__.py: an entry point's load() is caught, the class must subclass
        StatLoggerBase
    SGLang v0.5.21 (tag e00930c)  srt/plugins/hook_registry.py: HookRegistry.register(target,
        hook, HookType.AFTER), the AFTER wrapper `hook(result, *args, **kwargs)`, a failed target
        logged and skipped by apply_hooks; srt/managers/tokenizer_manager.py:
        TokenizerManager.collect_metrics(state, recv_obj, i); srt/managers/io_struct.py: the
        parallel lists of BatchStrOutput; srt/observability: SchedulerMetricsCollector.log_stats,
        SchedulerStats with QueueCount, SchedulerReqTimeStats, APIServerReqTimeStats

The real engines run in the night pass (tools/proxy_engines_check.sh --plugins).

Run: python tests/test_proxy_plugins.py
"""

from __future__ import annotations

import abc
import dataclasses
import enum
import filecmp
import importlib
import json
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "plugins", "vllm"))
sys.path.insert(0, os.path.join(ROOT, "plugins", "sglang"))
from proxy_util import run  # noqa: E402


# ----------------------------------------------------------------------------------- vLLM stand-in

def _mod(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


class FinishReason(enum.IntEnum):
    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4

    def __str__(self):
        return ("stop", "length", "abort", "error", "repetition")[self.value]


@dataclasses.dataclass
class PrefixCacheStats:
    reset: bool = False
    requests: int = 0
    queries: int = 0
    hits: int = 0
    preempted_requests: int = 0
    preempted_queries: int = 0
    preempted_hits: int = 0


@dataclasses.dataclass
class SpecDecodingStats:
    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list = dataclasses.field(default_factory=list)
    num_draft_tokens_per_pos: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SchedulerStats:
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    num_skipped_waiting_reqs: int = 0
    step_counter: int = 0
    current_wave: int = 0
    kv_cache_usage: float = 0.0
    iteration_details: object = None
    prefix_cache_stats: PrefixCacheStats = dataclasses.field(default_factory=PrefixCacheStats)
    connector_prefix_cache_stats: object = None
    kv_cache_eviction_events: list = dataclasses.field(default_factory=list)
    spec_decoding_stats: object = None
    kv_connector_stats: object = None
    waiting_lora_adapters: dict = dataclasses.field(default_factory=dict)
    running_lora_adapters: dict = dataclasses.field(default_factory=dict)
    cudagraph_stats: object = None
    perf_stats: object = None


@dataclasses.dataclass
class FinishedRequestStats:
    finish_reason: FinishReason
    request_id: str | None = None
    e2e_latency: float = 0.0
    num_prompt_tokens: int = 0
    num_preemptions: int = 0
    num_generation_tokens: int = 0
    max_tokens_param: int | None = None
    queued_time: float = 0.0
    prefill_time: float = 0.0
    inference_time: float = 0.0
    decode_time: float = 0.0
    mean_time_per_output_token: float = 0.0
    is_corrupted: bool = False
    num_cached_tokens: int = 0


class IterationStats:
    def __init__(self):
        self.iteration_timestamp = time.time()
        self.num_generation_tokens = 0
        self.num_preempted_reqs = 0
        self.finished_requests: list = []
        self.num_corrupted_reqs = 0


class StatLoggerBase(abc.ABC):
    @abc.abstractmethod
    def __init__(self, vllm_config, engine_index: int = 0): ...

    @abc.abstractmethod
    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0): ...

    @abc.abstractmethod
    def log_engine_initialized(self): ...

    def log(self):
        pass


def install_vllm(with_base: bool = True) -> None:
    for k in [k for k in sys.modules if k == "vllm" or k.startswith("vllm.")
              or k.startswith("tandem_vllm_plugin")]:
        del sys.modules[k]
    _mod("vllm", __version__="0.30.0", __path__=[])
    _mod("vllm.v1", __path__=[])
    _mod("vllm.v1.metrics", __path__=[])
    loggers = _mod("vllm.v1.metrics.loggers")
    if with_base:
        loggers.StatLoggerBase = StatLoggerBase
    _mod("vllm.v1.metrics.stats", FinishedRequestStats=FinishedRequestStats,
         SchedulerStats=SchedulerStats, PrefixCacheStats=PrefixCacheStats,
         IterationStats=IterationStats)


def vllm_load(group_class):
    """vLLM v0.30.0's `load_stat_logger_plugin_factories` on one entry point's class."""
    if not isinstance(group_class, type) or not issubclass(group_class, StatLoggerBase):
        raise TypeError("Stat logger plugin must be a subclass of StatLoggerBase")
    return [group_class]


class Posts:
    def __init__(self, status=200):
        self.status = status
        self.bodies: list[dict] = []

    def __call__(self, url, token, data):
        self.bodies.append({"url": url, "token": token, "batch": json.loads(data)})
        return self.status


def fresh_push(mod, **kw):
    mod._SHARED.clear()
    return mod


# ----------------------------------------------------------------------------------- vLLM tests

def test_vllm_logger_reads_the_finished_requests():
    install_vllm()
    logger = importlib.import_module("tandem_vllm_plugin.logger")
    from tandem_vllm_plugin import _push
    fresh_push(_push)
    assert vllm_load(logger.TandemStatLogger) == [logger.TandemStatLogger]
    lg = logger.TandemStatLogger(None, 0)               # PerEngineStatLoggerAdapter's call
    it = IterationStats()
    it.iteration_timestamp = 1000.0
    it.finished_requests = [FinishedRequestStats(
        finish_reason=FinishReason.LENGTH, request_id="chatcmpl-px-abc", e2e_latency=2.5,
        num_prompt_tokens=1200, num_preemptions=1, num_generation_tokens=101,
        max_tokens_param=100, queued_time=0.25, prefill_time=0.5, inference_time=2.2,
        decode_time=1.7, mean_time_per_output_token=0.017, is_corrupted=False,
        num_cached_tokens=1024)]
    sched = SchedulerStats(num_running_reqs=3, num_waiting_reqs=1, kv_cache_usage=0.42,
                           prefix_cache_stats=PrefixCacheStats(requests=1, queries=1200,
                                                               hits=1024))
    lg.record(sched, it, None, 0)
    p = logger.pusher()
    items = p.requests.after(0)
    assert len(items) == 1
    r = items[0]
    assert r["id"] == "chatcmpl-px-abc" and r["finish_reason"] == "length"
    assert (r["prompt_tokens"], r["cached_tokens"], r["completion_tokens"]) == (1200, 1024, 101)
    assert (r["queue_ms"], r["prefill_ms"], r["decode_ms"], r["e2e_ms"]) == (250.0, 500.0,
                                                                              1700.0, 2500.0)
    assert r["preemptions"] == 1 and r["corrupted"] is False and r["seq"] == 1
    assert r["t_end"] == 1000.0 and r["engine_idx"] == 0
    # nothing of a prompt or an answer: only these keys
    assert set(r) <= {"id", "finish_reason", "prompt_tokens", "cached_tokens",
                      "completion_tokens", "preemptions", "queue_ms", "prefill_ms", "decode_ms",
                      "inference_ms", "e2e_ms", "tpot_ms", "max_tokens", "corrupted", "t_end",
                      "engine_idx", "seq"}


def test_vllm_logger_folds_steps_into_one_sample_a_second():
    install_vllm()
    logger = importlib.import_module("tandem_vllm_plugin.logger")
    from tandem_vllm_plugin import _push
    fresh_push(_push)
    lg = logger.TandemStatLogger(None, 0)
    t = 2000.0
    for k in range(25):                                  # 25 steps, 50 ms apart: 1.2 s
        it = IterationStats()
        it.iteration_timestamp = t + k * 0.05
        it.num_preempted_reqs = 1 if k == 3 else 0
        sd = SpecDecodingStats(num_spec_tokens=3, num_drafts=2, num_draft_tokens=6,
                               num_accepted_tokens=4, num_accepted_tokens_per_pos=[2, 1, 1])
        s = SchedulerStats(num_running_reqs=2, num_waiting_reqs=0, kv_cache_usage=0.1 + k / 100,
                           prefix_cache_stats=PrefixCacheStats(queries=100, hits=40),
                           spec_decoding_stats=sd,
                           kv_cache_eviction_events=[object()] if k == 5 else [])
        lg.record(s, it)
    steps = logger.pusher().steps.after(0)
    assert len(steps) == 1, steps                        # the window closed at k = 20 (1.0 s)
    s0 = steps[0]
    assert s0["steps"] == 21 and s0["running"] == 2 and abs(s0["kv_usage"] - 0.30) < 1e-9
    assert s0["prefix_queries"] == 2100 and s0["prefix_hits"] == 840
    assert s0["preempted"] == 1 and s0["evictions"] == 1
    assert s0["spec"] == {"drafts": 42, "draft_tokens": 126, "accepted": 84,
                          "accepted_per_pos": [42, 21, 21]}


def test_vllm_logger_never_raises_into_vllm():
    install_vllm()
    logger = importlib.import_module("tandem_vllm_plugin.logger")
    from tandem_vllm_plugin import _push
    fresh_push(_push)
    lg = logger.TandemStatLogger(None, 0)

    class Broken:                                       # a stats object of another shape
        @property
        def finished_requests(self):
            raise RuntimeError("renamed")
    for _ in range(logger.MAX_FAILURES):
        lg.record(None, Broken())
    assert lg._off is True
    lg.record(SchedulerStats(), IterationStats())        # off: a no-op, no exception
    # a renamed field is a null, not a failure
    it = IterationStats()
    it.finished_requests = [types.SimpleNamespace(request_id="x", finish_reason=None)]
    lg2 = logger.TandemStatLogger(None, 1)
    lg2.record(None, it, engine_idx=1)
    got = logger.pusher().requests.after(0)[-1]
    assert got["id"] == "x" and got["queue_ms"] is None and got["engine_idx"] == 1


def test_vllm_plugin_without_its_base_class_fails_at_import_only():
    """vLLM's loader catches an exception from an entry point's load() and starts without it."""
    install_vllm(with_base=False)
    try:
        importlib.import_module("tandem_vllm_plugin.logger")
    except ImportError:
        pass
    else:
        raise AssertionError("expected an ImportError for vLLM's loader to catch")
    install_vllm()


def test_vllm_pull_route_answers_with_the_ring_and_checks_the_token():
    install_vllm()
    importlib.import_module("tandem_vllm_plugin.logger")
    from tandem_vllm_plugin import _push, endpoint
    fresh_push(_push)
    from tandem_vllm_plugin.logger import pusher
    pusher().add_request({"id": "a"})
    pusher().add_request({"id": "b"})
    st, body = endpoint.answer(None, 1, 0, env={})
    assert st == 200 and [r["id"] for r in body["requests"]] == ["b"]
    assert body["engine"] == "vllm" and body["plugin"] == "tandemllm-vllm-plugin"
    st, _ = endpoint.answer("Bearer nope", 0, 0, env={"TANDEM_COLLECT_TOKEN": "t" * 20})
    assert st == 401
    st, body = endpoint.answer("Bearer " + "t" * 20, 0, 0, env={"TANDEM_COLLECT_TOKEN": "t" * 20})
    assert st == 200 and len(body["requests"]) == 2
    ep = endpoint.TandemStatsEndpoint()
    assert ep.name == "tandem_stats_endpoint" and ep.required_tasks is None


# ----------------------------------------------------------------------------------- the poster

def test_poster_batches_heartbeats_and_backs_off():
    from tandem_vllm_plugin import _push
    clock = [0.0]
    posts = Posts()
    p = _push.Pusher(plugin="x", plugin_version="0", engine="vllm", engine_version="0.30.0",
                     url="http://proxy/v1/collect/v", token="tok", clock=lambda: clock[0],
                     post=posts, start=False, env={})
    assert p.tick() is True                     # the first round is a heartbeat
    assert posts.bodies[-1]["batch"]["requests"] == [] and posts.bodies[-1]["token"] == "tok"
    clock[0] = 1.0
    assert p.tick() is False                    # nothing new, the heartbeat is 10 s away
    for i in range(3):
        p.add_request({"id": f"r{i}"})
    p.add_step({"t": 1})
    assert p.tick() is True
    b = posts.bodies[-1]["batch"]
    assert [r["seq"] for r in b["requests"]] == [1, 2, 3] and len(b["steps"]) == 1
    assert b["instance"] == p.instance and b["engine_version"] == "0.30.0"
    p.add_request({"id": "r3"})
    posts.status = 503
    clock[0] = 2.0
    assert p.tick() is False and p._backoff == 1.0
    clock[0] = 2.5
    assert p.tick() is False and len(posts.bodies) == 3          # still waiting
    clock[0] = 3.1
    posts.status = 200
    assert p.tick() is True
    assert [r["id"] for r in posts.bodies[-1]["batch"]["requests"]] == ["r3"]
    posts.status = 401
    p.add_request({"id": "r4"})
    clock[0] = 4.0
    assert p.tick() is False and p.stats["refused"] == 1
    clock[0] = 30.0
    assert p.tick() is False                    # a refused token waits 60 s


def test_ring_keeps_the_newest_and_counts_the_dropped():
    from tandem_vllm_plugin import _push
    r = _push.Ring(5)
    for i in range(8):
        r.add({"i": i})
    assert r.dropped == 3 and [x["seq"] for x in r.after(0)] == [4, 5, 6, 7, 8]
    assert [x["seq"] for x in r.after(6)] == [7, 8] and r.after(8) == []
    assert [x["seq"] for x in r.after(4, limit=2)] == [5, 6]


def test_the_two_packages_ship_the_same_poster_and_the_license():
    a = os.path.join(ROOT, "plugins", "vllm", "tandem_vllm_plugin", "_push.py")
    b = os.path.join(ROOT, "plugins", "sglang", "tandem_sglang_plugin", "_push.py")
    assert filecmp.cmp(a, b, shallow=False), "plugins/*/_push.py must stay identical"
    for d in ("vllm", "sglang"):
        assert filecmp.cmp(os.path.join(ROOT, "LICENSE"),
                           os.path.join(ROOT, "plugins", d, "LICENSE"), shallow=False)
    import tomllib
    for d, group in (("vllm", "vllm.stat_logger_plugins"), ("sglang", "sglang.srt.plugins")):
        with open(os.path.join(ROOT, "plugins", d, "pyproject.toml"), "rb") as f:
            pp = tomllib.load(f)
        assert pp["project"]["dependencies"] == []
        assert group in pp["project"]["entry-points"]


# ----------------------------------------------------------------------------------- SGLang stand-in

class HookType(enum.Enum):
    BEFORE = "before"
    AFTER = "after"
    AROUND = "around"
    REPLACE = "replace"


class HookRegistry:
    _hooks: dict = {}
    failed: list = []

    @classmethod
    def register(cls, target, hook, hook_type=HookType.AFTER, *, source=None):
        cls._hooks.setdefault(target, []).append((hook_type, hook))

    @classmethod
    def apply_hooks(cls):
        import functools
        import pkgutil
        for target, hooks in cls._hooks.items():
            try:
                obj_path, attr = target.rsplit(".", 1)
                obj = pkgutil.resolve_name(obj_path)
                original = getattr(obj, attr)
                wrapped = original
                for ht, hook in hooks:
                    assert ht == HookType.AFTER

                    def make(orig, hk):
                        @functools.wraps(orig)
                        def wrapper(*args, **kwargs):
                            result = orig(*args, **kwargs)
                            modified = hk(result, *args, **kwargs)
                            return modified if modified is not None else result
                        return wrapper
                    wrapped = make(wrapped, hook)
                setattr(obj, attr, wrapped)
            except Exception:                                      # noqa: BLE001
                cls.failed.append(target)       # SGLang logs it and starts the same


@dataclasses.dataclass
class QueueCount:
    total: int = 0
    by_priority: dict | None = None


@dataclasses.dataclass
class SGSchedulerStats:
    num_running_reqs: QueueCount = dataclasses.field(default_factory=QueueCount)
    num_queue_reqs: QueueCount = dataclasses.field(default_factory=QueueCount)
    gen_throughput: float = 0.0
    cache_hit_rate: float = 0.0
    token_usage: float = 0.0
    kv_available_tokens: int = 0
    kv_used_tokens: int = 0
    spec_accept_length: float = 0.0
    spec_accept_rate: float = 0.0
    num_retracted_reqs: int = 0


@dataclasses.dataclass
class SchedulerReqTimeStats:
    queue_duration_s: float = 0.0
    forward_entry_time: float = 0.0
    prefill_finished_time: float = 0.0
    completion_time: float = 0.0


@dataclasses.dataclass
class APIServerReqTimeStats:
    created_time: float = 0.0
    finished_time: float = 0.0
    first_token_time: float = 0.0


class SchedulerMetricsCollector:
    def __init__(self, labels):
        self.labels = labels
        self.logged = 0

    def log_stats(self, stats):
        self.logged += 1


class TokenizerManager:
    def __init__(self, k=4):
        self.server_args = types.SimpleNamespace(speculative_num_draft_tokens=k)
        self.collected = 0

    def collect_metrics(self, state, recv_obj, i):
        self.collected += 1


def install_sglang(renamed: bool = False) -> None:
    for k in [k for k in sys.modules if k == "sglang" or k.startswith("sglang.")
              or k.startswith("tandem_sglang_plugin")]:
        del sys.modules[k]
    HookRegistry._hooks = {}
    HookRegistry.failed = []
    _mod("sglang", __path__=[])
    _mod("sglang.version", __version__="0.5.21")
    _mod("sglang.srt", __path__=[])
    _mod("sglang.srt.plugins", __path__=[])
    _mod("sglang.srt.plugins.hook_registry", HookRegistry=HookRegistry, HookType=HookType)
    _mod("sglang.srt.managers", __path__=[])
    tm = _mod("sglang.srt.managers.tokenizer_manager")
    tm.TokenizerManager = type("TokenizerManager", (TokenizerManager,), {})
    if renamed:                                      # a release that renamed the method
        tm.TokenizerManager = type("TokenizerManager", (), {
            "collect_request_metrics": TokenizerManager.collect_metrics})
    _mod("sglang.srt.observability", __path__=[])
    _mod("sglang.srt.observability.metrics_collector",
         SchedulerMetricsCollector=type("SchedulerMetricsCollector",
                                        (SchedulerMetricsCollector,), {}))


def batch_out(i_finished: bool = True):
    t = 500.0
    return types.SimpleNamespace(
        rids=["other", "chatcmpl-px-9f"], finished_reasons=[None, {"type": "stop"}],
        prompt_tokens=[10, 900], completion_tokens=[1, 61], cached_tokens=[0, 512],
        reasoning_tokens=[0, 20], cached_tokens_details=[None, {"device": 500, "host": 12}],
        retraction_counts=[0, 2], spec_verify_ct=[0, 20], spec_num_correct_drafts=[0, 40],
        dp_ranks=None,
        time_stats=[None, SchedulerReqTimeStats(queue_duration_s=0.125, forward_entry_time=t,
                                                prefill_finished_time=t + 0.3)])


# ----------------------------------------------------------------------------------- SGLang tests

def test_sglang_registers_two_hooks_and_reads_a_finished_request():
    install_sglang()
    plug = importlib.import_module("tandem_sglang_plugin")
    from tandem_sglang_plugin import _push, hooks
    fresh_push(_push)
    plug.register()
    assert set(HookRegistry._hooks) == {plug.TOKENIZER_TARGET, plug.SCHEDULER_TARGET}
    HookRegistry.apply_hooks()
    assert HookRegistry.failed == []
    from sglang.srt.managers.tokenizer_manager import TokenizerManager as TM
    tm = TM(k=4)
    state = types.SimpleNamespace(finished=False, time_stats=APIServerReqTimeStats(
        created_time=100.0, first_token_time=100.5, finished_time=102.0))
    tm.collect_metrics(state, batch_out(), 1)            # not finished yet: nothing
    assert tm.collected == 1 and len(hooks.pusher().requests) == 0
    state.finished = True
    assert tm.collect_metrics(state, batch_out(), 1) is None
    r = hooks.pusher().requests.after(0)[0]
    assert r["id"] == "chatcmpl-px-9f" and r["finish_reason"] == "stop"
    assert (r["prompt_tokens"], r["cached_tokens"], r["completion_tokens"]) == (900, 512, 61)
    assert r["cached_detail"] == {"device": 500, "host": 12} and r["reasoning_tokens"] == 20
    assert r["preemptions"] == 2 and r["queue_ms"] == 125.0 and r["prefill_ms"] == 300.0
    assert (r["decode_ms"], r["ttft_ms"], r["e2e_ms"]) == (1500.0, 500.0, 2000.0)
    assert (r["spec_verify_ct"], r["spec_accepted"], r["spec_draft_tokens"]) == (20, 40, 60)
    assert hooks.pusher().engine_version == "0.5.21"


def test_sglang_scheduler_hook_samples_once_a_second():
    install_sglang()
    plug = importlib.import_module("tandem_sglang_plugin")
    from tandem_sglang_plugin import _push, hooks
    fresh_push(_push)
    hooks._state.update(off=False, failures=0, last_step={})
    plug.register()
    HookRegistry.apply_hooks()
    from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector as SC
    c = SC({"tp_rank": 0, "dp_rank": None})
    st = SGSchedulerStats(num_running_reqs=QueueCount(3), num_queue_reqs=QueueCount(1),
                          gen_throughput=88.5, cache_hit_rate=0.4, token_usage=0.25,
                          spec_accept_length=2.6, spec_accept_rate=0.55, num_retracted_reqs=1)
    c.log_stats(st)
    c.log_stats(st)                                     # within the second: kept out
    assert c.logged == 2
    s = hooks.pusher().steps.after(0)
    assert len(s) == 1
    assert (s[0]["running"], s[0]["waiting"], s[0]["kv_usage"]) == (3, 1, 0.25)
    assert s[0]["spec"] == {"accept_length": 2.6, "accept_rate": 0.55} and s[0]["retracted"] == 1


def test_sglang_hook_failures_stay_inside_the_plugin():
    install_sglang()
    plug = importlib.import_module("tandem_sglang_plugin")
    from tandem_sglang_plugin import _push, hooks
    fresh_push(_push)
    hooks._state.update(off=False, failures=0, last_step={})
    plug.register()
    HookRegistry.apply_hooks()
    from sglang.srt.managers.tokenizer_manager import TokenizerManager as TM
    tm = TM()

    class Bad:
        @property
        def finished(self):
            raise RuntimeError("renamed")
    for _ in range(hooks.MAX_FAILURES):
        assert tm.collect_metrics(Bad(), batch_out(), 1) is None
    assert hooks._state["off"] is True and tm.collected == hooks.MAX_FAILURES
    hooks._state.update(off=False, failures=0)
    # a renamed target: SGLang's apply_hooks logs it and goes on; the other hook still applies
    install_sglang(renamed=True)
    plug = importlib.import_module("tandem_sglang_plugin")
    plug.register()
    HookRegistry.apply_hooks()
    assert HookRegistry.failed == [plug.TOKENIZER_TARGET]


def test_sglang_register_without_the_hook_registry_is_a_log_line():
    for k in [k for k in sys.modules if k.startswith("sglang") or
              k.startswith("tandem_sglang_plugin")]:
        del sys.modules[k]
    _mod("sglang", __path__=[])
    _mod("sglang.srt", __path__=[])
    plug = importlib.import_module("tandem_sglang_plugin")
    plug.register()                                  # no exception


if __name__ == "__main__":
    run(globals())
