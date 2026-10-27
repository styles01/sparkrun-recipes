"""Drafters. A drafter is one method: given the tokens so far, propose the next few.

The verify step corrects whatever a drafter gets wrong, so a drafter is free to be approximate; it
is not free to be expensive. On this board a proposal that reads a gigabyte of weights has spent
4 ms of a ~150 ms budget before the verify step starts, and it has to earn that back in accepted
tokens. The cost of each drafter, in bytes read per proposed block, is the number to compare them
by.
"""

from __future__ import annotations


class Drafter:
    name = "none"
    last_q = None                  # per-token q rows when this drafter samples

    def requires(self) -> dict:
        """What this drafter needs from the target, checked at load (`check_target`).

        Keys, all optional: `hidden_size` (its taps read hidden states of this width),
        `tap_layers` (target layers it reads), `tensors` (target tensors it reads by name, e.g. the
        head or the embedding), `vocab_size` (the id space of its proposals)."""
        return {}

    def propose(self, context: list[int], k: int) -> list[int]:
        """Up to `k` tokens continuing `context`. May return fewer, including none."""
        raise NotImplementedError

    def observe(self, tokens: list[int]) -> None:
        """Tokens that were actually accepted, in order. Cheap: this runs every step."""

    def reset(self) -> None:
        pass

    def set_sampling(self, sampler) -> None:
        """Hand the request's sampling profile to a drafter that can use one.

        Drafters that can sample their own proposals (the block drafter's head) override this and
        carry `last_q`; the default is a no-op, so the loop can hand it to any arm..
        """


def check_target(drafter, eng) -> None:
    """Refuse a drafter whose declared needs the target does not meet, naming the first one."""
    req = drafter.requires() if hasattr(drafter, "requires") else {}
    cfg = eng.cfg
    name = getattr(drafter, "name", type(drafter).__name__)
    if "hidden_size" in req and int(req["hidden_size"]) != int(cfg.hidden_size):
        raise ValueError(f"drafter {name}: hidden_size {req['hidden_size']} != target {cfg.hidden_size}")
    for lid in req.get("tap_layers", ()):
        if not 0 <= int(lid) < cfg.num_hidden_layers:
            raise ValueError(f"drafter {name}: tap layer {lid} outside the target's "
                             f"{cfg.num_hidden_layers} layers")
    have = getattr(getattr(eng, "w", None), "t", None)
    for t in (req.get("tensors", ()) if have is not None else ()):   # a stub engine has no tensor map
        if t not in have:
            raise ValueError(f"drafter {name}: needs the target tensor {t!r}, which is not loaded")
    if "vocab_size" in req and int(req["vocab_size"]) > int(cfg.vocab_size):
        raise ValueError(f"drafter {name}: vocab {req['vocab_size']} larger than the target's {cfg.vocab_size}")


def run_steps(steps):
    """Drive a `*_steps` generator to its end and return what it returns.

    a proposal that launches a draft on the device is written as a generator that stops
    once, right after the launch, so the serving loop can stream the last block's tokens while the
    draft runs and only then wait for it. Everything else calls the plain method, which is this
    over the same generator, so the two orders run the same code and decide the same things.
    """
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def tree_steps(drafter, context: list[int], k: int):
    """`drafter.propose_tree_steps`, or its plain `propose_tree` for a drafter without one (a
    lookup drafter launches nothing, so it has nothing to stop for)."""
    steps = getattr(drafter, "propose_tree_steps", None)
    if steps is not None:
        return (yield from steps(context, k))
    return drafter.propose_tree(context, k)
