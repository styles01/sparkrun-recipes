"""How the conductor fits the served model (ENG-E10): what the model's adapter says the router may
ask of it, read once at the server's start.

The router's documents are written for Qwen3.8: the served model answers the class and package
questions with thinking off, and a turn carries its class's packages, its effort sentence and the
memory notes. A model whose adapter declares `Capabilities.conductor` gets the router with these
keys changed (`DEFAULTS` is what a model without the field gets, the router as before):

    served_classifier  the served model answers the class and package questions (the `qwen`
                       stage); False: the decider stage alone, asked also where the router
                       document leaves it off; a turn below its accept bar falls back
    fallback           `class`: a turn whose class nobody decided (no answer at the accept bar, or
                       a tool turn of such a session) gets the default class's whole policy;
                       `lean`: the profile's default effort and no effort sentence; the class's
                       packages, core and rules as `packages` and `rules` let them (SRV-534)
    packages           `all`: the class's general packages, its core and the package question's
                       pick; `core`: the class's core and the pins; `none`: the pins alone
    no_package_classes classes that get no package at all (pins excepted)
    effort_notes       the effort level's sentence goes into the note; False: the level sets the
                       thinking switch and the budget alone
    memory             the memory card and index go in under a profile with memory on
    rules              the class's rules and note go in

`QWEN38_CONDUCTOR_FIT`: `model` (the default) the adapter's fit, `off` the router as written,
`k=v,k=v` the adapter's fit with these keys replaced (an A/B without a code change).
"""

from __future__ import annotations

import copy

from engine.settings import SETTINGS as _S

DEFAULTS: dict = {"served_classifier": True, "fallback": "class", "packages": "all",
                  "no_package_classes": (), "effort_notes": True, "memory": True,
            "rules": True}
CHOICES = {"fallback": ("class", "lean"), "packages": ("all", "core", "none")}

#: the fit the router reads (`resolve` at the server's start; DEFAULTS until then)
FIT: dict = dict(DEFAULTS)
SOURCE = "default"


def _value(key: str, raw: str):
    base = DEFAULTS[key]
    if isinstance(base, bool):
        if raw.lower() in ("1", "true", "on", "yes"):
            return True
        if raw.lower() in ("0", "false", "off", "no"):
            return False
        raise ValueError(f"{key}: {raw!r} is not a switch")
    if isinstance(base, tuple):
        return tuple(x for x in raw.replace("+", " ").split() if x)
    if raw not in CHOICES.get(key, (raw,)):
        raise ValueError(f"{key}: {raw!r} is not one of {', '.join(CHOICES[key])}")
    return raw


def resolve(adapter_fit: dict | None, setting: str | None = None) -> tuple[dict, str]:
    """(the fit, where it came from) for an adapter's `Capabilities.conductor` and the setting."""
    setting = (setting if setting is not None else (_S.get("CONDUCTOR_FIT") or "model")).strip()
    fit = dict(DEFAULTS)
    if setting.lower() == "off":
        return fit, "off"
    unknown = sorted(set(adapter_fit or {}) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"the adapter's conductor fit has unknown keys: {', '.join(unknown)}")
    fit.update(copy.deepcopy(adapter_fit or {}))
    source = "model" if adapter_fit else "default"
    if setting.lower() not in ("", "model"):
        for item in setting.split(","):
            k, sep, v = item.partition("=")
            k = k.strip()
            if not sep or k not in DEFAULTS:
                raise ValueError(f"QWEN38_CONDUCTOR_FIT: {item!r} is not key=value of "
                                 f"{', '.join(DEFAULTS)}")
            fit[k] = _value(k, v.strip())
        source = f"{source}+{setting}"
    for k, ok in CHOICES.items():
        if fit[k] not in ok:
            raise ValueError(f"{k}: {fit[k]!r} is not one of {', '.join(ok)}")
    fit["no_package_classes"] = tuple(fit["no_package_classes"])
    return fit, source


def install(adapter_fit: dict | None, setting: str | None = None) -> str:
    """Resolve and make it the router's fit. Returns the log line."""
    global FIT, SOURCE
    FIT, SOURCE = resolve(adapter_fit, setting)
    changed = {k: v for k, v in FIT.items() if v != DEFAULTS[k]}
    return (f"[conductor] model fit ({SOURCE}): "
            + (", ".join(f"{k}={'+'.join(v) if isinstance(v, tuple) else v}"
                         for k, v in changed.items()) or "the router as written"))


def summary() -> dict:
    """The journal's view: the keys that differ from the router as written."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in FIT.items()
            if v != DEFAULTS[k]}
