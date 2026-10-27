"""A single model card answers only to a key, as the list route has always required.

Before the fix the /v1/models branch checked authorize() and /v1/models/{id} did not:
a deployment that handed out keys could still be read card-by-card by anyone who
knew a model name — the 404-vs-401 difference even made enumeration free.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxy_util import Proxy  # noqa: E402

KEY = "tdm-0123456789abcdef"


def test_card_needs_the_key_the_list_needs():
    with Proxy(env={"TANDEM_API_KEYS": f"ci:{KEY}"}) as px:
        px.svc.cfg.apply_env()
        assert px.svc.cfg.users_active()
        # the harness runs on loopback and trust_loopback is on by default; the outside
        # client this fix is about is one that loopback trust does not cover
        px.svc.auth.trusted_local = lambda *a, **k: False

        code, _, _ = px.get("/v1/models")
        assert code == 401, "the list was already gated: unchanged behavior"

        code, _, _ = px.get("/v1/models/anything")
        assert code == 401, "the card must not answer below the gate"

        code, _, _ = px.get("/v1/models/anything",
                           headers={"Authorization": f"Bearer {KEY}"})
        assert code == 404, "with the key the card's own 'no model' answer is back"


def test_open_deployment_unchanged():
    with Proxy() as px:
        code, _, _ = px.get("/v1/models/anything")
        assert code == 404, "no users configured: nobody is refused"


if __name__ == "__main__":
    test_card_needs_the_key_the_list_needs()
    test_open_deployment_unchanged()
    print("2 passed")
