"""pytest's setup for tests/, loaded before any test module is collected.

chatd/sandbox.py reads its limit (QSE_SANDBOX_MEM_GB) and its socket (QSE_SANDBOX_SOCKET) once,
when it is imported. tests/chatd_util.py pins both for the tests, but a test module that imports
chatd before it (tests/test_sandbox_box.py did; pytest collects in the order it is given) fixed
the defaults for the whole session: the 4 GiB limit failed six tests in
tests/test_chatd_agent.py, and on a box with Docker a Chat could find the real sandbox. The same
pins here hold whatever is collected first; chatd_util keeps them for a run without pytest.
"""

import os
import tempfile

if os.environ.get("QSE_SANDBOX_BOX") != "1":
    os.environ.setdefault("QSE_SANDBOX_SOCKET",
                          os.path.join(tempfile.gettempdir(), "qse-tests-no-sandbox.sock"))
os.environ.setdefault("QSE_SANDBOX_MEM_GB", "1")
