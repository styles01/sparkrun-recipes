"""A module that imports chatd before tests/chatd_util.py and sets nothing: collected first by
tests/test_chatd_import_order.py, it stands for any test module that does so."""

from chatd import sandbox  # noqa: F401
