"""The setup guide's server half (docs/setup.md): the model registry, the memory estimate, the
assistant Wren, the apply job and the reset levels behind `/v1/setup/*` (server/setup_api.py).

Nothing here is on the serving path: a request to the OpenAI API never reaches this package, and
the engine imports it only when a `/v1/setup/*` route is called or the server starts in setup mode.
"""
