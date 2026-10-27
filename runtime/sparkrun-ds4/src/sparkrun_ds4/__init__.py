"""sparkrun-ds4 — sparkrun runtime plugin for the ds4 CUDA engine.

This package provides the :class:`Ds4CudaRuntime` plugin, registered under
the ``sparkrun.runtimes`` entry point group as ``ds4-cuda``.

The ds4 CUDA engine is Bleysg's fork of antirez/ds4 — a native C/CUDA
inference server (NOT a Docker container) that serves GGUF models with an
OpenAI-compatible HTTP API.  Because it runs as a native process, the
recipe should set ``executor: local`` so sparkrun uses the LocalExecutor
(no-container) path instead of DockerExecutor.
"""

from sparkrun_ds4.runtime import Ds4CudaRuntime

__all__ = ["Ds4CudaRuntime"]
__version__ = "0.1.0"