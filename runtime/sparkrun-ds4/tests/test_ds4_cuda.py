"""Tests for the Ds4CudaRuntime plugin.

These tests mock the sparkrun base classes so they run without a full
sparkrun installation.  They verify the flag mapping, env defaults, and
command generation logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ─── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def runtime():
    """Create a Ds4CudaRuntime instance."""
    from sparkrun_ds4.runtime import Ds4CudaRuntime
    return Ds4CudaRuntime()


@pytest.fixture
def mock_recipe():
    """A mock recipe with minimal config."""
    recipe = MagicMock()
    recipe.model = "~/gguf/test-model.gguf"
    recipe.command = None  # No explicit command template
    recipe.defaults = {
        "context": 131072,
        "port": 8000,
        "host": "0.0.0.0",
    }
    recipe.env = {
        "DS4_CONT_DSPARK": "1",
        "DS4_DSPARK_MODEL": "~/gguf/drafter.gguf",
    }
    recipe.executor = "local"
    recipe.max_nodes = 1
    recipe.container = ""

    # build_config_chain returns a dict-like object
    config = MagicMock()
    config_values = {"model": recipe.model, **recipe.defaults}

    def _get(key, default=None):
        return config_values.get(key, default)

    config.get = _get
    config.set = lambda k, v: config_values.update({k: v})
    recipe.build_config_chain.return_value = config
    recipe.render_command.return_value = None
    return recipe


# ─── Tests ────────────────────────────────────────────────────────

class TestDs4CudaRuntimeBasics:
    def test_runtime_name(self, runtime):
        assert runtime.runtime_name == "ds4-cuda"

    def test_cluster_strategy_is_native(self, runtime):
        assert runtime.cluster_strategy() == "native"

    def test_default_executor_is_local(self, runtime):
        assert runtime.default_executor() == "local"

    def test_get_family(self, runtime):
        assert runtime.get_family() == "ds4-cuda"


class TestCommandGeneration:
    def test_generates_ds4_serve_command(self, runtime, mock_recipe):
        """The generated command should start with ds4-serve and include -m and -c."""
        cmd = runtime.generate_command(mock_recipe, {}, is_cluster=False)
        assert cmd.startswith("ds4-serve")
        assert "-m" in cmd
        assert "~/gguf/test-model.gguf" in cmd
        assert "-c" in cmd
        assert "131072" in cmd
        assert "--port" in cmd
        assert "8000" in cmd
        assert "--host" in cmd
        assert "0.0.0.0" in cmd

    def test_uses_explicit_command_when_present(self, runtime, mock_recipe):
        """When recipe.command is set, it should be rendered and used."""
        mock_recipe.command = "ds4-serve -m {model} --port {port}"
        mock_recipe.render_command.return_value = "ds4-serve -m ~/gguf/test-model.gguf --port 8000"
        cmd = runtime.generate_command(mock_recipe, {}, is_cluster=False)
        assert cmd == "ds4-serve -m ~/gguf/test-model.gguf --port 8000"

    def test_max_model_len_translates_to_context(self, runtime, mock_recipe):
        """max_model_len (vLLM key) should translate to context (ds4 key)."""
        mock_recipe.defaults = {"max_model_len": 65536, "port": 8000, "host": "0.0.0.0"}
        config = MagicMock()
        config_values = {"model": mock_recipe.model, **mock_recipe.defaults}

        def _get(key, default=None):
            return config_values.get(key, default)

        config.get = _get
        config.set = lambda k, v: config_values.update({k: v})
        mock_recipe.build_config_chain.return_value = config
        mock_recipe.render_command.return_value = None

        cmd = runtime.generate_command(mock_recipe, {}, is_cluster=False)
        # context should be set from max_model_len
        assert config_values.get("context") == 65536

    def test_served_model_name_added(self, runtime, mock_recipe):
        """served_model_name should be added as --served-model-name."""
        mock_recipe.defaults = {
            "context": 131072,
            "port": 8000,
            "host": "0.0.0.0",
            "served_model_name": "deepseek-v4-flash",
        }
        config = MagicMock()
        config_values = {"model": mock_recipe.model, **mock_recipe.defaults}

        def _get(key, default=None):
            return config_values.get(key, default)

        config.get = _get
        config.set = lambda k, v: config_values.update({k: v})
        mock_recipe.build_config_chain.return_value = config
        mock_recipe.render_command.return_value = None

        cmd = runtime.generate_command(mock_recipe, {}, is_cluster=False)
        assert "--served-model-name" in cmd
        assert "deepseek-v4-flash" in cmd


class TestEnvDefaults:
    def test_extra_env_has_defaults(self, runtime):
        env = runtime.get_extra_env()
        assert env["DS4_BATCH_FIT_HEADROOM_MB"] == "8192"
        assert env["DS4_SERVER_SERIAL_MAX_TOKENS"] == "131072"
        assert env["DS4_SERVER_COALESCE_MAX"] == "2"

    def test_common_env_hf_offline(self, runtime):
        env = runtime.get_common_env()
        assert env.get("HF_HUB_OFFLINE") == "1"


class TestValidation:
    def test_warns_on_docker_executor(self, runtime, mock_recipe):
        """Should warn when recipe sets executor=docker."""
        mock_recipe.executor = "docker"
        issues = runtime.validate_recipe(mock_recipe)
        assert any("local" in i for i in issues)

    def test_warns_on_multi_node(self, runtime, mock_recipe):
        """Should warn when max_nodes > 1."""
        mock_recipe.max_nodes = 2
        issues = runtime.validate_recipe(mock_recipe)
        assert any("multi-node" in i for i in issues)

    def test_no_warning_for_local_single_node(self, runtime, mock_recipe):
        mock_recipe.executor = "local"
        mock_recipe.max_nodes = 1
        issues = runtime.validate_recipe(mock_recipe)
        assert issues == []