"""Standalone BF16-origin mul1 EXL3 loading; executable by ops/gate.sh, no pytest dependency."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.loader import Weights  # noqa: E402
from tools import exl3_linear as X  # noqa: E402


class StandaloneEXL3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.saved_env = os.environ.pop("QWEN38_EXL3", None)

    def tearDown(self):
        os.environ.pop("QWEN38_EXL3", None)
        if self.saved_env is not None:
            os.environ["QWEN38_EXL3"] = self.saved_env
        self.tmp.cleanup()

    def checkpoint(self, path=None, prefix="model.language_model.", head_bits=5, nested=False,
                   dtype="bfloat16", dtype_field="dtype", indexed=False, tied=False, plain_head=False):
        path = path or self.root
        path.mkdir(exist_ok=True)
        tensors = {
            prefix + "embed_tokens.weight": torch.full((128, 128), 0.25, dtype=torch.bfloat16),
            prefix + "layers.0.linear_attn.in_proj_a.weight": torch.full((48, 128), 0.25, dtype=torch.float16),
            prefix + "norm.weight": torch.ones(128, dtype=torch.bfloat16),
        }
        for base, bits in ((prefix + "layers.0.mlp.down_proj", 3), ("lm_head", head_bits)):
            if base == "lm_head" and (tied or plain_head):
                continue
            block = X.random_block(128, 128, bits, device="cpu", seed=1)
            tensors.update({base + ".trellis": block.t, base + ".suh": block.suh,
                            base + ".svh": block.svh,
                            base + ".mul1": torch.tensor(X.MUL1 - (1 << 32), dtype=torch.int32)})
        if plain_head:
            tensors["lm_head.weight"] = torch.full((128, 128), 0.25, dtype=torch.float16)
        q = {"quant_method": "exl3"}
        text: dict[str, object] = {"tie_word_embeddings": tied}
        if dtype is not None:
            text[dtype_field] = dtype
        config = ({"text_config": {**text, "quantization_config": q}} if nested
                  else {**text, "quantization_config": q})
        (path / "config.json").write_text(json.dumps(config))
        if indexed:
            first, second = {}, {}
            for i, (key, tensor) in enumerate(tensors.items()):
                (first if i % 2 == 0 else second)[key] = tensor
            files = {"model-00001-of-00002.safetensors": first,
                     "model-00002-of-00002.safetensors": second}
            for name, data in files.items():
                save_file(data, str(path / name))
            (path / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {key: name for name, data in files.items() for key in data}}))
        else:
            save_file(tensors, str(path / "model.safetensors"))
        return tensors

    def weights(self, path=None, **kwargs):
        return Weights(str(path or self.root), device="cpu", skip_mtp=True, **kwargs)

    def test_gate_and_quantized_projection_preserve_stored_values(self):
        original = self.checkpoint()
        w = self.weights(exl3=str(self.root))
        gate = w.t["layers.0.linear_attn.in_proj_a.weight"]
        self.assertEqual(gate.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(gate.float(), original["model.language_model.layers.0.linear_attn.in_proj_a.weight"].float()))
        result = torch.nn.functional.linear(torch.ones(1, 128, dtype=torch.bfloat16), gate)
        self.assertTrue(torch.equal(result, torch.full((1, 48), 32.0, dtype=torch.bfloat16)))
        block = w.q["layers.0.mlp.down_proj"]
        self.assertEqual(block.kb, 3)
        self.assertTrue(torch.equal(block.t, original["model.language_model.layers.0.mlp.down_proj.trellis"]))
        self.assertEqual(w.t["lm_head.weight"].kb, 5)

    def test_config_detection(self):
        self.checkpoint()
        self.assertEqual(self.weights().exl3_source, str(self.root))

    def test_config_wrapper_and_supported_rate_matrix(self):
        for i, (prefix, nested, bits, field) in enumerate(itertools.product(
                ("model.language_model.", "model.", ""), (False, True), X.KS, ("dtype", "torch_dtype"))):
            with self.subTest(prefix=prefix, nested=nested, bits=bits, field=field):
                p = self.root / str(i)
                self.checkpoint(p, prefix=prefix, nested=nested, head_bits=bits, dtype_field=field)
                w = self.weights(p)
                self.assertEqual(w.t["lm_head.weight"].kb, bits)
                self.assertEqual(w.q["layers.0.mlp.down_proj"].kb, 3)
                self.assertEqual(w.t["layers.0.linear_attn.in_proj_a.weight"].dtype, torch.bfloat16)

    def test_lossy_plain_conversion_refused(self):
        data = self.checkpoint()
        data["model.language_model.layers.0.linear_attn.in_proj_a.weight"].fill_(0.1)
        save_file(data, str(self.root / "model.safetensors"))
        with self.assertRaisesRegex(ValueError, "in_proj_a.*losslessly"):
            self.weights()

    def test_nonfinite_plain_values_refused(self):
        for i, value in enumerate((float("inf"), -float("inf"), float("nan"))):
            with self.subTest(value=value):
                p = self.root / str(i)
                data = self.checkpoint(p)
                data["model.language_model.layers.0.linear_attn.in_proj_a.weight"][0, 0] = value
                save_file(data, str(p / "model.safetensors"))
                with self.assertRaisesRegex(ValueError, "in_proj_a.*non-finite"):
                    self.weights(p)

    def test_declared_fp16_or_missing_source_dtype_refused(self):
        for i, dtype in enumerate(("float16", None, "float32")):
            with self.subTest(dtype=dtype):
                p = self.root / str(i)
                self.checkpoint(p, dtype=dtype)
                with self.assertRaisesRegex(ValueError, "declared BF16"):
                    self.weights(p)

    def test_indexed_multishard_checkpoint(self):
        self.checkpoint(indexed=True)
        w = self.weights()
        self.assertEqual(w.t["lm_head.weight"].kb, 5)
        self.assertEqual(w.q["layers.0.mlp.down_proj"].kb, 3)

    def test_plain_fp16_head(self):
        original = self.checkpoint(plain_head=True)
        head = self.weights().t["lm_head.weight"]
        self.assertEqual(head.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(head.float(), original["lm_head.weight"].float()))

    def test_tied_embeddings(self):
        self.checkpoint(tied=True)
        w = self.weights()
        self.assertIs(w.t["lm_head.weight"], w.t["embed_tokens.weight"])

    def test_symlink_equivalence(self):
        pack = self.root / "pack"
        self.checkpoint(pack)
        alias = self.root / "alias"
        alias.symlink_to(pack, target_is_directory=True)
        w = self.weights(alias, exl3=str(pack))
        self.assertEqual(w.t["layers.0.linear_attn.in_proj_a.weight"].dtype, torch.bfloat16)

    def test_overlay_keeps_separate_base_dtypes(self):
        base, overlay = self.root / "base", self.root / "overlay"
        self.checkpoint(base)
        self.checkpoint(overlay, head_bits=3)
        w = self.weights(base, exl3=str(overlay))
        self.assertEqual(w.exl3_source, str(overlay))
        self.assertEqual(w.t["lm_head.weight"].kb, 3)
        self.assertEqual(w.t["layers.0.linear_attn.in_proj_a.weight"].dtype, torch.float16)

    def test_environment_override_and_argument_precedence(self):
        base, environment, explicit = self.root / "base", self.root / "env", self.root / "arg"
        self.checkpoint(base)
        self.checkpoint(environment, head_bits=2)
        self.checkpoint(explicit, head_bits=6)
        os.environ["QWEN38_EXL3"] = str(environment)
        self.assertEqual(self.weights(base).t["lm_head.weight"].kb, 2)
        self.assertEqual(self.weights(base, exl3=str(explicit)).t["lm_head.weight"].kb, 6)

    def test_empty_argument_and_environment_disable_detection(self):
        data = self.checkpoint()
        data["lm_head.weight"] = torch.ones(128, 128, dtype=torch.bfloat16)
        save_file(data, str(self.root / "model.safetensors"))
        self.assertIsNone(self.weights(exl3="").exl3_source)
        os.environ["QWEN38_EXL3"] = ""
        self.assertIsNone(self.weights().exl3_source)

    def test_non_exl3_checkpoint_preserves_dtype(self):
        data = self.checkpoint()
        data = {k: v for k, v in data.items() if not k.endswith((".trellis", ".suh", ".svh", ".mul1"))}
        data["lm_head.weight"] = torch.ones(128, 128, dtype=torch.bfloat16)
        save_file(data, str(self.root / "model.safetensors"))
        (self.root / "config.json").write_text(json.dumps({"tie_word_embeddings": False}))
        w = self.weights()
        self.assertIsNone(w.exl3_source)
        self.assertEqual(w.t["layers.0.linear_attn.in_proj_a.weight"].dtype, torch.float16)

    def test_unsupported_codebook_still_rejected(self):
        data = self.checkpoint()
        data["lm_head.mcg"] = torch.zeros((), dtype=torch.int32)
        save_file(data, str(self.root / "model.safetensors"))
        with self.assertRaisesRegex(ValueError, "mcg codebook is not supported"):
            self.weights()

    def test_plain_fp32_norm_is_not_downcast(self):
        data = self.checkpoint()
        data["model.language_model.norm.weight"] = torch.full((128,), 0.1, dtype=torch.float32)
        save_file(data, str(self.root / "model.safetensors"))
        self.assertEqual(self.weights().t["norm.weight"].dtype, torch.float32)

    def test_conflicting_source_dtype_fields_refused(self):
        self.checkpoint()
        p = self.root / "config.json"
        config = json.loads(p.read_text())
        config["torch_dtype"] = "float16"
        p.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "declared BF16"):
            self.weights()

    def test_dtype_metadata_falls_back_to_wrapper(self):
        self.checkpoint(nested=True)
        p = self.root / "config.json"
        config = json.loads(p.read_text())
        config["text_config"].pop("dtype")
        config["torch_dtype"] = "torch.bfloat16"
        p.write_text(json.dumps(config))
        self.assertEqual(self.weights().t["layers.0.linear_attn.in_proj_a.weight"].dtype, torch.bfloat16)

    def test_duplicate_shard_key_refused(self):
        data = self.checkpoint()
        save_file({"lm_head.mul1": data["lm_head.mul1"]}, str(self.root / "duplicate.safetensors"))
        with self.assertRaisesRegex(ValueError, "duplicate tensor"):
            self.weights()

    def test_roundtrip_comparison_is_chunked(self):
        from engine.loader import _stored_fp16_to_bf16
        original = torch.full((2 * (1 << 20) + 7,), 0.25, dtype=torch.float16)
        sizes = []
        equal = torch.equal
        def record(a, b):
            sizes.append(a.numel())
            return equal(a, b)
        with patch.object(torch, "equal", side_effect=record):
            result = _stored_fp16_to_bf16(original, "test.weight")
        self.assertGreater(len(sizes), 1)
        self.assertLessEqual(max(sizes), 1 << 20)
        self.assertTrue(torch.equal(result.float(), original.float()))


if __name__ == "__main__":
    unittest.main(buffer=True)
