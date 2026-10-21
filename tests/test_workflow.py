"""CPU-only checks: no ML imports, network requests, or model weights."""
import ast
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from zipfile import ZipFile

from experiments.package import build
from experiments.runner import ROOT, BASELINES, validate_answer, run


class ContractTests(unittest.TestCase):
    def test_baseline_answer_contract_and_formatting(self):
        for strategy, folder in BASELINES.items():
            with self.subTest(strategy=strategy):
                tree = ast.parse((ROOT / folder / "main.py").read_text(encoding="utf-8"))
                node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "answer")
                self.assertEqual([a.arg for a in node.args.args], ["question", "corpus_dir", "model_id"])
                self.assertEqual(ast.unparse(node.returns), "str")
                inputs = {"input_ids": SimpleNamespace(shape=(1, 2))}
                batch = Mock()
                batch.to.return_value = inputs
                tokenizer = Mock(pad_token_id=0)
                tokenizer.apply_chat_template.return_value = batch
                tokenizer.decode.return_value = "  scientific answer\n"
                model = Mock(device="fake-device")
                model.generate.return_value = [[10, 11, 12, 13]]
                loaded = (tokenizer, model) if strategy == "zero_shot" else (tokenizer, model, [], [], None)
                namespace = {"_load": Mock(return_value=loaded), "_build_context": Mock(return_value="context: "),
                             "SYSTEM_PROMPT": "system", "MAX_NEW_TOKENS": 2048,
                             "torch": SimpleNamespace(no_grad=nullcontext)}
                # Execute the actual answer function with fake model/tokenizer boundaries.
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(folder), "exec"), namespace)
                self.assertEqual(namespace["answer"]("question", "/corpus", "model"), "scientific answer")
                tokenizer.decode.assert_called_once_with([12, 13], skip_special_tokens=True)
                kwargs = tokenizer.apply_chat_template.call_args.kwargs
                self.assertTrue(kwargs["add_generation_prompt"])
                messages = tokenizer.apply_chat_template.call_args.args[0]
                self.assertEqual(messages[-1]["content"], ("context: " if strategy == "rag" else "") + "question")
                self.assertFalse(model.generate.call_args.kwargs["do_sample"])

    def test_answer_validation(self):
        self.assertEqual(validate_answer("Answer."), "Answer.")
        for value in (None, {}, "", " ", " padded "):
            with self.assertRaises(ValueError):
                validate_answer(value)

    def test_result_recording_without_models(self):
        from unittest.mock import patch
        cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: None,
                               reset_peak_memory_stats=lambda: None,
                               get_device_name=lambda _: "fake GPU",
                               get_device_properties=lambda _: SimpleNamespace(total_memory=100),
                               max_memory_allocated=lambda: 10, max_memory_reserved=lambda: 20)
        module = SimpleNamespace(_cache={}, MAX_NEW_TOKENS=128, SYSTEM_PROMPT="test",
                                 answer=Mock(return_value="Answer."))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}), patch(
                "experiments.runner.subprocess.check_output", return_value="test\n"
            ):
                result = run(module, model_id="fake", strategy="zero_shot",
                             corpus_dir=tmp, limit=2, output=path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), result)
            self.assertEqual([a["cold_call"] for a in result["answers"]], [True, False])
            schema = json.loads((ROOT / "experiments/results.schema.json").read_text())
            self.assertTrue(set(schema["required"]).issubset(result))
            required = schema["properties"]["answers"]["items"]["required"]
            self.assertTrue(all(set(required).issubset(a) for a in result["answers"]))
            # Only text/path/model arguments cross the submission interface.
            self.assertEqual(len(module.answer.call_args.args), 3)

    def test_zip_root_and_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = build("zero_shot", Path(tmp) / "submission.zip")
            with ZipFile(path) as archive:
                self.assertEqual(archive.namelist(), ["main.py"])
                self.assertEqual(archive.read("main.py"), (ROOT / "01_zero_shot/main.py").read_bytes())
                compile(archive.read("main.py"), "main.py", "exec")

    def test_rag_packaging_requires_assets(self):
        # Synthetic tree makes this independent of any assets prepared later.
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            with patch("experiments.package.ROOT", Path(tmp)):
                with self.assertRaises(ValueError):
                    build("rag", Path(tmp) / "submission.zip")
                source = Path(tmp) / "02_rag"
                source.mkdir()
                (source / "main.py").write_text("# fixture")
                for name in ("docling_models", "embed_model"):
                    (source / name).mkdir()
                    (source / name / "asset").write_text("fixture")
                with ZipFile(build("rag", Path(tmp) / "submission.zip")) as archive:
                    self.assertEqual(set(archive.namelist()), {"main.py", "docling_models/asset", "embed_model/asset"})

    def test_notebook_safe_defaults(self):
        notebook = json.loads((ROOT / "notebooks/baseline_colab.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), "notebook", "exec")
        source = "\n".join("".join(c["source"]) for c in notebook["cells"])
        self.assertIn("ALLOW_MODEL_DOWNLOADS = False", source)
        self.assertIn("RUN_INFERENCE = False", source)


if __name__ == "__main__":
    unittest.main()
