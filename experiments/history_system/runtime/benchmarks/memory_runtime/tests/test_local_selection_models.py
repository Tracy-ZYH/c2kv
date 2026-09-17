"""CPU-only tests for lazy local recovery selection models."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from memory_runtime.recovery import local_selection_models as local


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    @property
    def device(self):
        return "cpu"

    @property
    def dtype(self):
        return self.value.dtype

    def to(self, _device):
        return self

    def float(self):
        return FakeTensor(self.value.astype(np.float32))

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value.tolist()

    def item(self):
        return self.value.item()

    def sum(self, dim=None):
        return FakeTensor(self.value.sum(axis=dim))

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        return iter(self.value)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = tuple(
                part.value.astype(int) if isinstance(part, FakeTensor) else part
                for part in key
            )
        elif isinstance(key, FakeTensor):
            key = key.value.astype(int)
        return FakeTensor(self.value[key])

    def __setitem__(self, key, value):
        self.value[key] = value.value if isinstance(value, FakeTensor) else value

    def __add__(self, value):
        return FakeTensor(self.value + value)

    def __sub__(self, value):
        return FakeTensor(self.value - value)

    def __eq__(self, value):
        return FakeTensor(self.value == value)


class InferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class FakeTorch:
    long = np.int64
    float32 = np.float32
    float16 = np.float16
    bfloat16 = np.float32

    def __init__(self):
        self.nn = SimpleNamespace(
            functional=SimpleNamespace(normalize=self._normalize)
        )

    @staticmethod
    def tensor(value, dtype=None):
        return FakeTensor(np.asarray(value, dtype=dtype))

    @staticmethod
    def zeros(shape, dtype=None):
        return FakeTensor(np.zeros(shape, dtype=dtype))

    @staticmethod
    def stack(values, dim=0):
        return FakeTensor(np.stack([value.value for value in values], axis=dim))

    @staticmethod
    def cat(values, dim=0):
        return FakeTensor(np.concatenate([value.value for value in values], axis=dim))

    @staticmethod
    def arange(stop, device=None):
        return FakeTensor(np.arange(stop, dtype=np.int64))

    @staticmethod
    def softmax(value, dim):
        shifted = value.value - np.max(value.value, axis=dim, keepdims=True)
        exponent = np.exp(shifted)
        return FakeTensor(exponent / exponent.sum(axis=dim, keepdims=True))

    @staticmethod
    def inference_mode():
        return InferenceMode()

    @staticmethod
    def device(value):
        return value

    @staticmethod
    def _normalize(value, p, dim):
        assert p == 2
        norm = np.linalg.norm(value.value, axis=dim, keepdims=True)
        return FakeTensor(value.value / norm)


torch = FakeTorch()


class Batch(dict):
    def to(self, device):
        for key, value in self.items():
            self[key] = value.to(device)
        return self


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = "fixture-qwen3-instruct-template"

    def __init__(self):
        self.calls = []
        self.pad_calls = []
        self.template_calls = []

    def _ids(self, text):
        if text.isdigit():
            return [20 + int(text)]
        return [10 + (ord(character) % 40) for character in text] or [3]

    def __call__(self, texts, **kwargs):
        values = [texts] if isinstance(texts, str) else list(texts)
        self.calls.append((values, dict(kwargs)))
        return {"input_ids": [self._ids(text) for text in values]}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self._ids(text)

    def pad(self, encoded, **kwargs):
        rows = [list(row) for row in encoded["input_ids"]]
        self.pad_calls.append((rows, dict(kwargs)))
        width = max(len(row) for row in rows)
        padded = [[0] * (width - len(row)) + row for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return Batch(
            input_ids=torch.tensor(padded, dtype=torch.long),
            attention_mask=torch.tensor(masks, dtype=torch.long),
        )

    def convert_tokens_to_ids(self, token):
        return {"no": 7, "yes": 8}.get(token)

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, dict(kwargs)))
        return "\n".join(message["content"] for message in messages)

    def decode(self, ids, **kwargs):
        assert kwargs["skip_special_tokens"] is True
        return "".join(str(token - 20) for token in ids)


class FakeBaseModel:
    def __init__(self, kind, choice=3):
        self.kind = kind
        self.choice = choice
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(
            _commit_hash=f"resolved-{kind}", max_position_embeddings=32768
        )
        self.forward_calls = 0
        self.forward_kwargs = []
        self.generate_calls = []

    def eval(self):
        return self

    def to(self, device):
        self.device = torch.device(device)
        return self

    def __call__(self, input_ids, attention_mask, **kwargs):
        self.forward_calls += 1
        self.forward_kwargs.append(dict(kwargs))
        if self.kind == "embedding":
            values = input_ids.float()
            hidden = torch.stack([values, values + 1.0, values + 2.0], dim=-1)
            return SimpleNamespace(last_hidden_state=hidden)
        batch, length = input_ids.shape
        if kwargs.get("logits_to_keep"):
            length = kwargs["logits_to_keep"]
        logits = torch.zeros((batch, length, 64), dtype=torch.float32)
        for index in range(batch):
            logits[index, -1, 7] = 0.0
            logits[index, -1, 8] = 2.0 if index == 0 else -2.0
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, attention_mask, **kwargs):
        self.generate_calls.append(dict(kwargs))
        callback = kwargs["prefix_allowed_tokens_fn"]
        token = 20 + self.choice
        assert token in callback(0, input_ids[0])
        prefix = torch.cat([input_ids[0], torch.tensor([token])])
        assert kwargs["eos_token_id"] in callback(0, prefix)
        tail = torch.tensor(
            [[token, kwargs["eos_token_id"]]], dtype=input_ids.dtype
        )
        return torch.cat([input_ids, tail], dim=1)


class FakeFactory:
    def __init__(self, role, registry):
        self.role = role
        self.registry = registry

    def from_pretrained(self, name, **kwargs):
        self.registry.append((self.role, name, dict(kwargs)))
        if self.role == "tokenizer":
            tokenizer = FakeTokenizer()
            self.registry.append(("tokenizer_instance", name, tokenizer))
            return tokenizer
        kind = "embedding" if self.role == "embedding" else (
            "reranker" if "Reranker" in name else "selector"
        )
        model = FakeBaseModel(kind)
        self.registry.append(("model_instance", name, model))
        return model


class FakeConfigFactory:
    def __init__(self, registry, values=None):
        self.registry = registry
        self.values = values or {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "max_position_embeddings": 262144,
            "use_sliding_window": False,
        }

    def from_pretrained(self, name, **kwargs):
        self.registry.append(("config", name, dict(kwargs)))
        return SimpleNamespace(**self.values)


def fake_runtime(registry):
    return SimpleNamespace(
        __version__="5.8.0",
        AutoConfig=FakeConfigFactory(registry),
        AutoTokenizer=FakeFactory("tokenizer", registry),
        AutoModel=FakeFactory("embedding", registry),
        AutoModelForCausalLM=FakeFactory("causal", registry),
    )


def install_runtime(monkeypatch):
    registry = []
    transformers = fake_runtime(registry)
    monkeypatch.setattr(local, "_import_runtime", lambda _device: (torch, transformers))
    return registry


def instances(registry, kind):
    return [value for role, _name, value in registry if role == kind]


def selector_payload(candidate_count=2):
    candidates = [
        {
            "unit_id": f"u{index}",
            "event_id": f"event-{index}",
            "text": f"exact archived text {index}",
            "provenance": {"source": index},
        }
        for index in range(1, candidate_count + 1)
    ]
    actions = [()]
    actions.extend((candidate["unit_id"],) for candidate in candidates)
    if candidate_count >= 2:
        actions.append((candidates[0]["unit_id"], candidates[1]["unit_id"]))
    payload = {
        "q_t": {
            "user_request": "find the correct invoice",
            "latest_action_observation": {"result": "missing"},
            "currently_visible_raw": [],
        },
        "d_t": {"tool_calls": [], "text": "draft", "kind": "STOP"},
        "candidates": candidates,
        "allowed_sets": [
            {"action_id": index, "selected_ids": list(action)}
            for index, action in enumerate(actions)
        ],
    }
    return payload, actions


def test_defaults_are_pinned_local_and_construction_is_lazy(monkeypatch):
    monkeypatch.setattr(
        local,
        "_import_runtime",
        lambda _device: pytest.fail("construction must not load a model"),
    )
    models = local.LocalSelectionModels({})
    config = models.public_config()
    assert config["embedding"]["revision"] == local.EMBEDDING_REVISION
    assert config["reranker"]["revision"] == local.RERANKER_REVISION
    assert config["selector"]["revision"] == local.SELECTOR_REVISION
    assert config["selector"]["model_name_or_path"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert config["embedding"]["max_input_tokens"] == 32768
    assert config["reranker"]["max_input_tokens"] == 32768
    assert all(config[role]["local_files_only"] for role in config)
    assert models.drain_receipts() == []


def test_embedding_separates_query_document_and_caches_documents(monkeypatch):
    registry = install_runtime(monkeypatch)
    models = local.LocalSelectionModels(
        {"embedding": {"query_instruction": "retrieve corrective evidence"}}
    )
    query = models.embed(texts=["draft lookup"], purpose="query", config={})
    document = models.embed(texts=["archived exact value"], purpose="document")
    cached = models.embed(texts=["archived exact value"], purpose="document")

    tokenizer = instances(registry, "tokenizer_instance")[0]
    model = instances(registry, "model_instance")[0]
    tokenized_texts = [text for call, _kwargs in tokenizer.calls for text in call]
    assert "Instruct: retrieve corrective evidence\nQuery:draft lookup" in tokenized_texts
    assert "archived exact value" in tokenized_texts
    assert all("Instruct:" not in text for text in tokenized_texts if text == "archived exact value")
    assert query and document == cached
    assert math.isclose(sum(value * value for value in query[0]), 1.0, rel_tol=1e-6)
    assert model.forward_calls == 2
    assert len([row for row in registry if row[0] == "embedding"]) == 1

    receipts = models.drain_receipts()
    assert [row["capability"] for row in receipts].count("model_load") == 1
    assert receipts[-1]["capability"] == "embed"
    assert receipts[-1]["cache_hit_count"] == 1
    assert receipts[-1]["computed_count"] == 0
    assert receipts[-1]["usage"]["prompt_tokens"] == 0
    assert receipts[-1]["cost"] is None
    assert models.drain_receipts() == []


def test_embedding_budget_error_never_requests_tokenizer_truncation(monkeypatch):
    registry = install_runtime(monkeypatch)
    models = local.LocalSelectionModels(
        {"embedding": {"max_input_tokens": 4, "query_instruction": "instruction"}}
    )
    with pytest.raises(local.PayloadBudgetError, match="input was not truncated"):
        models.embed(texts=["long query"], purpose="query")
    tokenizer = instances(registry, "tokenizer_instance")[0]
    assert tokenizer.calls[-1][1]["truncation"] is False


def test_embedding_budget_is_capped_by_loaded_model_context(monkeypatch):
    registry = install_runtime(monkeypatch)
    models = local.LocalSelectionModels({})
    bundle = models._bundle("embedding")
    bundle["model"].config.max_position_embeddings = 4
    with pytest.raises(local.PayloadBudgetError, match="explicit 4-token budget"):
        models.embed(texts=["archive"], purpose="document")
    assert instances(registry, "model_instance")[0].forward_calls == 0


def test_retrieval_query_limit_keeps_under_cap_routes_byte_identical(monkeypatch):
    install_runtime(monkeypatch)
    models = local.LocalSelectionModels(
        {"embedding": {"max_input_tokens": 256, "query_instruction": "retrieve"}}
    )
    task, draft = "task alpha", "CALLS draft beta"
    expected = models.embed(
        texts=[task, draft, task + "\n" + draft], purpose="query", config={}
    )
    actual, receipt = models.embed_retrieval_queries(
        task=task,
        draft=draft,
        overflow_policy="task_head_tail_preserve_draft_v1",
    )
    assert actual == expected
    assert all(not row["overflow_applied"] for row in receipt["routes"])
    assert receipt["combined_retained_fields"]["task"]["retention"] == "full"
    assert receipt["combined_retained_fields"]["draft"]["retention"] == "full"


def test_retrieval_query_limit_bounds_only_combined_and_records_fields(monkeypatch):
    install_runtime(monkeypatch)
    models = local.LocalSelectionModels(
        {"embedding": {"max_input_tokens": 160, "query_instruction": "retrieve"}}
    )
    task, draft = "t" * 125, "DRAFT-CALL-WITH-ARGUMENTS"
    vectors, receipt = models.embed_retrieval_queries(
        task=task,
        draft=draft,
        overflow_policy="task_head_tail_preserve_draft_v1",
    )
    assert len(vectors) == 3
    routes = {row["name"]: row for row in receipt["routes"]}
    assert routes["task_semantic"]["retained_sha256"] == routes["task_semantic"]["original_sha256"]
    assert routes["draft_semantic"]["retained_sha256"] == routes["draft_semantic"]["original_sha256"]
    assert routes["combined_semantic"]["overflow_applied"] is True
    assert routes["combined_semantic"]["retained_input_tokens"] <= 160
    fields = receipt["combined_retained_fields"]
    assert fields["task"]["retention"] == "head_tail"
    assert 0 <= fields["task"]["retained_tokens"] < fields["task"]["original_tokens"]
    assert fields["draft"]["retention"] == "full"
    assert fields["draft"]["retained_tokens"] == fields["draft"]["original_tokens"]


def test_retrieval_query_limit_bounds_standalone_task_but_never_draft(monkeypatch):
    install_runtime(monkeypatch)
    models = local.LocalSelectionModels(
        {"embedding": {"max_input_tokens": 96, "query_instruction": "retrieve"}}
    )
    _, receipt = models.embed_retrieval_queries(
        task="t" * 120,
        draft="draft",
        overflow_policy="task_head_tail_preserve_draft_v1",
    )
    task_route = receipt["routes"][0]
    assert task_route["overflow_applied"] is True
    assert task_route["retained_input_tokens"] <= 96
    assert task_route["retained_fields"]["task"]["retention"] == "head_tail"

    with pytest.raises(
        local.PayloadBudgetError, match="complete draft was required and was not truncated"
    ):
        models.embed_retrieval_queries(
            task="task",
            draft="d" * 120,
            overflow_policy="task_head_tail_preserve_draft_v1",
        )


def test_retrieval_query_limit_replays_35895_token_overflow_shape(monkeypatch):
    install_runtime(monkeypatch)
    models = local.LocalSelectionModels(
        {"embedding": {"max_input_tokens": 32768, "query_instruction": "i"}}
    )
    # FakeTokenizer is one token per character, so this reproduces the observed
    # production overflow count without storing the prompt itself.
    overhead = len("Instruct: i\nQuery:")
    task = "t" * 32000
    draft = "d" * (35895 - overhead - 1 - len(task))
    _, receipt = models.embed_retrieval_queries(
        task=task,
        draft=draft,
        overflow_policy="task_head_tail_preserve_draft_v1",
    )
    combined = receipt["routes"][2]
    assert combined["original_input_tokens"] == 35895
    assert combined["retained_input_tokens"] <= 32768
    assert receipt["combined_retained_fields"]["draft"]["retention"] == "full"


def test_missing_runtime_dependency_is_actionable(monkeypatch):
    def unavailable(_device):
        raise local.DependencyUnavailableError(
            "local selection models require torch and transformers>=4.51.0"
        )

    monkeypatch.setattr(local, "_import_runtime", unavailable)
    models = local.LocalSelectionModels({})
    with pytest.raises(local.DependencyUnavailableError, match="transformers>=4.51.0"):
        models.embed(texts=["archive"], purpose="document")


def test_model_load_error_names_offline_provisioning(monkeypatch):
    class BrokenFactory:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise OSError("snapshot absent")

    runtime = SimpleNamespace(
        __version__="5.8.0",
        AutoConfig=BrokenFactory,
        AutoTokenizer=BrokenFactory,
        AutoModel=BrokenFactory,
        AutoModelForCausalLM=BrokenFactory,
    )
    monkeypatch.setattr(local, "_import_runtime", lambda _device: (torch, runtime))
    models = local.LocalSelectionModels({})
    with pytest.raises(local.ModelLoadError, match="provision that snapshot locally"):
        models.embed(texts=["archive"], purpose="document")


def test_reranker_returns_yes_probability_and_receipt(monkeypatch):
    registry = install_runtime(monkeypatch)
    models = local.LocalSelectionModels({})
    scores = models.rerank("verify invoice", ["supporting value", "unrelated value"])
    assert scores == pytest.approx([0.880797, 0.119203], rel=1e-5)

    tokenizer = instances(registry, "tokenizer_instance")[0]
    model = instances(registry, "model_instance")[0]
    assert all(call[1]["truncation"] is False for call in tokenizer.calls)
    assert model.forward_kwargs == [{"logits_to_keep": 1}]
    receipt = models.last_receipt("rerank")
    assert receipt["model"] == local.RERANKER_MODEL
    assert receipt["resolved_revision"] == "resolved-reranker"
    assert receipt["input_count"] == 2
    assert receipt["usage"]["prompt_tokens"] > 0
    assert receipt["usage"]["completion_tokens"] == 0
    assert receipt["cost"] is None


def test_reranker_rejects_over_budget_pair_without_truncating(monkeypatch):
    registry = install_runtime(monkeypatch)
    models = local.LocalSelectionModels({"reranker": {"max_input_tokens": 8}})
    with pytest.raises(local.PayloadBudgetError, match="input was not truncated"):
        models.rerank("query", ["full exact document"])
    tokenizer = instances(registry, "tokenizer_instance")[0]
    assert tokenizer.calls[-1][1]["truncation"] is False


def test_retrieval_reranker_is_byte_identical_below_cap(monkeypatch):
    install_runtime(monkeypatch)
    models = local.LocalSelectionModels({})
    task, draft = "verify invoice", "CALLS lookup"
    documents = ["supporting value", "unrelated value"]
    expected = models.rerank(task + "\n" + draft, documents)
    actual = models.rerank_retrieval_candidates(
        task=task,
        draft=draft,
        documents=documents,
        overflow_policy="task_head_tail_preserve_draft_v1",
    )
    assert actual == expected
    receipt = models.last_receipt("reranker_input_limit")
    assert receipt["overflow_applied"] is False
    assert receipt["original_query_sha256"] == receipt["retained_query_sha256"]
    assert all(row["retention"] == "full" for row in receipt["fields"]["documents"])


def test_retrieval_reranker_limits_one_task_query_and_keeps_all_documents(monkeypatch):
    install_runtime(monkeypatch)
    models = local.LocalSelectionModels({"reranker": {"max_input_tokens": 512}})
    documents = ["first exact candidate", "second exact candidate is longer"]
    scores = models.rerank_retrieval_candidates(
        task="t" * 400,
        draft="CALLS draft with arguments",
        documents=documents,
        overflow_policy="task_head_tail_preserve_draft_v1",
    )
    assert len(scores) == len(documents)
    receipt = models.last_receipt("reranker_input_limit")
    assert receipt["overflow_applied"] is True
    assert max(receipt["retained_pair_tokens"]) <= 512
    assert receipt["fields"]["task"]["retention"] == "head_tail"
    assert receipt["fields"]["draft"]["retention"] == "full"
    assert [row["sha256"] for row in receipt["fields"]["documents"]] == [
        local._sha256(document) for document in documents
    ]


def test_selector_uses_non_thinking_deterministic_constrained_generation(monkeypatch):
    registry = install_runtime(monkeypatch)
    models = local.LocalSelectionModels({})
    payload, actions = selector_payload()
    assert models.choose_action(payload, actions) == 3

    tokenizer = instances(registry, "tokenizer_instance")[0]
    model = instances(registry, "model_instance")[0]
    assert tokenizer.template_calls[-1][1] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    generation = model.generate_calls[-1]
    assert generation["do_sample"] is False
    assert generation["max_new_tokens"] == 16
    assert callable(generation["prefix_allowed_tokens_fn"])

    selector_load = next(
        row for row in registry if row[0] == "causal" and row[1] == local.SELECTOR_MODEL
    )
    assert selector_load[2]["local_files_only"] is True
    assert selector_load[2]["revision"] == local.SELECTOR_REVISION
    receipt = models.last_receipt("choose_action")
    assert receipt["candidate_count"] == 2
    assert receipt["action_count"] == 4
    assert receipt["selected_action_id"] == 3
    assert receipt["usage"]["completion_tokens"] == 2


def test_selector_action_zero_is_the_legal_empty_set(monkeypatch):
    registry = install_runtime(monkeypatch)
    models = local.LocalSelectionModels({})
    bundle = models._bundle("selector")
    bundle["model"].choice = 0
    payload, actions = selector_payload()

    action_id = models.choose_action(payload, actions)

    assert action_id == 0
    assert actions[action_id] == ()
    assert models.last_receipt("choose_action")["selected_action_id"] == 0


def test_custom_selector_path_does_not_inherit_default_hub_revision():
    models = local.LocalSelectionModels(
        {"selector": {"model_name_or_path": "/models/plain-qwen3-4b"}}
    )
    assert models.public_config()["selector"]["revision"] is None


def test_selector_rejects_c2kv_actor_config_before_weight_load(monkeypatch):
    registry = []
    runtime = fake_runtime(registry)
    runtime.AutoConfig = FakeConfigFactory(
        registry,
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "max_position_embeddings": 262144,
            "use_sliding_window": False,
            "history_memory_arm": "C",
            "gist_type": "dynamic-interleave",
        },
    )
    monkeypatch.setattr(local, "_import_runtime", lambda _device: (torch, runtime))
    models = local.LocalSelectionModels({})
    payload, actions = selector_payload()

    with pytest.raises(local.ModelLoadError, match="not a C2KV actor checkpoint"):
        models.choose_action(payload, actions)

    assert any(row[0] == "config" for row in registry)
    assert not any(row[0] in {"tokenizer", "causal"} for row in registry)


def test_local_selector_requires_every_indexed_weight_shard(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"layer":"missing.safetensors"}}', encoding="utf-8"
    )
    with pytest.raises(local.ModelLoadError, match="missing indexed weight shards"):
        local._local_artifact_identity(str(tmp_path), require_weights=True)


def test_selector_rejects_configured_budget_above_native_context():
    settings = local.LocalSelectionModels({}).public_config()["selector"]
    config = SimpleNamespace(
        model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
        max_position_embeddings=settings["max_input_tokens"],
        use_sliding_window=False,
    )
    with pytest.raises(local.ModelLoadError, match="exceeding native context"):
        local._validate_selector_model_config(config, settings)


def test_selector_rejects_oversized_exact_payload_without_loading(monkeypatch):
    registry = install_runtime(monkeypatch)
    models = local.LocalSelectionModels(
        {"selector": {"max_input_tokens": 8}}
    )
    payload, actions = selector_payload()
    with pytest.raises(local.PayloadBudgetError, match="input was not truncated"):
        models.choose_action(payload, actions)
    tokenizer = instances(registry, "tokenizer_instance")[0]
    assert tokenizer.calls[-1][1]["truncation"] is False


def test_selector_rejects_more_than_eight_candidates_before_model_load(monkeypatch):
    monkeypatch.setattr(
        local,
        "_import_runtime",
        lambda _device: pytest.fail("invalid payload must fail before model load"),
    )
    models = local.LocalSelectionModels({})
    payload, actions = selector_payload(candidate_count=9)
    with pytest.raises(local.PayloadBudgetError, match="at most eight"):
        models.choose_action(payload, actions)


def test_selector_requires_payload_action_table_to_match_legal_actions(monkeypatch):
    monkeypatch.setattr(
        local,
        "_import_runtime",
        lambda _device: pytest.fail("invalid payload must fail before model load"),
    )
    models = local.LocalSelectionModels({})
    payload, actions = selector_payload()
    payload["allowed_sets"][1]["selected_ids"] = ["u2"]
    with pytest.raises(ValueError, match="differs from allowed_actions"):
        models.choose_action(payload, actions)


def test_selector_fixed_generation_contract_cannot_be_overridden():
    with pytest.raises(ValueError, match="max_new_tokens is fixed at 16"):
        local.LocalSelectionModels({"selector": {"max_new_tokens": 32}})
    with pytest.raises(ValueError, match="enable_thinking is fixed at false"):
        local.LocalSelectionModels({"selector": {"enable_thinking": True}})
    with pytest.raises(ValueError, match="do_sample is fixed at false"):
        local.LocalSelectionModels({"selector": {"do_sample": True}})


def test_action_prefix_trie_supports_overlapping_multitoken_ids():
    callback = local._prefix_allowed_tokens(
        prompt_length=2, choices=[(11,), (11, 10), (12,)], eos_token_id=2
    )
    assert callback(0, FakeTensor([90, 91])) == [11, 12]
    assert callback(0, FakeTensor([90, 91, 11])) == [2, 10]
    assert callback(0, FakeTensor([90, 91, 11, 10])) == [2]


def test_full_selector_smoke_fixture_hits_protocol_bounds():
    payload, actions = local._selector_smoke_payload(full_fixture=True)
    assert len(payload["candidates"]) == 8
    assert len(actions) == 39
    assert actions[0] == ()
    assert all(1 <= len(action) <= 4 for action in actions[1:])
    assert payload["allowed_sets"][-1]["action_id"] == 38


class FakeRNG:
    def __init__(self, state):
        self.state = state

    def get_rng_state(self):
        return self.state

    def set_rng_state(self, state):
        self.state = state


class FakeDeviceRNG:
    def __init__(self):
        self.states = {}
        self.get_calls = []
        self.set_calls = []
        self.synchronize_calls = []

    def get_rng_state(self, device):
        self.get_calls.append(device)
        return self.states.setdefault(device, 100)

    def set_rng_state(self, state, device):
        self.set_calls.append(device)
        self.states[device] = state

    def synchronize(self, device):
        self.synchronize_calls.append(device)


class RNGTorch(FakeTorch):
    def __init__(self):
        super().__init__()
        self.random = FakeRNG(50)
        self.cuda = FakeDeviceRNG()
        self.npu = FakeDeviceRNG()


def _numpy_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


@pytest.mark.parametrize(
    ("device", "backend_name"), (("cuda:2", "cuda"), ("npu:3", "npu"))
)
@pytest.mark.parametrize("fail", (False, True))
def test_first_lazy_load_restores_rng_on_success_and_error(
    monkeypatch, device, backend_name, fail
):
    rng_torch = RNGTorch()
    selected = getattr(rng_torch, backend_name)

    def consume_rng():
        random.random()
        np.random.random()
        rng_torch.random.state += 1
        selected.states[device] = selected.states.get(device, 100) + 1

    class ConsumingModel(FakeBaseModel):
        def __call__(self, *args, **kwargs):
            consume_rng()
            return super().__call__(*args, **kwargs)

    class ConsumingFactory:
        def __init__(self, role):
            self.role = role

        def from_pretrained(self, *_args, **_kwargs):
            consume_rng()
            if fail:
                raise OSError("fixture load failure")
            if self.role == "tokenizer":
                return FakeTokenizer()
            return ConsumingModel("embedding")

    runtime = SimpleNamespace(
        __version__="5.8.0",
        AutoConfig=ConsumingFactory("config"),
        AutoTokenizer=ConsumingFactory("tokenizer"),
        AutoModel=ConsumingFactory("model"),
        AutoModelForCausalLM=ConsumingFactory("model"),
    )
    monkeypatch.setattr(local, "_import_runtime", lambda _device: (rng_torch, runtime))
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    cpu_before = rng_torch.random.state
    selected.states[device] = 73

    models = local.LocalSelectionModels({"embedding": {"device": device}})
    if fail:
        with pytest.raises(local.ModelLoadError, match="failed to load"):
            models.embed(texts=["archive"], purpose="document")
    else:
        assert models.embed(texts=["archive"], purpose="document")

    assert random.getstate() == python_before
    assert _numpy_state_equal(np.random.get_state(), numpy_before)
    assert rng_torch.random.state == cpu_before
    assert selected.states[device] == 73
    assert selected.get_calls and set(selected.get_calls) == {device}
    assert selected.synchronize_calls == ([] if fail else [device, device])
    unrelated = rng_torch.npu if backend_name == "cuda" else rng_torch.cuda
    assert unrelated.get_calls == []
    assert unrelated.set_calls == []
    assert unrelated.synchronize_calls == []
