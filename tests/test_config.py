from __future__ import annotations

import os
from pathlib import Path

import pytest

from imo_agent import config as config_mod

ROOT = Path(__file__).resolve().parent.parent


def test_shipped_config_validates(cfg):
    assert config_mod.validate(cfg) == []          # no unreachable nodes
    assert cfg.pipeline.start == "solve"
    assert cfg.default_model in cfg.models


def test_every_prompt_reference_resolves(cfg):
    for node in cfg.pipeline.nodes.values():
        if node.system:
            assert node.system in cfg.prompts
        for spec in node.messages:
            if spec.prompt:
                assert spec.prompt in cfg.prompts


def test_yaml_boolean_labels_are_normalised(cfg):
    judge = cfg.pipeline.nodes["judge"]
    assert set(judge.edges) == {"yes", "no"}
    assert "yes" in judge.match


def test_env_expansion_with_default():
    os.environ.pop("IMO_UNSET_FOR_TEST", None)
    assert config_mod.expand_env("${IMO_UNSET_FOR_TEST:-fallback}") == "fallback"
    os.environ["IMO_SET_FOR_TEST"] = "real"
    assert config_mod.expand_env("${IMO_SET_FOR_TEST:-fallback}") == "real"


def test_undefined_next_target_is_rejected(cfg):
    cfg.pipeline.nodes["solve"].next = "nowhere"
    with pytest.raises(config_mod.ConfigError, match="nowhere"):
        config_mod.validate(cfg)


def test_guard_referencing_an_unwritten_name_is_rejected(cfg):
    cfg.pipeline.nodes["check"].accept_if = "mystery >= 1"
    with pytest.raises(config_mod.ConfigError, match="mystery"):
        config_mod.validate(cfg)


def test_missing_prompt_is_rejected(cfg):
    cfg.pipeline.nodes["solve"].system = "not_a_prompt"
    with pytest.raises(config_mod.ConfigError, match="not_a_prompt"):
        config_mod.validate(cfg)
