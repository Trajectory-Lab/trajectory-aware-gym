"""Tests for cost normalization utilities and config integration."""

from __future__ import annotations

import pytest

from trajectory_aware_gym.config.core import Settings
from trajectory_aware_gym.metrics.cost_normalization import compute_normalized_cost

# Reference prices fixture matching the production YAML defaults
_REF_PRICES = {
    "ollama/qwen3-1.7b-base": {
        "input_per_1m_tokens": 0.10,
        "output_per_1m_tokens": 0.10,
    },
    "ollama/qwen3-4b-base": {
        "input_per_1m_tokens": 0.22,
        "output_per_1m_tokens": 0.22,
    },
}


# ---------------------------------------------------------------------------
# compute_normalized_cost
# ---------------------------------------------------------------------------


class TestComputeNormalizedCost:
    @pytest.mark.parametrize(
        ("model_id", "prompt_tokens", "completion_tokens", "expected"),
        [
            # 1M prompt + 1M completion @ $0.10/$0.10 = $0.20
            ("ollama/qwen3-1.7b-base", 1_000_000, 1_000_000, 0.20),
            # 100 prompt + 50 completion @ $0.10/$0.10 = $0.000015
            ("ollama/qwen3-1.7b-base", 100, 50, 0.000015),
            # 500K prompt + 200K completion @ $0.22/$0.22 = $0.154
            ("ollama/qwen3-4b-base", 500_000, 200_000, 0.154),
        ],
    )
    def test_known_model(self, model_id, prompt_tokens, completion_tokens, expected):
        result = compute_normalized_cost(
            model_id,
            prompt_tokens,
            completion_tokens,
            _REF_PRICES,
        )
        assert result == pytest.approx(expected)

    def test_unknown_model_returns_none(self):
        result = compute_normalized_cost(
            "bedrock/llama-8b",
            1000,
            500,
            _REF_PRICES,
        )
        assert result is None

    def test_zero_tokens_returns_zero(self):
        result = compute_normalized_cost(
            "ollama/qwen3-1.7b-base",
            0,
            0,
            _REF_PRICES,
        )
        assert result == 0.0

    def test_empty_reference_prices(self):
        result = compute_normalized_cost(
            "ollama/qwen3-1.7b-base",
            1000,
            500,
            {},
        )
        assert result is None

    def test_asymmetric_pricing(self):
        """Input and output prices can differ."""
        prices = {
            "custom/model": {
                "input_per_1m_tokens": 1.00,
                "output_per_1m_tokens": 3.00,
            }
        }
        # 1M input @ $1.00 + 1M output @ $3.00 = $4.00
        result = compute_normalized_cost("custom/model", 1_000_000, 1_000_000, prices)
        assert result == pytest.approx(4.00)


# ---------------------------------------------------------------------------
# Settings integration
# ---------------------------------------------------------------------------


class TestCostNormalizationConfig:
    @pytest.fixture(autouse=True)
    def _reset_settings(self):
        Settings.reset()
        yield
        # Reset and force reload from production YAML so downstream tests
        # that rely on the module-level ``settings`` singleton see correct values.
        Settings.reset()
        Settings()

    def test_settings_loads_cost_normalization(self):
        """Settings() loads the cost_normalization section from YAML."""
        s = Settings()
        ref = s.cost_normalization.reference_prices
        assert "ollama/qwen3-1.7b-base" in ref
        assert "ollama/qwen3-4b-base" in ref
        assert ref["ollama/qwen3-1.7b-base"]["input_per_1m_tokens"] == 0.10
        assert ref["ollama/qwen3-4b-base"]["output_per_1m_tokens"] == 0.22

    def test_settings_end_to_end_with_compute(self):
        """Full pipeline: Settings → reference_prices → compute_normalized_cost."""
        s = Settings()
        ref = s.cost_normalization.reference_prices
        result = compute_normalized_cost(
            "ollama/qwen3-1.7b-base",
            10_000,
            5_000,
            ref,
        )
        # 10K × $0.10/1M + 5K × $0.10/1M = $0.001 + $0.0005 = $0.0015
        assert result == pytest.approx(0.0015)

    def test_empty_section_yields_empty_dict(self, tmp_path):
        """If cost_normalization section is absent, reference_prices defaults to {}."""
        yaml_file = tmp_path / "config.yaml"
        # Minimal YAML with all required sections but no cost_normalization
        yaml_file.write_text(
            "aws:\n  region: us-east-1\n  bedrock_claude_sonnet_4_5: x\n"
            "  bedrock_llama_1b: x\n  bedrock_llama_3b: x\n  bedrock_llama_8b: x\n"
            "  bedrock_gemma_4b: x\n  bedrock_mistral_7b: x\n  bedrock_nemotron_9b: x\n"
            "  s3_bucket: x\n  s3_prefix: x\n"
            "ollama:\n  api_base: http://localhost\n  task_model_1_7b: x\n  task_model_4b: x\n"
            "sagemaker:\n  region: us-east-1\n  role_arn: x\n  instance_type: x\n"
            "  tgi_image_uri: x\n  endpoint_1_7b: x\n  endpoint_4b: x\n"
            "  model_id_1_7b: x\n  model_id_4b: x\n  deploy_timeout: 120\n"
            "gem:\n  max_steps: 10\n  temperature_train: 1.0\n  temperature_eval: 0.0\n"
            "  tool_timeout: 30\n"
            "gepa:\n  num_threads: 1\n  reflection_model: x\n"
            "experiment:\n  name: test\n  random_seed: 1\n  num_replications: 1\n"
            "logging:\n  level: INFO\n  file: test.log\n"
            "cost_tracking:\n  enabled: false\n  alert_threshold: 100.0\n"
            "fitness:\n  gamma: 0.99\n  lambda: 0.1\n  loop_penalty_weight: 1.0\n"
            "  step_efficiency_weight: 1.0\n  call_efficiency_weight: 0.0\n"
            "  max_steps: 50\n  loop_window: 3\n  call_budget_per_step: 8\n"
            "retry:\n  max_attempts: 1\n  initial_wait_seconds: 0.1\n  max_wait_seconds: 1.0\n"
            "  exponential_base: 2.0\n  jitter: false\n  litellm_num_retries: 0\n"
            "  boto3_retry_mode: standard\n  boto3_max_attempts: 1\n"
            "  sagemaker_read_timeout_seconds: 10\n  inference_semaphore_size: 1\n"
        )
        s = Settings(yaml_path=yaml_file)
        assert s.cost_normalization.reference_prices == {}
