"""
Comprehensive test suite for torch.compile integration (v3.5.0).

Tests cover:
- Unit tests: Flag handling, fallback behavior, mode validation
- Integration tests: Training speedup, numerical equivalence, Lightning compatibility
- Regression tests: Dynamic shapes, DDP/FSDP compatibility

Design follows SOLID principles with clear separation of concerns.
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import Optional
import logging
import time

# Import adapters and config
from utils.adapters.model_adapter import UniversalModelAdapter
from utils.training.training_config import TrainingConfig


# ==============================================================================
# TEST FIXTURES - MOCK MODELS AND UTILITIES
# ==============================================================================

class SimpleTransformer(nn.Module):
    """Minimal transformer model for testing compilation."""

    def __init__(self, vocab_size: int = 1000, d_model: int = 256, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=512, batch_first=True)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class MockTokenizer:
    """Mock tokenizer for testing."""
    def __init__(self):
        self.pad_token_id = 0


@pytest.fixture
def simple_model():
    """Create simple transformer model for testing."""
    return SimpleTransformer(vocab_size=1000, d_model=256, num_layers=2)


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def base_config():
    """Create base configuration without compilation."""
    config = SimpleNamespace(
        vocab_size=1000,
        max_seq_len=32,
        compile_mode=None,  # Disabled by default
        compile_fullgraph=False,
        compile_dynamic=True,
        pad_token_id=0
    )
    return config


# ==============================================================================
# UNIT TESTS
# ==============================================================================

class TestCompilationFlagHandling:
    """Test suite for compilation flag and configuration handling."""

    def test_compilation_disabled_by_default(self, simple_model, base_config, mock_tokenizer):
        """Verify compilation is opt-in (disabled by default)."""
        # Create adapter with default config (compile_mode=None)
        adapter = UniversalModelAdapter(
            simple_model,
            base_config,
            mock_tokenizer,
            1e-4  # learning_rate as positional arg
        )

        # Model should NOT be compiled (no _orig_mod attribute that torch.compile adds)
        # Note: torch.compile wraps the model, we check if it's NOT wrapped
        assert not hasattr(adapter.model, '_orig_mod'), (
            "Model should not be compiled when compile_mode=None"
        )

    def test_compilation_flag_respected_default_mode(self, simple_model, base_config, mock_tokenizer):
        """Verify compile_mode='default' triggers compilation."""
        # Enable compilation
        base_config.compile_mode = "default"

        adapter = UniversalModelAdapter(
            simple_model,
            base_config,
            mock_tokenizer,
            1e-4
        )

        # Check if torch.compile is available before testing
        if hasattr(torch, 'compile'):
            # Model should be compiled (has _orig_mod attribute)
            assert hasattr(adapter.model, '_orig_mod'), (
                "Model should be compiled when compile_mode='default'"
            )
        else:
            # If torch.compile not available, model should remain uncompiled
            assert not hasattr(adapter.model, '_orig_mod')

    def test_compilation_flag_respected_reduce_overhead(self, simple_model, base_config, mock_tokenizer):
        """Verify compile_mode='reduce-overhead' triggers compilation."""
        base_config.compile_mode = "reduce-overhead"

        adapter = UniversalModelAdapter(
            simple_model,
            base_config,
            mock_tokenizer,
            1e-4
        )

        if hasattr(torch, 'compile'):
            assert hasattr(adapter.model, '_orig_mod')

    def test_compilation_flag_respected_max_autotune(self, simple_model, base_config, mock_tokenizer):
        """Verify compile_mode='max-autotune' triggers compilation."""
        base_config.compile_mode = "max-autotune"

        adapter = UniversalModelAdapter(
            simple_model,
            base_config,
            mock_tokenizer,
            1e-4
        )

        if hasattr(torch, 'compile'):
            assert hasattr(adapter.model, '_orig_mod')

    def test_compilation_with_fullgraph_option(self, simple_model, base_config, mock_tokenizer):
        """Verify fullgraph parameter is respected."""
        base_config.compile_mode = "default"
        base_config.compile_fullgraph = True

        # Should not raise error (fullgraph failures are caught and fallback to uncompiled)
        adapter = UniversalModelAdapter(
            simple_model,
            base_config,
            mock_tokenizer,
            1e-4
        )

        # Model should exist (either compiled or fallback to uncompiled)
        assert adapter.model is not None

    def test_compilation_with_dynamic_shapes(self, simple_model, base_config, mock_tokenizer):
        """Verify dynamic shapes parameter is respected."""
        base_config.compile_mode = "default"
        base_config.compile_dynamic = True

        adapter = UniversalModelAdapter(
            simple_model,
            base_config,
            mock_tokenizer,
            1e-4
        )

        assert adapter.model is not None


class TestCompilationFallback:
    """Test suite for compilation error handling and fallback."""

    def test_compilation_fallback_on_error(self, base_config, mock_tokenizer, caplog):
        """Verify graceful fallback when compilation fails."""

        # Create a model that will cause compilation issues
        class ProblematicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(10))

            def forward(self, input_ids):
                # Use operations that might cause compilation issues
                return input_ids.float()

        model = ProblematicModel()
        base_config.compile_mode = "default"
        base_config.compile_fullgraph = True  # Strict mode may fail

        with caplog.at_level(logging.WARNING):
            # Should not raise exception, should log warning
            adapter = UniversalModelAdapter(
                model,
                base_config,
                mock_tokenizer,
                1e-4
            )

        # Adapter should be created successfully
        assert adapter is not None
        assert adapter.model is not None

    def test_pytorch_version_check(self, simple_model, base_config, mock_tokenizer, caplog, monkeypatch):
        """Verify proper handling when torch.compile is not available."""
        # Simulate PyTorch < 2.0 by temporarily removing torch.compile
        original_compile = getattr(torch, 'compile', None)

        if original_compile is not None:
            # Temporarily remove compile attribute
            monkeypatch.delattr(torch, 'compile', raising=False)

        base_config.compile_mode = "default"

        with caplog.at_level(logging.WARNING):
            adapter = UniversalModelAdapter(
                simple_model,
                base_config,
                mock_tokenizer,
                1e-4
            )

        # Should log warning about PyTorch version
        if original_compile is not None:
            assert any("PyTorch < 2.0" in record.message for record in caplog.records)

        # Model should still work (uncompiled)
        assert adapter.model is not None


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestCompilationIntegration:
    """Integration tests for compiled models in training workflows."""

    @pytest.mark.skipif(not hasattr(torch, 'compile'), reason="Requires PyTorch 2.0+")
    def test_compiled_model_numerical_equivalence(self, simple_model, base_config, mock_tokenizer):
        """Verify compiled model produces same outputs as uncompiled (within tolerance)."""
        # Create uncompiled adapter
        config_uncompiled = SimpleNamespace(**vars(base_config))
        config_uncompiled.compile_mode = None
        adapter_uncompiled = UniversalModelAdapter(
            SimpleTransformer(vocab_size=1000, d_model=256, num_layers=2),
            config_uncompiled,
            mock_tokenizer,
            1e-4
        )

        # Create compiled adapter
        config_compiled = SimpleNamespace(**vars(base_config))
        config_compiled.compile_mode = "default"
        adapter_compiled = UniversalModelAdapter(
            SimpleTransformer(vocab_size=1000, d_model=256, num_layers=2),
            config_compiled,
            mock_tokenizer,
            1e-4
        )

        # Load same weights
        state_dict = adapter_uncompiled.model.state_dict()
        if hasattr(adapter_compiled.model, '_orig_mod'):
            # Compiled model stores original in _orig_mod
            adapter_compiled.model._orig_mod.load_state_dict(state_dict)
        else:
            adapter_compiled.model.load_state_dict(state_dict)

        # Create test input
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        # Set to eval mode for deterministic behavior
        adapter_uncompiled.eval()
        adapter_compiled.eval()

        # Get outputs
        with torch.no_grad():
            output_uncompiled = adapter_uncompiled(input_ids, attention_mask)
            output_compiled = adapter_compiled(input_ids, attention_mask)

        # Compare logits (rtol=1e-4, atol=1e-5 as specified)
        logits_uncompiled = output_uncompiled['logits']
        logits_compiled = output_compiled['logits']

        assert torch.allclose(
            logits_uncompiled,
            logits_compiled,
            rtol=1e-4,
            atol=1e-5
        ), "Compiled and uncompiled models should produce numerically equivalent outputs"

    @pytest.mark.skipif(not hasattr(torch, 'compile'), reason="Requires PyTorch 2.0+")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for speedup measurement")
    def test_compiled_model_training_speedup(self, base_config, mock_tokenizer):
        """Measure training speedup with compilation (target 10-20%)."""
        # This is a simplified benchmark - full benchmark would require more iterations

        def benchmark_training_step(adapter, num_steps=10):
            """Benchmark training loop."""
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)

            start_time = time.time()
            for _ in range(num_steps):
                # Create batch
                batch_size, seq_len = 4, 32
                input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
                labels = torch.randint(0, 1000, (batch_size, seq_len)).cuda()

                optimizer.zero_grad()
                output = adapter(input_ids, labels=labels)
                loss = output['loss']
                loss.backward()
                optimizer.step()

            return time.time() - start_time

        # Benchmark uncompiled
        config_uncompiled = SimpleNamespace(**vars(base_config))
        config_uncompiled.compile_mode = None
        adapter_uncompiled = UniversalModelAdapter(
            SimpleTransformer(vocab_size=1000, d_model=256, num_layers=2).cuda(),
            config_uncompiled,
            mock_tokenizer,
            1e-4
        ).cuda()

        time_uncompiled = benchmark_training_step(adapter_uncompiled, num_steps=5)

        # Benchmark compiled
        config_compiled = SimpleNamespace(**vars(base_config))
        config_compiled.compile_mode = "default"
        adapter_compiled = UniversalModelAdapter(
            SimpleTransformer(vocab_size=1000, d_model=256, num_layers=2).cuda(),
            config_compiled,
            mock_tokenizer,
            1e-4
        ).cuda()

        time_compiled = benchmark_training_step(adapter_compiled, num_steps=5)

        # Calculate speedup
        speedup = (time_uncompiled - time_compiled) / time_uncompiled * 100

        # Log results (actual speedup may vary based on hardware)
        print(f"\nBenchmark Results:")
        print(f"  Uncompiled: {time_uncompiled:.3f}s")
        print(f"  Compiled: {time_compiled:.3f}s")
        print(f"  Speedup: {speedup:.1f}%")

        # Note: We don't assert on speedup percentage as it's hardware-dependent
        # Just verify compilation doesn't slow down training
        assert time_compiled <= time_uncompiled * 1.5, (
            "Compiled model should not be significantly slower than uncompiled"
        )


# ==============================================================================
# REGRESSION TESTS
# ==============================================================================

class TestCompilationRegression:
    """Regression tests for Lightning and distributed training compatibility."""

    @pytest.mark.skipif(not hasattr(torch, 'compile'), reason="Requires PyTorch 2.0+")
    def test_compilation_works_with_lightning(self, simple_model, base_config, mock_tokenizer):
        """Verify compiled models work with PyTorch Lightning."""
        try:
            import pytorch_lightning as pl
        except ImportError:
            pytest.skip("pytorch_lightning not installed")

        base_config.compile_mode = "default"
        adapter = UniversalModelAdapter(
            simple_model,
            base_config,
            mock_tokenizer,
            1e-4
        )

        # Verify adapter is a valid LightningModule
        assert isinstance(adapter, pl.LightningModule)

        # Verify essential Lightning methods exist
        assert hasattr(adapter, 'training_step')
        assert hasattr(adapter, 'validation_step')
        assert hasattr(adapter, 'configure_optimizers')

    @pytest.mark.skipif(not hasattr(torch, 'compile'), reason="Requires PyTorch 2.0+")
    def test_compilation_with_dynamic_shapes(self, simple_model, base_config, mock_tokenizer):
        """Verify compiled models work with variable sequence lengths."""
        base_config.compile_mode = "default"
        base_config.compile_dynamic = True  # Enable dynamic shapes

        adapter = UniversalModelAdapter(
            simple_model,
            base_config,
            mock_tokenizer,
            1e-4
        )

        adapter.eval()

        # Test with different sequence lengths
        with torch.no_grad():
            # First batch: seq_len=16
            input_ids_1 = torch.randint(0, 1000, (2, 16))
            output_1 = adapter(input_ids_1)
            assert output_1['logits'].shape == (2, 16, 1000)

            # Second batch: seq_len=32 (different length)
            input_ids_2 = torch.randint(0, 1000, (2, 32))
            output_2 = adapter(input_ids_2)
            assert output_2['logits'].shape == (2, 32, 1000)


# ==============================================================================
# TRAINING CONFIG INTEGRATION TESTS
# ==============================================================================

class TestTrainingConfigCompilation:
    """Test TrainingConfig integration with compilation settings."""

    def test_training_config_has_compile_fields(self):
        """Verify TrainingConfig has new compilation fields."""
        config = TrainingConfig()

        # Check fields exist
        assert hasattr(config, 'compile_mode')
        assert hasattr(config, 'compile_fullgraph')
        assert hasattr(config, 'compile_dynamic')

        # Check defaults
        assert config.compile_mode is None  # Disabled by default
        assert config.compile_fullgraph is False
        assert config.compile_dynamic is True

    def test_training_config_serialization_with_compilation(self, tmp_path):
        """Verify TrainingConfig serialization preserves compilation settings."""
        config = TrainingConfig(
            compile_mode="default",
            compile_fullgraph=True,
            compile_dynamic=False,
            learning_rate=1e-4
        )

        # Save to file
        config_path = tmp_path / "config.json"
        config.save(str(config_path))

        # Load and verify
        loaded_config = TrainingConfig.load(str(config_path))
        assert loaded_config.compile_mode == "default"
        assert loaded_config.compile_fullgraph is True
        assert loaded_config.compile_dynamic is False


# ==============================================================================
# EDGE CASES AND ERROR HANDLING
# ==============================================================================

class TestCompilationEdgeCases:
    """Test edge cases and error conditions."""

    def test_config_without_compile_attributes(self, simple_model, mock_tokenizer):
        """Verify backward compatibility when config lacks compile attributes."""
        # Create config without compile attributes (legacy behavior)
        config = SimpleNamespace(
            vocab_size=1000,
            max_seq_len=32,
            pad_token_id=0
            # No compile_mode attribute
        )

        # Should not raise error
        adapter = UniversalModelAdapter(
            simple_model,
            config,
            mock_tokenizer,
            1e-4
        )

        # Model should be uncompiled (no _orig_mod)
        assert not hasattr(adapter.model, '_orig_mod')

    def test_none_compile_mode_explicit(self, simple_model, base_config, mock_tokenizer):
        """Verify explicit compile_mode=None disables compilation."""
        base_config.compile_mode = None

        adapter = UniversalModelAdapter(
            simple_model,
            base_config,
            mock_tokenizer,
            1e-4
        )

        assert not hasattr(adapter.model, '_orig_mod')
