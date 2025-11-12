"""
Unit tests for model adapter components.

Tests ModelSignatureInspector, ComputationalGraphExecutor, and UniversalModelAdapter
with various model architectures and signature patterns.
"""

import pytest
import torch
import torch.nn as nn
from utils.adapters.model_adapter import ModelSignatureInspector


# ==============================================================================
# TEST FIXTURES - MOCK MODELS
# ==============================================================================

class SimpleModel(nn.Module):
    """Model with simple forward(input_ids) signature."""
    def __init__(self, vocab_size=50257, d_model=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        return self.output(x)


class SimpleModelWithMask(nn.Module):
    """Model with forward(input_ids, attention_mask) signature."""
    def __init__(self, vocab_size=50257, d_model=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        return self.output(x)


class ComplexModel(nn.Module):
    """Model with complex signature requiring intermediate outputs."""
    def __init__(self, vocab_size=50257, d_model=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.ffn = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_0_tokens, mhsa_0_output, residual_0_output):
        """
        Simulates a generated model signature with intermediate outputs.
        In reality, these would be computed elsewhere, but for testing
        we accept them as parameters.
        """
        # This is just for signature testing - actual computation doesn't matter
        x = mhsa_0_output + residual_0_output
        x = self.ffn(x)
        return self.output(x)


class VeryComplexModel(nn.Module):
    """Model with many intermediate outputs."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 512)

    def forward(self, input_0_tokens, mhsa_0_output, residual_0_output,
                ffn_0_output, mhsa_1_output, residual_1_output,
                attention_mask=None):
        """Multiple intermediate outputs + optional param."""
        x = mhsa_0_output + residual_0_output + ffn_0_output
        x = x + mhsa_1_output + residual_1_output
        return self.linear(x)


# ==============================================================================
# TESTS FOR MODEL SIGNATURE INSPECTOR
# ==============================================================================

class TestModelSignatureInspector:
    """Test suite for ModelSignatureInspector."""

    def test_simple_model_signature(self):
        """Test inspector with simple forward(input_ids) signature."""
        model = SimpleModel()
        inspector = ModelSignatureInspector(model)

        # Check parameters
        params = inspector.get_parameters()
        assert params == ['input_ids']

        # Check required vs optional
        required = inspector.get_required_params()
        optional = inspector.get_optional_params()
        assert required == ['input_ids']
        assert optional == []

        # Check intermediate detection
        assert not inspector.requires_intermediate_outputs()
        assert inspector.is_simple_signature()
        assert inspector.get_intermediate_params() == []

    def test_simple_model_with_mask(self):
        """Test inspector with forward(input_ids, attention_mask=None)."""
        model = SimpleModelWithMask()
        inspector = ModelSignatureInspector(model)

        # Check parameters
        params = inspector.get_parameters()
        assert set(params) == {'input_ids', 'attention_mask'}

        # Check required vs optional
        required = inspector.get_required_params()
        optional = inspector.get_optional_params()
        assert required == ['input_ids']
        assert 'attention_mask' in optional

        # Should still be simple (no intermediates)
        assert not inspector.requires_intermediate_outputs()
        assert inspector.is_simple_signature()
        assert inspector.get_intermediate_params() == []

    def test_complex_model_signature(self):
        """Test inspector with complex signature requiring intermediates."""
        model = ComplexModel()
        inspector = ModelSignatureInspector(model)

        # Check parameters
        params = inspector.get_parameters()
        assert set(params) == {'input_0_tokens', 'mhsa_0_output', 'residual_0_output'}

        # All required (no defaults)
        required = inspector.get_required_params()
        assert len(required) == 3

        # Check intermediate detection
        assert inspector.requires_intermediate_outputs()
        assert not inspector.is_simple_signature()

        intermediates = inspector.get_intermediate_params()
        assert 'mhsa_0_output' in intermediates
        assert 'residual_0_output' in intermediates

    def test_very_complex_model_signature(self):
        """Test inspector with many intermediates and mixed params."""
        model = VeryComplexModel()
        inspector = ModelSignatureInspector(model)

        # Check intermediate detection
        assert inspector.requires_intermediate_outputs()
        assert not inspector.is_simple_signature()

        # Check intermediates
        intermediates = inspector.get_intermediate_params()
        expected_intermediates = [
            'mhsa_0_output',
            'residual_0_output',
            'ffn_0_output',
            'mhsa_1_output',
            'residual_1_output'
        ]
        assert set(intermediates) == set(expected_intermediates)

        # attention_mask should be optional
        optional = inspector.get_optional_params()
        assert 'attention_mask' in optional

    def test_analyze_method(self):
        """Test the analyze() method returns complete information."""
        model = ComplexModel()
        inspector = ModelSignatureInspector(model)

        analysis = inspector.analyze()

        # Check all expected keys present
        expected_keys = {
            'all_params',
            'required_params',
            'optional_params',
            'intermediate_params',
            'requires_intermediates',
            'is_simple',
            'signature_str'
        }
        assert set(analysis.keys()) == expected_keys

        # Verify content
        assert analysis['requires_intermediates'] is True
        assert analysis['is_simple'] is False
        assert len(analysis['intermediate_params']) == 2
        assert 'mhsa_0_output' in analysis['intermediate_params']

    def test_prefix_detection(self):
        """Test that all intermediate prefixes are detected correctly."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input_ids, mhsa_out, residual_out, ffn_out,
                        attention_out, mlp_out, layer_out):
                return input_ids

        model = TestModel()
        inspector = ModelSignatureInspector(model)

        # All prefixed params should be detected
        intermediates = inspector.get_intermediate_params()
        assert len(intermediates) == 6  # All except input_ids

        # Should require intermediates
        assert inspector.requires_intermediate_outputs()
        assert not inspector.is_simple_signature()

    def test_repr(self):
        """Test string representation."""
        model = SimpleModel()
        inspector = ModelSignatureInspector(model)

        repr_str = repr(inspector)
        assert 'ModelSignatureInspector' in repr_str
        assert 'SimpleModel' in repr_str
        assert 'input_ids' in repr_str


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestInspectorIntegration:
    """Integration tests with real-world scenarios."""

    def test_with_actual_transformer(self):
        """Test inspector with a real transformer architecture."""
        # Create a minimal transformer model
        class MiniTransformer(nn.Module):
            def __init__(self, vocab_size=1000, d_model=256):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True),
                    num_layers=2
                )
                self.output = nn.Linear(d_model, vocab_size)

            def forward(self, input_ids, attention_mask=None):
                x = self.embedding(input_ids)
                x = self.transformer(x)
                return self.output(x)

        model = MiniTransformer()
        inspector = ModelSignatureInspector(model)

        # Should be simple (no intermediates)
        assert inspector.is_simple_signature()
        assert not inspector.requires_intermediate_outputs()

        # Should handle attention_mask properly
        params = inspector.get_parameters()
        assert 'input_ids' in params
        assert 'attention_mask' in params

    def test_inspector_with_forward_pass(self):
        """Test that inspector doesn't interfere with actual forward pass."""
        model = SimpleModel()
        inspector = ModelSignatureInspector(model)

        # Inspector should not modify model
        input_ids = torch.randint(0, 1000, (2, 10))

        # Model should still work normally
        output = model(input_ids)
        assert output.shape == (2, 10, 50257)

        # Inspector analysis should still work
        assert inspector.is_simple_signature()


# ==============================================================================
# EDGE CASES
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_model_with_no_params(self):
        """Test model with only self in forward()."""
        class NoParamModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.randn(1, 10)

            def forward(self):
                return self.const

        model = NoParamModel()
        inspector = ModelSignatureInspector(model)

        # Should have empty param list
        assert inspector.get_parameters() == []
        assert inspector.get_required_params() == []
        assert inspector.is_simple_signature()

    def test_model_with_kwargs(self):
        """Test model with **kwargs in signature."""
        class KwargsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, input_ids, **kwargs):
                return self.linear(input_ids)

        model = KwargsModel()
        inspector = ModelSignatureInspector(model)

        # Should detect input_ids and **kwargs
        params = inspector.get_parameters()
        assert 'input_ids' in params
        # Note: **kwargs appears as 'kwargs' in parameters
        assert any('kwargs' in p.lower() for p in params)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
