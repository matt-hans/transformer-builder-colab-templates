"""
Tests for distributed training guardrails with notebook detection.

This module tests the automatic detection of Jupyter/Colab notebook environments
and the guardrails that prevent DDP/FSDP strategies from creating zombie processes.
"""

import os
import pytest
import sys
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from utils.training.training_core import TrainingCoordinator


class TestNotebookDetection:
    """Test suite for _is_running_in_notebook() method."""

    def test_notebook_detection_colab(self, monkeypatch):
        """Test detection of Google Colab environment."""
        # Mock google.colab import to simulate Colab environment
        fake_colab = MagicMock()
        monkeypatch.setitem(sys.modules, 'google.colab', fake_colab)

        # Should detect Colab as notebook
        assert TrainingCoordinator._is_running_in_notebook() is True

        # Clean up
        monkeypatch.delitem(sys.modules, 'google.colab', raising=False)

    def test_notebook_detection_jupyter(self, monkeypatch):
        """Test detection of Jupyter notebook."""
        # Mock get_ipython() by patching it in builtins
        class MockZMQShell:
            pass

        # Set the class name properly
        MockZMQShell.__name__ = 'ZMQInteractiveShell'

        mock_shell = MockZMQShell()

        def mock_get_ipython():
            return mock_shell

        # Patch get_ipython in builtins
        import builtins
        builtins.get_ipython = mock_get_ipython

        try:
            # Should detect Jupyter notebook
            assert TrainingCoordinator._is_running_in_notebook() is True
        finally:
            # Clean up
            if hasattr(builtins, 'get_ipython'):
                delattr(builtins, 'get_ipython')

    def test_not_notebook_ipython_terminal(self, monkeypatch):
        """Test IPython terminal is NOT detected as notebook."""
        # Mock get_ipython() to return TerminalInteractiveShell (IPython terminal)
        class MockTerminalShell:
            __class__.__name__ = 'TerminalInteractiveShell'

        def mock_get_ipython():
            return MockTerminalShell()

        # Patch get_ipython in the global namespace
        import builtins
        monkeypatch.setattr(builtins, 'get_ipython', mock_get_ipython, raising=False)

        # Should NOT detect IPython terminal as notebook
        assert TrainingCoordinator._is_running_in_notebook() is False

    def test_not_notebook_standard_python(self):
        """Test standard Python interpreter returns False."""
        # In standard Python, get_ipython() raises NameError
        # No mocking needed - this is the default state

        # Should NOT detect standard Python as notebook
        assert TrainingCoordinator._is_running_in_notebook() is False


class TestDDPGuardrails:
    """Test suite for DDP/FSDP guardrails in notebook environments."""

    def test_ddp_blocked_in_notebook(self, monkeypatch, tmp_path):
        """Test DDP strategy is forced to 'auto' in notebooks."""
        # Mock notebook detection - use staticmethod for proper signature
        monkeypatch.setattr(
            TrainingCoordinator,
            '_is_running_in_notebook',
            staticmethod(lambda: True)
        )

        # Ensure no override environment variable
        monkeypatch.delenv('ALLOW_NOTEBOOK_DDP', raising=False)

        # Create coordinator with DDP strategy
        coordinator = TrainingCoordinator(
            output_dir=str(tmp_path),
            strategy='ddp'
        )

        # Strategy should be forced to 'auto' in notebook
        assert coordinator.strategy == 'auto'

    def test_fsdp_blocked_in_notebook(self, monkeypatch, tmp_path):
        """Test FSDP strategy is forced to 'auto' in notebooks."""
        # Mock notebook detection
        monkeypatch.setattr(
            TrainingCoordinator,
            '_is_running_in_notebook',
            staticmethod(lambda: True)
        )

        # Ensure no override environment variable
        monkeypatch.delenv('ALLOW_NOTEBOOK_DDP', raising=False)

        # Create coordinator with FSDP strategy
        coordinator = TrainingCoordinator(
            output_dir=str(tmp_path),
            strategy='fsdp_native'
        )

        # Strategy should be forced to 'auto' in notebook
        assert coordinator.strategy == 'auto'

    def test_ddp_allowed_with_override(self, monkeypatch, tmp_path):
        """Test ALLOW_NOTEBOOK_DDP override allows DDP in notebooks."""
        # Mock notebook detection
        monkeypatch.setattr(
            TrainingCoordinator,
            '_is_running_in_notebook',
            staticmethod(lambda: True)
        )

        # Set override environment variable
        monkeypatch.setenv('ALLOW_NOTEBOOK_DDP', '1')

        # Create coordinator with DDP strategy
        coordinator = TrainingCoordinator(
            output_dir=str(tmp_path),
            strategy='ddp'
        )

        # Strategy should NOT be changed (override respected)
        assert coordinator.strategy == 'ddp'


class TestIntegrationTests:
    """Integration tests for distributed guardrails."""

    def test_ddp_allowed_in_standard_python(self, tmp_path):
        """Test DDP works normally outside notebooks."""
        # No mocking - standard Python environment
        # (get_ipython() will raise NameError, detecting non-notebook)

        # Create coordinator with DDP strategy
        coordinator = TrainingCoordinator(
            output_dir=str(tmp_path),
            strategy='ddp'
        )

        # Strategy should remain 'ddp' (no notebook detected)
        assert coordinator.strategy == 'ddp'

    def test_override_env_var_variations(self, monkeypatch, tmp_path):
        """Test various environment variable values for override."""
        # Mock notebook detection
        monkeypatch.setattr(
            TrainingCoordinator,
            '_is_running_in_notebook',
            staticmethod(lambda: True)
        )

        # Test '1' value
        monkeypatch.setenv('ALLOW_NOTEBOOK_DDP', '1')
        coord = TrainingCoordinator(output_dir=str(tmp_path), strategy='ddp')
        assert coord.strategy == 'ddp'

        # Test 'true' value
        monkeypatch.setenv('ALLOW_NOTEBOOK_DDP', 'true')
        coord = TrainingCoordinator(output_dir=str(tmp_path / 'test2'), strategy='ddp')
        assert coord.strategy == 'ddp'

        # Test 'yes' value
        monkeypatch.setenv('ALLOW_NOTEBOOK_DDP', 'yes')
        coord = TrainingCoordinator(output_dir=str(tmp_path / 'test3'), strategy='ddp')
        assert coord.strategy == 'ddp'

        # Test 'TRUE' (uppercase should work due to .lower())
        monkeypatch.setenv('ALLOW_NOTEBOOK_DDP', 'TRUE')
        coord = TrainingCoordinator(output_dir=str(tmp_path / 'test4'), strategy='ddp')
        assert coord.strategy == 'ddp'

        # Test invalid value (should force to 'auto')
        monkeypatch.setenv('ALLOW_NOTEBOOK_DDP', 'invalid')
        coord = TrainingCoordinator(output_dir=str(tmp_path / 'test5'), strategy='ddp')
        assert coord.strategy == 'auto'


class TestRegressionTests:
    """Regression tests to ensure existing guardrails still work."""

    def test_existing_device_count_guardrails_unchanged(self, monkeypatch, tmp_path):
        """Test existing device count guardrails still work after new changes."""
        # Mock no notebook environment
        monkeypatch.setattr(
            TrainingCoordinator,
            '_is_running_in_notebook',
            staticmethod(lambda: False)
        )

        # Create coordinator with DDP and devices=1 (should trigger device count guardrail)
        coordinator = TrainingCoordinator(
            output_dir=str(tmp_path),
            strategy='ddp',
            devices=1
        )

        # Create a minimal mock model for training
        import torch.nn as nn
        model = nn.Linear(10, 10)

        # Mock torch.cuda.is_available to return False (CPU environment)
        with patch('torch.cuda.is_available', return_value=False):
            # This should trigger the existing device count guardrail in train()
            # We're not actually calling train() here since that requires full setup
            # Just verify coordinator initialization works
            assert coordinator.strategy == 'ddp'  # Strategy unchanged during __init__

        # Note: The actual device count guardrail runs during train() method,
        # not during __init__. This test verifies __init__ doesn't break existing behavior.


# Additional edge case tests
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_auto_strategy_unchanged_in_notebook(self, monkeypatch, tmp_path):
        """Test 'auto' strategy is not modified in notebooks."""
        # Mock notebook detection
        monkeypatch.setattr(
            TrainingCoordinator,
            '_is_running_in_notebook',
            staticmethod(lambda: True)
        )

        # Create coordinator with 'auto' strategy (should not be changed)
        coordinator = TrainingCoordinator(
            output_dir=str(tmp_path),
            strategy='auto'
        )

        # Strategy should remain 'auto'
        assert coordinator.strategy == 'auto'

    def test_none_strategy_unchanged_in_notebook(self, monkeypatch, tmp_path):
        """Test None strategy is not modified in notebooks."""
        # Mock notebook detection
        monkeypatch.setattr(
            TrainingCoordinator,
            '_is_running_in_notebook',
            staticmethod(lambda: True)
        )

        # Create coordinator with None strategy (should not be changed)
        coordinator = TrainingCoordinator(
            output_dir=str(tmp_path),
            strategy=None
        )

        # Strategy should remain None
        assert coordinator.strategy is None

    def test_custom_strategy_unchanged_in_notebook(self, monkeypatch, tmp_path):
        """Test custom strategies (not ddp/fsdp_native) are not modified."""
        # Mock notebook detection
        monkeypatch.setattr(
            TrainingCoordinator,
            '_is_running_in_notebook',
            staticmethod(lambda: True)
        )

        # Create coordinator with custom strategy
        coordinator = TrainingCoordinator(
            output_dir=str(tmp_path),
            strategy='dp'  # DataParallel strategy
        )

        # Strategy should remain 'dp' (not in guardrail list)
        assert coordinator.strategy == 'dp'
