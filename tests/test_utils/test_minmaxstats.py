"""Unit tests for MinMaxStats class used in MuZero MCTS."""

import torch
import pytest

from sheeprl.algos.muzero.utils import MinMaxStats


class TestMinMaxStatsInitialization:
    """Test proper initialization of MinMaxStats."""

    def test_initial_values(self):
        """Test that min/max are initialized to allow first update."""
        stats = MinMaxStats()
        assert stats.maximum == -float("inf"), "maximum should start at -inf"
        assert stats.minimum == float("inf"), "minimum should start at +inf"

    def test_normalize_before_update_returns_unchanged(self):
        """Test that normalize returns value unchanged before any updates."""
        stats = MinMaxStats()
        value = torch.tensor([1.0, 2.0, 3.0])
        result = stats.normalize(value)
        torch.testing.assert_close(result, value)


class TestMinMaxStatsUpdate:
    """Test the update mechanism."""

    def test_update_single_value(self):
        """Test updating with a single value."""
        stats = MinMaxStats()
        stats.update(5.0)
        assert stats.maximum == 5.0
        assert stats.minimum == 5.0

    def test_update_multiple_values(self):
        """Test updating with multiple values."""
        stats = MinMaxStats()
        stats.update(5.0)
        stats.update(10.0)
        stats.update(2.0)
        assert stats.maximum == 10.0
        assert stats.minimum == 2.0

    def test_update_negative_values(self):
        """Test updating with negative values."""
        stats = MinMaxStats()
        stats.update(-5.0)
        stats.update(-10.0)
        stats.update(-2.0)
        assert stats.maximum == -2.0
        assert stats.minimum == -10.0

    def test_update_mixed_positive_negative(self):
        """Test updating with mix of positive and negative values."""
        stats = MinMaxStats()
        stats.update(10.0)
        stats.update(-5.0)
        stats.update(0.0)
        assert stats.maximum == 10.0
        assert stats.minimum == -5.0

    def test_update_same_value_multiple_times(self):
        """Test that updating with same value doesn't break."""
        stats = MinMaxStats()
        stats.update(5.0)
        stats.update(5.0)
        stats.update(5.0)
        assert stats.maximum == 5.0
        assert stats.minimum == 5.0


class TestMinMaxStatsNormalize:
    """Test the normalization mechanism."""

    def test_normalize_single_value(self):
        """Test normalization with a single value (min == max)."""
        stats = MinMaxStats()
        stats.update(5.0)
        
        # When min == max, normalization should return unchanged
        value = torch.tensor([5.0])
        result = stats.normalize(value)
        # Condition fails: maximum > minimum is False when they're equal
        torch.testing.assert_close(result, value)

    def test_normalize_in_range(self):
        """Test normalizing values within the observed range."""
        stats = MinMaxStats()
        stats.update(0.0)
        stats.update(10.0)
        
        # Value at min -> 0.0
        value_min = torch.tensor([0.0])
        result_min = stats.normalize(value_min)
        assert result_min.item() == pytest.approx(0.0)
        
        # Value at max -> 1.0
        value_max = torch.tensor([10.0])
        result_max = stats.normalize(value_max)
        assert result_max.item() == pytest.approx(1.0)
        
        # Value in middle -> 0.5
        value_mid = torch.tensor([5.0])
        result_mid = stats.normalize(value_mid)
        assert result_mid.item() == pytest.approx(0.5)

    def test_normalize_batch_tensor(self):
        """Test normalizing a batch of values."""
        stats = MinMaxStats()
        stats.update(0.0)
        stats.update(10.0)
        
        values = torch.tensor([0.0, 2.5, 5.0, 7.5, 10.0])
        expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        
        result = stats.normalize(values)
        torch.testing.assert_close(result, expected)

    def test_normalize_outside_range(self):
        """Test normalizing values outside the observed range."""
        stats = MinMaxStats()
        stats.update(0.0)
        stats.update(10.0)
        
        # Value below min
        value_below = torch.tensor([-5.0])
        result_below = stats.normalize(value_below)
        assert result_below.item() == pytest.approx(-0.5)
        
        # Value above max
        value_above = torch.tensor([15.0])
        result_above = stats.normalize(value_above)
        assert result_above.item() == pytest.approx(1.5)

    def test_normalize_negative_range(self):
        """Test normalizing with negative value range."""
        stats = MinMaxStats()
        stats.update(-10.0)
        stats.update(-5.0)
        
        value = torch.tensor([-7.5])
        result = stats.normalize(value)
        # (-7.5 - (-10)) / ((-5) - (-10)) = 2.5 / 5 = 0.5
        assert result.item() == pytest.approx(0.5)

    def test_normalize_multidimensional_tensor(self):
        """Test normalizing multi-dimensional tensors."""
        stats = MinMaxStats()
        stats.update(0.0)
        stats.update(10.0)
        
        values = torch.tensor([[0.0, 5.0], [10.0, 2.5]])
        expected = torch.tensor([[0.0, 0.5], [1.0, 0.25]])
        
        result = stats.normalize(values)
        torch.testing.assert_close(result, expected)


class TestMinMaxStatsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_update_with_zero(self):
        """Test updating with zero."""
        stats = MinMaxStats()
        stats.update(0.0)
        assert stats.maximum == 0.0
        assert stats.minimum == 0.0

    def test_update_very_large_values(self):
        """Test with very large floating point values."""
        stats = MinMaxStats()
        stats.update(1e10)
        stats.update(1e-10)
        assert stats.maximum == 1e10
        assert stats.minimum == 1e-10

    def test_normalize_preserves_device(self):
        """Test that normalization preserves tensor device."""
        stats = MinMaxStats()
        stats.update(0.0)
        stats.update(10.0)
        
        # CPU tensor
        value_cpu = torch.tensor([5.0])
        result_cpu = stats.normalize(value_cpu)
        assert result_cpu.device == value_cpu.device

    def test_normalize_preserves_dtype(self):
        """Test that normalization preserves tensor dtype."""
        stats = MinMaxStats()
        stats.update(0.0)
        stats.update(10.0)
        
        # Float32
        value_f32 = torch.tensor([5.0], dtype=torch.float32)
        result_f32 = stats.normalize(value_f32)
        assert result_f32.dtype == torch.float32
        
        # Float64
        value_f64 = torch.tensor([5.0], dtype=torch.float64)
        result_f64 = stats.normalize(value_f64)
        assert result_f64.dtype == torch.float64


class TestMinMaxStatsMCTSUsage:
    """Test MinMaxStats behavior in typical MCTS scenarios."""

    def test_mcts_value_normalization_scenario(self):
        """Simulate typical MCTS usage where values are discovered incrementally."""
        stats = MinMaxStats()
        
        # Simulate MCTS discovering values during search
        mcts_values = [0.5, 0.8, 0.3, 0.9, 0.2, 1.0, 0.1]
        
        for value in mcts_values:
            stats.update(value)
        
        # Check final min/max
        assert stats.minimum == 0.1
        assert stats.maximum == 1.0
        
        # Normalize the discovered values
        values_tensor = torch.tensor(mcts_values)
        normalized = stats.normalize(values_tensor)
        
        # All values should be in [0, 1] after normalization
        assert normalized.min().item() == pytest.approx(0.0)
        assert normalized.max().item() == pytest.approx(1.0)
        assert torch.all(normalized >= 0.0)
        assert torch.all(normalized <= 1.0)

    def test_sequential_update_and_normalize(self):
        """Test interleaved update and normalize operations."""
        stats = MinMaxStats()
        
        # First value
        stats.update(5.0)
        result1 = stats.normalize(torch.tensor([5.0]))
        torch.testing.assert_close(result1, torch.tensor([5.0]))  # No normalization yet
        
        # Second value
        stats.update(10.0)
        result2 = stats.normalize(torch.tensor([5.0, 10.0]))
        torch.testing.assert_close(result2, torch.tensor([0.0, 1.0]))
        
        # Third value
        stats.update(0.0)
        result3 = stats.normalize(torch.tensor([0.0, 5.0, 10.0]))
        torch.testing.assert_close(result3, torch.tensor([0.0, 0.5, 1.0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
