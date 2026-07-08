"""
Integration tests for ALPHANSO homogeneous calculations.

These tests compare current calculation results against validated
ground truth to ensure no regressions have occurred.
"""

import pytest
from alphanso.transport import Transport
from tests.integration.configs import HOMOGENEOUS_CONFIGS
from tests.utils.comparison import compare_results


class TestHomogeneousCalculations:
    """Integration tests for homogeneous problem calculations."""

    @pytest.mark.integration
    @pytest.mark.parametrize("config", HOMOGENEOUS_CONFIGS, ids=lambda c: c['name'])
    def test_homogeneous_calculation(self, config, homogeneous_ground_truth):
        """
        Test that homogeneous calculations match ground truth within tolerance.

        Compares:
        - an_yield: (alpha,n) neutron yield (n/s/g)
        - sf_yield: Spontaneous fission yield (n/s/g)
        - combined_yield: Total neutron yield (n/s/g)
        - an_spectrum: Normalized (alpha,n) spectrum
        - sf_spectrum: Normalized SF spectrum
        - combined_spectrum: Combined normalized spectrum
        """
        name = config['name']

        # Skip if no ground truth for this config
        if name not in homogeneous_ground_truth:
            pytest.skip(f"No ground truth for {name}")

        expected = homogeneous_ground_truth[name]

        # Check for error in ground truth
        if 'error' in expected:
            pytest.skip(f"Ground truth recorded error: {expected['error']}")

        # Run calculation
        actual = Transport.calculate(config)

        # Compare results
        passed, messages = compare_results(actual, expected, 'homogeneous')

        # Build assertion message
        full_message = f"\n{name} comparison results:\n" + "\n".join(f"  {m}" for m in messages)

        assert passed, full_message

        assert 'delayedn_strength' in actual
        assert isinstance(actual['delayedn_strength'], float)
        assert actual['delayedn_strength'] >= 0.0
        assert 'delayedn_spectrum' in actual
        assert len(actual['delayedn_spectrum']) == len(actual['neutron_energy_bins']) - 1
