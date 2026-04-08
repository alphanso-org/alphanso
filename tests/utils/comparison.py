"""
Tolerance-based comparison utilities for ALPHANSO integration tests.

Comparison Strategy:
- Yields (scalar values): Use relative tolerance with absolute floor
- Spectra (arrays): Use element-wise relative tolerance with L2 norm check
- Energy bins: Not compared (configuration-dependent, not computed)
"""

import numpy as np
from typing import Tuple, List


# Default tolerances
YIELD_REL_TOL = 1e-6      # 0.0001% relative tolerance for yields
YIELD_ABS_TOL = 1e-15     # Absolute floor for very small yields
SPECTRUM_REL_TOL = 1e-5   # 0.001% relative tolerance for spectrum elements
SPECTRUM_L2_REL_TOL = 1e-5  # L2 norm relative tolerance


def compare_scalar(
    actual: float,
    expected: float,
    rel_tol: float = YIELD_REL_TOL,
    abs_tol: float = YIELD_ABS_TOL,
    name: str = "value"
) -> Tuple[bool, str]:
    """
    Compare two scalar values with combined relative/absolute tolerance.

    Uses: |actual - expected| <= max(rel_tol * |expected|, abs_tol)

    Returns:
        (passed, message)
    """
    if expected is None and actual is None:
        return True, f"{name}: both None (OK)"
    if expected is None or actual is None:
        return False, f"{name}: None mismatch (expected={expected}, actual={actual})"

    diff = abs(actual - expected)
    threshold = max(rel_tol * abs(expected), abs_tol)

    if diff <= threshold:
        return True, f"{name}: {actual:.6e} vs {expected:.6e} (diff={diff:.2e}, OK)"
    else:
        rel_error = diff / abs(expected) if expected != 0 else float('inf')
        return False, f"{name}: {actual:.6e} vs {expected:.6e} (rel_err={rel_error:.2e}, FAIL)"


def compare_spectrum(
    actual: List[float],
    expected: List[float],
    rel_tol: float = SPECTRUM_REL_TOL,
    l2_rel_tol: float = SPECTRUM_L2_REL_TOL,
    name: str = "spectrum"
) -> Tuple[bool, str]:
    """
    Compare two spectrum arrays using:
    1. Element-wise relative tolerance check
    2. L2 norm relative tolerance check

    Returns:
        (passed, message)
    """
    if expected is None and actual is None:
        return True, f"{name}: both None (OK)"
    if expected is None or actual is None:
        return False, f"{name}: None mismatch"

    actual_arr = np.array(actual)
    expected_arr = np.array(expected)

    if actual_arr.shape != expected_arr.shape:
        return False, f"{name}: shape mismatch ({actual_arr.shape} vs {expected_arr.shape})"

    # Element-wise relative error (handle zeros gracefully)
    denom = np.maximum(np.abs(expected_arr), 1e-30)
    rel_errors = np.abs(actual_arr - expected_arr) / denom
    max_rel_error = np.max(rel_errors)

    # L2 norm check
    diff_l2 = np.linalg.norm(actual_arr - expected_arr)
    expected_l2 = np.linalg.norm(expected_arr)
    l2_rel_error = diff_l2 / expected_l2 if expected_l2 > 0 else diff_l2

    # Both checks must pass
    elem_pass = max_rel_error <= rel_tol
    l2_pass = l2_rel_error <= l2_rel_tol

    passed = elem_pass and l2_pass

    if passed:
        msg = f"{name}: max_rel_err={max_rel_error:.2e}, L2_rel_err={l2_rel_error:.2e} (OK)"
    else:
        msg = (f"{name}: max_rel_err={max_rel_error:.2e} (tol={rel_tol}), "
               f"L2_rel_err={l2_rel_error:.2e} (tol={l2_rel_tol}), FAIL")

    return passed, msg


def compare_yield_list(
    actual: List[float],
    expected: List[float],
    rel_tol: float = YIELD_REL_TOL,
    abs_tol: float = YIELD_ABS_TOL,
    name: str = "yield_list"
) -> Tuple[bool, str]:
    """Compare a list of yield values (e.g., yield_layers)."""
    if expected is None and actual is None:
        return True, f"{name}: both None (OK)"
    if expected is None or actual is None:
        return False, f"{name}: None mismatch"

    if len(actual) != len(expected):
        return False, f"{name}: length mismatch ({len(actual)} vs {len(expected)})"

    all_pass = True
    messages = []
    for i, (a, e) in enumerate(zip(actual, expected)):
        passed, msg = compare_scalar(a, e, rel_tol, abs_tol, f"{name}[{i}]")
        if not passed:
            all_pass = False
        messages.append(msg)

    return all_pass, "; ".join(messages)


def compare_results(actual: dict, expected: dict, calc_type: str) -> Tuple[bool, List[str]]:
    """
    Compare full result dictionaries based on calculation type.

    Args:
        actual: Actual results from Transport.calculate()
        expected: Expected ground truth results
        calc_type: One of 'beam', 'homogeneous', 'interface', 'sandwich'

    Returns:
        (all_passed, list_of_messages)
    """
    messages = []
    all_passed = True

    # Common fields for all calc types
    passed, msg = compare_scalar(actual.get('an_yield'), expected.get('an_yield'), name='an_yield')
    messages.append(msg)
    all_passed &= passed

    passed, msg = compare_spectrum(
        actual.get('an_spectrum'), expected.get('an_spectrum'), name='an_spectrum'
    )
    messages.append(msg)
    all_passed &= passed

    # Homogeneous-specific fields
    if calc_type == 'homogeneous':
        for field in ['sf_yield', 'combined_yield']:
            if field in expected:
                passed, msg = compare_scalar(actual.get(field), expected.get(field), name=field)
                messages.append(msg)
                all_passed &= passed

        for field in ['sf_spectrum', 'combined_spectrum']:
            if field in expected:
                passed, msg = compare_spectrum(
                    actual.get(field), expected.get(field), name=field
                )
                messages.append(msg)
                all_passed &= passed

        for field in ['delayedn_strength']:
            if field in expected:
                passed, msg = compare_scalar(actual.get(field), expected.get(field), name=field)
                messages.append(msg)
                all_passed &= passed

        for field in ['delayedn_spectrum']:
            if field in expected:
                passed, msg = compare_spectrum(actual.get(field), expected.get(field), name=field)
                messages.append(msg)
                all_passed &= passed

    # Sandwich-specific fields
    if calc_type == 'sandwich':
        for field in ['yield_target', 'yield_ab_b', 'yield_bc_b', 'yield_bc_c']:
            if field in expected:
                passed, msg = compare_scalar(actual.get(field), expected.get(field), name=field)
                messages.append(msg)
                all_passed &= passed

        if 'yield_layers' in expected:
            passed, msg = compare_yield_list(
                actual.get('yield_layers'), expected.get('yield_layers'), name='yield_layers'
            )
            messages.append(msg)
            all_passed &= passed

    return all_passed, messages
