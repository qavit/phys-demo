"""Pure numerical model for the two-source interference Streamlit app."""

from __future__ import annotations

import numpy as np


def m_range_for_levels(
    d_min: float,
    d_max: float,
    wavelength: float,
    phase_difference: float,
    offset: float,
) -> list[int]:
    """Return integers m whose path-difference level is visible."""
    shift = phase_difference / (2.0 * np.pi)
    lower = int(np.ceil(d_min / wavelength - offset + shift))
    upper = int(np.floor(d_max / wavelength - offset + shift))
    if upper < lower:
        return []
    return list(range(lower, upper + 1))


def path_difference_level(
    m: int,
    wavelength: float,
    phase_difference: float,
    offset: float,
) -> float:
    """Path difference for a constructive (0) or destructive (0.5) level."""
    shift = phase_difference / (2.0 * np.pi)
    return (m + offset - shift) * wavelength


def source_envelope(
    distance: np.ndarray,
    amplitude: float,
    wavelength: float,
    attenuate: bool,
) -> np.ndarray:
    """Return source amplitude, optionally using a regularized 2D far-field model."""
    if not attenuate:
        return np.full_like(distance, amplitude, dtype=float)

    # An ideal cylindrical wave scales as 1/sqrt(r) only away from the source.
    # A finite core avoids an unphysical singular pixel at r=0.
    core_radius = max(0.08 * wavelength, 0.02)
    return amplitude / np.sqrt(np.maximum(distance, core_radius))


def complex_field(
    r1: np.ndarray,
    r2: np.ndarray,
    wavelength: float,
    amplitude1: float,
    amplitude2: float,
    phase1: float,
    phase2: float,
    attenuate: bool = False,
) -> np.ndarray:
    """Complex phasor C whose real time evolution is Re(C exp(-i omega t))."""
    wave_number = 2.0 * np.pi / wavelength
    envelope1 = source_envelope(r1, amplitude1, wavelength, attenuate)
    envelope2 = source_envelope(r2, amplitude2, wavelength, attenuate)
    return (
        envelope1 * np.exp(1j * (wave_number * r1 + phase1))
        + envelope2 * np.exp(1j * (wave_number * r2 + phase2))
    )


def field_views(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return resultant amplitude, relative time-averaged intensity and phase."""
    amplitude = np.abs(field)
    intensity = amplitude**2
    phase = np.angle(field)
    return amplitude, intensity, phase


def wavefront_radii(
    wavelength: float,
    phase: float,
    maximum_radius: float,
    trough: bool = False,
) -> list[float]:
    """Radii of crests or troughs for cos(kr + phase)."""
    offset = 0.5 if trough else 0.0
    phase_shift = phase / (2.0 * np.pi)
    n_min = int(np.ceil(phase_shift - offset))
    n_max = int(np.floor(maximum_radius / wavelength + phase_shift - offset))
    radii = [
        (n + offset - phase_shift) * wavelength
        for n in range(n_min, n_max + 1)
    ]
    return [radius for radius in radii if radius > 0]
