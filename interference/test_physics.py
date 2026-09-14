import unittest

import numpy as np

from physics import (
    complex_field,
    field_views,
    m_range_for_levels,
    path_difference_level,
    source_envelope,
    wavefront_radii,
)


class PhysicsModelTests(unittest.TestCase):
    def test_equal_in_phase_sources_add_constructively(self):
        distance = np.array([[1.0]])
        field = complex_field(distance, distance, 1.0, 1.0, 1.0, 0.0, 0.0)
        amplitude, intensity, phase = field_views(field)
        self.assertAlmostEqual(float(amplitude[0, 0]), 2.0)
        self.assertAlmostEqual(float(intensity[0, 0]), 4.0)
        self.assertAlmostEqual(float(phase[0, 0]), 0.0, places=12)

    def test_equal_opposite_phase_sources_cancel(self):
        distance = np.array([[1.0]])
        field = complex_field(distance, distance, 1.0, 1.0, 1.0, 0.0, np.pi)
        amplitude, intensity, _ = field_views(field)
        self.assertLess(float(amplitude[0, 0]), 1e-12)
        self.assertLess(float(intensity[0, 0]), 1e-24)

    def test_unequal_opposite_sources_leave_residual_amplitude(self):
        distance = np.array([[1.0]])
        field = complex_field(distance, distance, 1.0, 2.0, 1.0, 0.0, np.pi)
        amplitude, _, _ = field_views(field)
        self.assertAlmostEqual(float(amplitude[0, 0]), 1.0)

    def test_phase_shift_moves_constructive_path_difference(self):
        self.assertAlmostEqual(path_difference_level(0, 0.5, np.pi, 0.0), -0.25)
        self.assertEqual(m_range_for_levels(-1.0, 1.0, 0.5, 0.0, 0.0), [-2, -1, 0, 1, 2])

    def test_attenuation_is_finite_at_source(self):
        envelope = source_envelope(np.array([0.0, 1.0]), 1.0, 0.5, True)
        self.assertTrue(np.all(np.isfinite(envelope)))
        self.assertGreater(envelope[0], envelope[1])

    def test_cos_wavefronts_include_expected_crest_and_trough(self):
        self.assertEqual(wavefront_radii(1.0, 0.0, 2.0), [1.0, 2.0])
        self.assertEqual(wavefront_radii(1.0, 0.0, 2.0, trough=True), [0.5, 1.5])


if __name__ == "__main__":
    unittest.main()
