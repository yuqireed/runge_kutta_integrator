import math
import unittest

from runge_kutta_integrator import integrate, integrate_step


class TestIntegrateStep(unittest.TestCase):
    def test_exponential_decay_one_step(self):
        # dy/dt = -y  =>  y(t) = y0 * exp(-t).  One RK4 step of size h.
        f = lambda t, y: [-y[0]]
        y0 = [1.0]
        h = 0.1
        t_new, y_new = integrate_step(f, 0.0, y0, h)
        self.assertAlmostEqual(t_new, h)
        # RK4 on exp decay: y1 = 1 - h + h^2/2 - h^3/6 + h^4/24.
        expected = 1.0 - h + h * h / 2.0 - h ** 3 / 6.0 + h ** 4 / 24.0
        self.assertAlmostEqual(y_new[0], expected, places=12)

    def test_input_not_mutated(self):
        f = lambda t, y: [y[0] * y[1], -y[0]]
        y0 = [3.0, 4.0]
        _ = integrate_step(f, 0.0, y0, 0.01)
        self.assertEqual(y0, [3.0, 4.0])

    def test_zero_step_raises(self):
        f = lambda t, y: [y[0]]
        with self.assertRaises(ValueError):
            integrate_step(f, 0.0, [1.0], 0.0)

    def test_empty_state_raises(self):
        f = lambda t, y: []
        with self.assertRaises(ValueError):
            integrate_step(f, 0.0, [], 0.1)

    def test_rhs_wrong_length_raises(self):
        f = lambda t, y: [y[0], 99.0]  # returns 2, y has 1.
        with self.assertRaises(TypeError):
            integrate_step(f, 0.0, [1.0], 0.1)

    def test_negative_step_allowed(self):
        # Negative h should integrate backward in time.
        f = lambda t, y: [1.0]  # dy/dt = 1  =>  y = t + C.
        t_new, y_new = integrate_step(f, 5.0, [5.0], -1.0)
        self.assertAlmostEqual(t_new, 4.0)
        self.assertAlmostEqual(y_new[0], 4.0, places=12)

    def test_two_equation_system(self):
        # Simple harmonic oscillator: y0' = y1, y1' = -y0.
        # Analytic: y0 = cos(t), y1 = -sin(t).
        f = lambda t, y: [y[1], -y[0]]
        t_new, y_new = integrate_step(f, 0.0, [1.0, 0.0], 0.1)
        self.assertAlmostEqual(y_new[0], math.cos(0.1), places=6)
        self.assertAlmostEqual(y_new[1], -math.sin(0.1), places=6)

    def test_rhs_receives_correct_time(self):
        seen_times = []

        def f(t, y):
            seen_times.append(t)
            return [y[0]]

        integrate_step(f, 2.0, [0.0], 0.4)
        self.assertEqual(len(seen_times), 4)
        self.assertAlmostEqual(seen_times[0], 2.0)
        self.assertAlmostEqual(seen_times[1], 2.2)
        self.assertAlmostEqual(seen_times[2], 2.2)
        self.assertAlmostEqual(seen_times[3], 2.4)


class TestIntegrate(unittest.TestCase):
    def test_zero_steps_returns_initial_only(self):
        f = lambda t, y: [y[0]]
        times, states = integrate(f, 1.0, [7.0], 0.1, 0)
        self.assertEqual(times, [1.0])
        self.assertEqual(states, [[7.0]])

    def test_negative_steps_raises(self):
        f = lambda t, y: [y[0]]
        with self.assertRaises(ValueError):
            integrate(f, 0.0, [1.0], 0.1, -1)

    def test_initial_state_copied(self):
        f = lambda t, y: [y[0]]
        y0 = [5.0]
        _ = integrate(f, 0.0, y0, 0.1, 3)
        self.assertEqual(y0, [5.0])

    def test_output_lengths(self):
        f = lambda t, y: [y[0]]
        times, states = integrate(f, 0.0, [1.0], 0.1, 5)
        self.assertEqual(len(times), 6)
        self.assertEqual(len(states), 6)

    def test_linear_growth_exact(self):
        # dy/dt = 1 is integrated exactly by RK4.
        f = lambda t, y: [1.0]
        times, states = integrate(f, 0.0, [0.0], 0.25, 4)
        for i, t in enumerate(times):
            self.assertAlmostEqual(states[i][0], t, places=12)

    def test_exponential_decay_convergence(self):
        # Halving h should cut global error by ~16x for RK4.
        f = lambda t, y: [-y[0]]
        t_end = 1.0
        exact = math.exp(-t_end)

        def error_for(steps):
            h = t_end / steps
            _, states = integrate(f, 0.0, [1.0], h, steps)
            return abs(states[-1][0] - exact)

        e_coarse = error_for(10)
        e_fine = error_for(20)
        ratio = e_coarse / e_fine
        # Expect ratio near 16; allow [10, 26] for float noise.
        self.assertGreater(ratio, 10.0)
        self.assertLess(ratio, 26.0)

    def test_returns_fresh_lists(self):
        # Each state in the output must be an independent list.
        f = lambda t, y: [y[0]]
        _, states = integrate(f, 0.0, [1.0], 0.1, 2)
        states[0][0] = 999.0
        self.assertNotEqual(states[1][0], 999.0)


if __name__ == "__main__":
    unittest.main()
