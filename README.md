# Runge Kutta Integrator

A small, dependency-free Python library implementing the classical
fourth-order Runge-Kutta method (RK4) for systems of ordinary differential
equations. It operates on plain Python callables and plain lists — no NumPy,
no classes to subclass, no wrappers.

## Usage

```python
from runge_kutta_integrator import integrate, integrate_step

# Simple harmonic oscillator: y0' = y1, y1' = -y0.
def f(t, y):
    return [y[1], -y[0]]

times, states = integrate(f, t0=0.0, y0=[1.0, 0.0], h=0.01, steps=1000)
# times[i] is the i-th sample time, states[i] is the state at that time.

# Or take a single step:
t1, y1 = integrate_step(f, 0.0, [1.0, 0.0], 0.01)
```

## Exports

- `integrate(f, t0, y0, h, steps)` — integrate for a fixed number of steps.
  Returns `(times, states)` where both lists have length `steps + 1`.
- `integrate_step(f, t, y, h)` — advance one RK4 step. Returns
  `(t + h, y_new)`.

`f` is a callable `f(t, y) -> list[float]` where `y` is a sequence of floats.

## Why this exists

The problem is integrating a system of ODEs with a known, fixed step size —
typically because the step is dictated by a coupled simulation or a data
sampling rate, not by an accuracy target. In that situation an adaptive
integrator adds control flow and heuristics that you do not need and cannot
inspect. Classical RK4 has a single code path, a known O(h⁴) local error,
and no internal state to get out of sync.

The trade-off is that this library will not choose a step size for you and
will not detect stiffness. If your dynamics span widely different time scales
you need a different tool.

## Edge cases

- A step size of exactly zero raises `ValueError`. Silently returning the
  input would hide caller bugs.
- An empty state vector raises `ValueError`.
- If the right-hand side returns a list whose length differs from the state,
  `integrate_step` raises `TypeError` on the first stage rather than failing
  later with a confusing `IndexError`.
- Negative step sizes are allowed and integrate backward in time. This is
  sometimes needed for terminal-value problems; refusing it would be an
  arbitrary restriction.
- The input state is never mutated. `integrate` copies `y0` before stepping,
  and `integrate_step` returns a fresh list.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

