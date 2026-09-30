"""Classical fourth-order Runge-Kutta integrator for systems of ODEs.

The integrator operates on plain Python callables so users can plug in any
right-hand-side function f(t, y) -> list[float] without subclassing or wrapping.
State is represented as a Python list of floats; this keeps the library
dependency-free and makes the results trivially inspectable.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple


# Type alias for the right-hand side: f(t, y) -> y'.
# We accept a Sequence for the input state but always return a List so the
# caller never mutates our internal buffers by accident.
RHS = Callable[[float, Sequence[float]], List[float]]


def _add_scaled(accumulator: List[float], vec: Sequence[float], scale: float) -> None:
    """In-place accumulator += scale * vec.

    Using an explicit loop instead of a list comprehension keeps allocations
    down: the four RK stages each add into the same k-buffer, and doing this
    with comprehensions would create two temporary lists per stage. For tight
    integration loops this is the hot path.
    """
    for i in range(len(accumulator)):
        accumulator[i] += scale * vec[i]


def integrate_step(
    f: RHS, t: float, y: Sequence[float], h: float
) -> Tuple[float, List[float]]:
    """Advance one classical-RK4 step of size *h*.

    Returns the new time ``t + h`` and the new state as a fresh list. The
    input *y* is never mutated, so callers can keep it for comparison or
    rollback.

    Raises:
        ValueError: if *h* is zero (a zero step makes no progress and is
            almost always a caller bug — silently returning the input hides
            mistakes, so we refuse).
        ValueError: if *y* is empty (no equations to integrate).
        TypeError: if *f* returns something that is not a sequence of numbers
            or has the wrong length — we catch this early rather than letting
            it surface as a confusing IndexError three stages in.
    """
    if h == 0.0:
        raise ValueError("step size h must be non-zero")
    n = len(y)
    if n == 0:
        raise ValueError("state vector y must not be empty")

    # Stage 1.
    k1 = f(t, y)
    if len(k1) != n:
        raise TypeError(
            f"f returned {len(k1)} values; expected {n} (length of y)"
        )

    # Intermediate state for stage 2: y + h/2 * k1.
    y_mid = [y[i] + 0.5 * h * k1[i] for i in range(n)]
    k2 = f(t + 0.5 * h, y_mid)
    if len(k2) != n:
        raise TypeError(
            f"f returned {len(k2)} values; expected {n} (length of y)"
        )

    # Stage 3 reuses the same midpoint offset but with k2.
    y_mid = [y[i] + 0.5 * h * k2[i] for i in range(n)]
    k3 = f(t + 0.5 * h, y_mid)
    if len(k3) != n:
        raise TypeError(
            f"f returned {len(k3)} values; expected {n} (length of y)"
        )

    # Stage 4 uses a full-step offset with k3.
    y_end = [y[i] + h * k3[i] for i in range(n)]
    k4 = f(t + h, y_end)
    if len(k4) != n:
        raise TypeError(
            f"f returned {len(k4)} values; expected {n} (length of y)"
        )

    # Combine: y_new = y + (h/6)(k1 + 2k2 + 2k3 + k4).
    # We accumulate into k1 in place to avoid one more allocation.
    result = k1  # k1 is already a fresh list owned by us.
    _add_scaled(result, k2, 2.0)
    _add_scaled(result, k3, 2.0)
    _add_scaled(result, k4, 1.0)
    for i in range(n):
        result[i] = y[i] + (h / 6.0) * result[i]

    return t + h, result


def integrate(
    f: RHS,
    t0: float,
    y0: Sequence[float],
    h: float,
    steps: int,
) -> Tuple[List[float], List[List[float]]]:
    """Integrate the system y' = f(t, y) for a fixed number of *steps*.

    Returns ``(times, states)`` where ``times`` has length ``steps + 1``
    (including the initial time) and ``states[i]`` is the state at
    ``times[i]``. The initial state ``y0`` is copied so the caller's list is
    never mutated.

    A fixed-step integrator is chosen deliberately: it is simple, has no
    internal control flow that depends on the physics, and its error is
    predictable (O(h^4) per step). It is the right tool when the step size is
    dictated externally — e.g. by a coupled simulation — rather than by an
    accuracy target. For problems where the dynamics change stiffness across
    the domain, an adaptive method would be a better fit; this library does
    not try to be that.

    Raises:
        ValueError: if *steps* is negative.
        ValueError: if *h* is zero.
        ValueError: if *y0* is empty.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if h == 0.0:
        raise ValueError("step size h must be non-zero")
    n = len(y0)
    if n == 0:
        raise ValueError("initial state y0 must not be empty")

    times: List[float] = [t0]
    states: List[List[float]] = [list(y0)]  # copy so caller's list is safe.

    t = t0
    y = list(y0)
    for _ in range(steps):
        t, y = integrate_step(f, t, y, h)
        times.append(t)
        states.append(y)

    return times, states
