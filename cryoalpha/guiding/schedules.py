from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Literal
import jax
import jax.numpy as jnp


def to_progress(
    x: jnp.ndarray,
    input_type: Literal["t", "sigma", "blur"] = "t",
    s_min: float = 0.0004 * 16,
    s_max: float = 160.0 * 16,
) -> jnp.ndarray:
    """Normalize either a time index t in [0,1], a noise level sigma, or the
    guidance blur in Å into a common 'progress' value in [0,1], where
    0=noisiest, 1=cleanest.
    This is the ONLY place the axis decision is made; every schedule
    below just consumes 'progress' and is agnostic to which axis it came from.

    "t": x is the fraction of the trajectory done, step / n_steps (TotalLoss passes it):
    progress = t, so PartSchedule(1.0, 0.2, 0.4, input_type="t") is on for steps 40-79 of 200.
    Note that the noise falls very unevenly with t: almost all shaping happens at t ~0.5-0.78
    (steps ~100-155): noise level at step 0 = 4608 A, 50 = 882, 100 = 101, 120 = 34, 130 = 18, 140 = 9.1,
    150 = 4.3, 160 = 1.0 (200 steps).

    "blur": x is the blur the sampler actually passes (16 Å at the start,
    ~0 Å at the end); progress = 1 - blur/16, so a threshold for a blur b
    is 1 - b/16 (12 Å -> 0.25, 8 Å -> 0.5, 3.5 Å -> 0.78).
    """
    if input_type == "t":
        return jnp.clip(x, 0.0, 1.0)
    elif input_type == "sigma":
        log_max = jnp.log(s_max)
        log_min = jnp.log(s_min)
        current_log = jnp.log(jnp.clip(x, s_min, s_max))
        return jnp.clip((log_max - current_log) / (log_max - log_min), 0.0, 1.0)
    elif input_type == "blur":
        return jnp.clip(1.0 - x / 16.0, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown input_type: {input_type}")


@dataclass
class LinearSchedule:
    start_val: float
    end_val: float
    input_type: Literal["t", "sigma", "blur"] = "t"
    s_min: float = 0.0004 * 16
    s_max: float = 160.0 * 16

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        progress = to_progress(x, self.input_type, self.s_min, self.s_max)
        return self.end_val + progress * (self.start_val - self.end_val)

@dataclass
class ConstantSchedule:
    value: float

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.full_like(x, self.value)

@dataclass
class StepSchedule:
    start_val: float
    end_val: float
    threshold: float
    input_type: Literal["t", "sigma", "blur"] = "t"
    s_min: float = 0.0004 * 16
    s_max: float = 160.0 * 16

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        progress = to_progress(x, self.input_type, self.s_min, self.s_max)
        return jnp.where(progress < self.threshold, self.start_val, self.end_val)

@dataclass
class PartSchedule:
    """`val` while peak_start <= progress < peak_end, 0 elsewhere
    (progress: 0 = noisiest, 1 = cleanest, as in to_progress)."""
    val: float
    peak_start: float
    peak_end: float
    input_type: Literal["t", "sigma", "blur"] = "t"
    s_min: float = 0.0004 * 16
    s_max: float = 160.0 * 16

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        progress = to_progress(x, self.input_type, self.s_min, self.s_max)
        inside = (progress >= self.peak_start) & (progress < self.peak_end)
        return jnp.where(inside, self.val, 0.0)

@dataclass
class SigmoidSchedule:
    start_val: float
    end_val: float
    steepness: float = 10.0
    center: float = 0.5
    input_type: Literal["t", "sigma", "blur"] = "t"
    s_min: float = 0.0004 * 16
    s_max: float = 160.0 * 16

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        progress = to_progress(x, self.input_type, self.s_min, self.s_max)

        sig = 1.0 / (1.0 + jnp.exp(-self.steepness * (progress - self.center)))

        sig_max = 1.0 / (1.0 + jnp.exp(-self.steepness * (1.0 - self.center)))
        sig_min = 1.0 / (1.0 + jnp.exp(-self.steepness * (0.0 - self.center)))
        sig_norm = (sig - sig_min) / (sig_max - sig_min + 1e-8)

        return self.end_val + sig_norm * (self.start_val - self.end_val)

@dataclass
class LogarithmicSchedule:
    start_val: float
    end_val: float
    k: float = 5.0
    input_type: Literal["t", "sigma", "blur"] = "t"
    s_min: float = 0.0004 * 16
    s_max: float = 160.0 * 16

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        progress = to_progress(x, self.input_type, self.s_min, self.s_max)
        warped_progress = jnp.log1p(self.k * progress) / jnp.log1p(self.k)
        return self.end_val + warped_progress * (self.start_val - self.end_val)

@dataclass
class ExponentialSchedule:
    start_val: float
    end_val: float
    k: float = 3.0
    input_type: Literal["t", "sigma", "blur"] = "t"
    s_min: float = 0.0004 * 16
    s_max: float = 160.0 * 16

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        progress = to_progress(x, self.input_type, self.s_min, self.s_max)
        warped_progress = (jnp.exp(self.k * progress) - 1.0) / (jnp.exp(self.k) - 1.0 + 1e-8)
        return self.end_val + warped_progress * (self.start_val - self.end_val)

@dataclass
class ParabolicSchedule:
    start_val: float
    end_val: float
    power: float = 2.0
    input_type: Literal["t", "sigma", "blur"] = "t"
    s_min: float = 0.0004 * 16
    s_max: float = 160.0 * 16

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        progress = to_progress(x, self.input_type, self.s_min, self.s_max)
        warped_progress = progress ** self.power
        return self.end_val + warped_progress * (self.start_val - self.end_val)

@dataclass
class DomeParabolicSchedule:
    peak_val: float
    base_val: float = 0.0
    input_type: Literal["t", "sigma", "blur"] = "t"
    s_min: float = 0.0004 * 16
    s_max: float = 160.0 * 16

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        progress = to_progress(x, self.input_type, self.s_min, self.s_max)
        dome = 4.0 * progress * (1.0 - progress)
        return self.base_val + dome * (self.peak_val - self.base_val)

@dataclass
class ProductSchedule:
    schedules: List[Callable]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        val = jnp.ones_like(x)
        for sched in self.schedules:
            val = val * sched(x)
        return val

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def af3_noise(
        t_val: jnp.ndarray,
        sigma_data: float = 16.0,
        s_min: float = 0.0004,
        s_max: float = 160.0,
        p: float = 7.0
    ) -> jnp.ndarray:
        return sigma_data * (s_max**(1 / p) + t_val * (s_min**(1 / p) - s_max**(1 / p)))**p


    S_MIN = 0.0004 * 16.0  
    S_MAX = 160.0 * 16.0   

    t_steps = jnp.linspace(0.0, 1.0, 300)
    sigmas = af3_noise(t_steps)

    schedules = {
        "Linear (t-driven)": LinearSchedule(2.0, 0.0, input_type="t"),
        "Linear (sigma-driven)": LinearSchedule(2.0, 0.0, input_type="sigma", s_min=S_MIN, s_max=S_MAX),
        "Step (t-driven, off at t=0.5)": StepSchedule(2.0, 0.0, threshold=0.5, input_type="t"),
        "Step (sigma-driven, off at progress=0.5)": StepSchedule(2.0, 0.0, threshold=0.5, input_type="sigma", s_min=S_MIN, s_max=S_MAX),
        "Sigmoid (t-driven, center=0.5)": SigmoidSchedule(2.0, 0.0, steepness=10.0, center=0.5, input_type="t"),
        "Logarithmic (t-driven, k=5)": LogarithmicSchedule(2.0, 0.0, k=5.0, input_type="t"),
        "Exponential (t-driven, k=3)": ExponentialSchedule(2.0, 0.0, k=3.0, input_type="t"),
        "Dome Parabolic (t-driven, peak=2.0)": DomeParabolicSchedule(peak_val=2.0, base_val=0.0, input_type="t"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for name, sched in schedules.items():
        x_input = sigmas if getattr(sched, "input_type", "t") == "sigma" else t_steps
        weights = sched(x_input)

        ax1.plot(t_steps, weights, label=name, linewidth=2)

        ax2.plot(sigmas, weights, label=name, linewidth=2)

    ax1.set_xlabel("Diffusion Time Index $t$ ($0.0 = \\text{Noise } [\\sigma=2560.0]$, $1.0 = \\text{Clean } [\\sigma=0.0064]$)", fontsize=11)
    ax1.set_ylabel("Guidance Weight $w$", fontsize=11)
    ax1.set_title("Schedule Trajectories vs. Time Step ($t$)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xscale("log")
    ax2.invert_xaxis()
    ax2.set_xlabel("Noise Level $\\sigma$ (Log Scale, High Noise $\\rightarrow$ Clean)", fontsize=11)
    ax2.set_ylabel("Guidance Weight $w$", fontsize=11)
    ax2.set_title("Schedule Trajectories vs. Noise Level ($\\sigma$)", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=True)

    plt.tight_layout()
    plt.savefig("af3_schedules_side_by_side.png", dpi=300, bbox_inches="tight")
    print("Dual-axis schedule visualization saved to 'af3_schedules_side_by_side.png'.")
    plt.show()