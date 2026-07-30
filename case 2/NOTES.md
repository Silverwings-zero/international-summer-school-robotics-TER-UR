# Case 2: working notes

## Robot dynamics & sim-to-real (from UR "AI for developers" deck 2026-03, slide 26)
Diagram: `reference/ur_dynamics.png`. Slide title: "Dynamic Access for Advanced
Control and Sim-to-Real".

Rigid-body equation of motion:

    M(q)·q̈ + C(q,q̇)·q̇ + F(q̇) + G(q) = τ

| Term    | Meaning |
|---------|---------|
| M(q)·q̈ | Inertia (mass matrix) × joint acceleration |
| C(q,q̇)·q̇ | Coriolis + centrifugal terms |
| F(q̇)   | Joint friction & damping |
| G(q)    | Gravity (compensation) |
| τ       | Torque command |

**URScript exposes these dynamics functions:** mass matrix, Coriolis/centrifugal
terms, Jacobians, and their time derivatives.

**Why it matters for Case 2:** these functions enable model-based control and let
you *align the simulator with the real robot's physics*, directly attacking the
reality gap the case is about. Framing to use: the gap model the students learn is
approximating what these dynamics terms (plus unmodeled friction/backlash/thermal
effects) do on the real UR10. RL then optimizes speed-vs-vibration on top.

**Possible slide use (Case 2 theory):** a "what is the reality gap" slide can show
this equation as *the physics the simulator idealizes*, then contrast: URSim assumes
perfect tracking (actuals = targets), the real robot deviates because F(q̇), payload,
and joint config are imperfectly modeled. Bridge → learn the gap → RL.

TODO when building Case 2: decide whether to expose dynamics via URScript in the
data-recording script, or keep the gap purely data-driven (RTDE target-vs-actual).
