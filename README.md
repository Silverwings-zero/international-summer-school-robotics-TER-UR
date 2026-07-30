# UR Industry Cases, Robotics Summer School 2026

Three hands-on cases on **agentic and learning-based robotics**, built around a
Universal Robots UR10 you run in simulation on your own laptop. Each case is a
small, readable Python project you extend from Bronze to Diamond. Everything here
is free and self-contained: one LLM client, one simulator, three case folders.

The cases are **explorative**. What ships in the repo is a working baseline that
is only *guiding*, not a spec. Read it, then change anything that makes your
solution better. The goal is to present something of value at the end.

## The three cases

**Case 1, MCP Server for Robot Tools** (`case 1/`)
Expose the robot as tools an LLM can call, and program it by conversation.
*How should a robot's capabilities be exposed as LLM-callable tools (which to
expose, and how to describe them) so an agent can operate the robot reliably from
natural language?*

**Case 2, Reality Gap and Motion Optimization** (`case 2/`)
Model how the real robot deviates from the plan, then use reinforcement learning
to move faster with less vibration.
*Can robot motion be robustly optimized from distilled information, a gap model
learned from real-robot data, to move faster with less vibration, without ever
training on the hardware?*

**Case 3, Autonomous Agent** (`case 3/`)
An LLM that plans, calls robot tools, checks the result, and self-corrects.
*Can an LLM orchestrator, grounded by a state serializer and an evaluator,
replace the human's plan/act/check loop and operate the robot autonomously from a
plain-language goal?*

## Get started

Two shared pieces set up once, then pick a case.

1. **A free LLM client** (needed for Cases 1 and 3). Bring your own model, either
   path is free. See [`llm-client/`](llm-client/).
   - Self-hosted (Bionic, local): a model on your laptop. Zero cost, offline, no
     account.
   - Cloud-hosted (NVIDIA NIM, free tier): a hosted OpenAI-compatible endpoint.
     Faster, tiny download, needs a free API key.
   - Already have Claude Code (paid)? You can skip this.

2. **The simulator** (the robot for all three cases). One container:
   ```bash
   cd "simulation environment"
   docker compose up -d      # ~40s first boot
   ```
   Then open http://localhost and power the robot on (release the brakes until it
   reads RUNNING). Exposes port 80 (web UI), 30001 (motion), 30004 (RTDE state).
   See [`simulation environment/`](simulation%20environment/).

3. **Pick a case** and follow its README. Each has setup, run commands, and the
   Bronze to Diamond tiers.

## Repository layout

| Path | What |
|------|------|
| `case 1/` | MCP server: `server.py` (worked tool), `ur_client.py` (robot seam), `test_server.py` |
| `case 2/` | Reality-gap RL: `dataset.py` / `record.py`, `gap_model.py`, `gap_env.py`, `metrics.py` |
| `case 3/` | Autonomous agent: `agent.py` (the loop), `prompts.py`, `serializer.py`, `evaluator.py` |
| `llm-client/` | The free LLM client setup (self-hosted and cloud-hosted guides) |
| `simulation environment/` | The PolyScope X (URSim) simulator and docker-compose |
| `docs/` | The presentation slides and the tiers document |

## Tiers

Every case runs Bronze to Diamond: increasing scope and difficulty, you pick how
far you take it. The full ladder per case is in the tiers document under
[`docs/`](docs/), and summarized in each case's README.

## Requirements

- Docker (for the simulator)
- Python 3.10+ (each case folder has its own `requirements.txt`)
- An LLM endpoint for Cases 1 and 3 (see `llm-client/`)

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE).
Copyright 2026 Universal Robots A/S. Use, modify, and redistribute freely; the
work is provided as-is, without warranty.

## Contact

Emil Stubbe Kolvig-Raun, Sr. AI Engineering (PhD), Teradyne Robotics / Universal
Robots. Contact: eskr@universal-robots.com
