# LLM Client (free)

The cases need an LLM that can call MCP tools. You do not have to pay for one.
This folder sets one up: a model you can chat with, call over an OpenAI-compatible
API, and connect MCP servers to. What to do with it is described in each case
folder.

Two free ways. Pick either.

## Self-hosted (local, on your laptop)
Runs a model on your own machine with the Bionic app. Zero cost, no account,
works offline. Slower than the cloud. Best if your laptop can run a mid-size
model.

Guide: [self-hosted.md](self-hosted.md)

## Cloud-hosted (free hosted endpoint)
Uses a model someone else runs (NVIDIA NIM and others), over an OpenAI-compatible
API. Faster, no big download, but needs a free account and an API key. Best for a
weak laptop or a stronger model.

Guide: [cloud-hosted.md](cloud-hosted.md)

## Which do I use?
- Strong laptop, want it fully offline and free: self-hosted.
- Weak laptop, or want a bigger and faster model: cloud-hosted.

Already have Claude Code (paid)? You can skip both. Each case folder has a short
"Connect Claude Code" section.

## Model choice (applies to both)
Do not rely on one specific model's tool-use behaviour. Different models handle
function calling differently. Keep prompts and tool-call handling model-agnostic
so any backend works. If a small local model refuses or malforms a tool call, try
a larger one, or the cloud path.
