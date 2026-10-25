# Spark-X2.6-4B on one M5 Ultra (TensorFold native, the bf16 release)

Serves [Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B), XHToken's 4B dense model with a 1M-token
context, from its bf16 release as published, with TensorFold's native Metal engine, as an OpenAI-compatible server
with reasoning, tool calls, structured output, the prompt cache and several streams. Nothing is converted: the
weights are 7.7 GiB, which fits any Mac the engine runs on, so there was no reason for a lossy quantization.

The engine's `spark2_5` family is in [ashhart/TensorFold#1297](https://github.com/ashhart/TensorFold/pull/1297) (CUDA)
and its Metal side in [#1320](https://github.com/ashhart/TensorFold/pull/1320), stacked on it; until they merge, this recipe builds the pinned commit of the Metal branch
(`spark25-metal` on [BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold/tree/spark25-metal)).

## Quick start

Needs an M5 Mac (the M5's tensor units; M1-M4 are untested), Python 3.11+, git, `hf` and
[Zig 0.17.0](https://ziglang.org/download/).

```bash
git clone https://github.com/BobClawblaw/TensorFold ~/github/TensorFold
hf download XHToken/Spark-X2.6-4B --revision 66e3ae7bee54e0a523d692cb8ef841f35409807b --local-dir ~/models/Spark-X2.6-4B
cp local.example.toml local.toml      # then set zig (and paths, if yours differ)
python3 -m mac build                  # the pinned engine and libtfgrammar, under ~/.cache/spark-x26-4b
python3 -m mac up                     # http://127.0.0.1:8895/v1, model "spark"
python3 -m mac chat "Write a Python function that merges overlapping intervals."
```

`python3 -m mac down|status|logs|command|validate` stop, inspect and check it. To serve the LAN on login:
`python3 -m mac launchd --port 8888` writes `~/Library/LaunchAgents/com.xhtoken.spark-x26-4b.plist` (not loaded), then
`launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.xhtoken.spark-x26-4b.plist`.

## Settings

`recipe.toml` holds the defaults; `local.toml` overrides them for one Mac, and an environment variable of the same
name in upper case overrides both (`PORT=8888 HOST=0.0.0.0 python3 -m mac up`).

| setting | default | |
|---|---|---|
| `context` | 262144 | a stream's window; the release allows 1,048,576 (`context = 1048576`, `parallel = 1`) |
| `parallel` | 4 | streams decoded together, each reply equal to its run alone |
| `max_tokens` | 32768 | a reply's default ceiling; thinking is on by default and counts |
| `temperature`, `top_p`, `top_k` | 1.0, 0.95, off | `generation_config.json`'s sampling for requests that set none |
| `prompt_cache_gib` | 8 | kept prompt states between turns; they give way to streams |
| `load_limit_gb` | 64 | weights, scratch and every stream's caches together (`SPARK_LOAD_LIMIT_GB`; 0: 75% of RAM) |
| `drafts` | true | the lane engine's copy drafts (replies are identical with them off) |

**Memory.** The weights take 7.7 GiB. The 9 full-attention layers keep 36 KiB a token (9.4 GiB a stream at
262,144, 37.7 GiB at 1,048,576); the 27 sliding layers keep a ring of their 512-token window (270 MiB a stream).
Streams and kept prompts share what `load_limit_gb` leaves; a stream that needs more room evicts kept prompts first.

**Thinking** is on by default, as in the chat template: replies reason in `reasoning_content` before the answer in
`content`. A request sends `"chat_template_kwargs": {"enable_thinking": false}` to turn it off.

## What works, and how it was checked

All on one M5 Ultra (256 GB, macOS 27), engine commit 483bf254, checkpoint revision 66e3ae7b. The evidence runs name
the commits as they were when measured (878697c6, 11f15133): the same code before docs-only amends, and the pinned
build's logits, greedy replies and a served reply's logprobs are byte-identical to theirs.

- **Text, greedy and sampled.** Against Hugging Face's own code (transformers 4.57.1) on the release: 1,500
  positions teacher-forced, mean KL 0.00076 from HF in fp32 (HF's own bf16 run: 0.00048), top-1 99.6%; four greedy
  chats (code, prose, Chinese, a 3,547-token document), 96 tokens each, all equal to HF's. One-row steps and
  windows of 7 or 16 rows give the same logits bit for bit after prompts of 64 to 4,100 tokens. Seeded sampling
  repeats.
- **Long context.** A passphrase at 10%, 50% and 90% of a 254,345-token prompt comes back exactly (73-76 s); at 50%
  of a 1,032,662-token prompt on a 1M-context server, in 1,035 s, with 76% of the Mac's memory still free.
- **Tools and structured output.** The harness's structured check (json_object, json_schema, regex, choice,
  grammar) 18/18 and drafted == plain 12/12; its tool cases (auto, required, named, thinking, streamed) 5/5.
- **Prompt cache.** A turn resumed from a kept state (5,499 of 5,624 tokens) equals the same request after a
  restart, and starts in 0.065 s against 0.59 s cold.
- **Concurrency.** A 60-turn soak (3 chats of 20 turns beside a decoding stream) and 12 concurrent replies each equal
  the request alone.
- **Drafts.** The checkpoint has no draft head; the engine's copy drafts give the same replies as `drafts = false`.
- **Vision.** Not applicable: Spark-X2.6-4B is text-only.

The suite's multi-turn check recalled 7 of 8 facts: asked for five short lines, the model gave five and left one fact
out (the same prompt on an 8-bit conversion of the weights recalled 8).

## Measured

Through [recipe-harness](https://github.com/BobClawblaw/recipe-harness) (target `spark-x26-4b-m5`, the qwen Spark
recipe's ruler at suite commit 277c842a1481), greedy, 256-token replies; the raw results are in `evidence/`.

| streams | prose | code | structured | list | first token |
|---:|---:|---:|---:|---:|---:|
| 1 | 105.9 | 106.0 | 105.9 | 105.9 | 0.05-0.07 s |
| 2 | 188.8 | 178.9 | 189.8 | 188.4 | 0.08-0.10 s |
| 4 | 321.5 | 296.2 | 325.5 | 318.0 | 0.13-0.17 s |

Tokens a second, all streams together (`evidence/bf16-speed`, `evidence/bf16-mlx-speed`). One stream reads the 7.7 GiB
of weights each step (9.5 ms), the Mac's memory bandwidth; four streams read them once for all four rows (12.8 ms a
round, 3.0 times one stream's tokens); an 8-bit conversion decodes at 166 tokens/s but costs a mean KL of 0.0021
against the release and parts from its greedy replies within a hundred tokens on two of four chats.

| cold prompt | 2,350 | 9,354 | 37,174 | 148,588 tokens |
|---|---:|---:|---:|---:|
| tokens a second | 9,438 | 10,940 | 9,058 | 5,002 |
| time to first token | 0.25 s | 0.86 s | 4.10 s | 29.7 s |

A fresh server, each prompt starting with words no earlier one did (`evidence/bf16-prefill`; with thinking on, within
1%). A multi-turn chat's third turn, resumed from 5,499 of its 5,624 tokens, starts in 0.065 s against 0.59 s cold
(`evidence/bf16-quick`).

## Performance options

Each keeps the replies' bits; the numbers are this Mac's (`evidence/`).

- **Streams (`parallel`).** One stream decodes at 106 tokens/s, bound by reading the 7.7 GiB of weights each step.
  The weights are read once a round for every stream's row: two streams give 179-190 tokens/s in all, four 296-326
  (12.8 ms a round against 9.6), each reply equal to its run alone; the first token moves from 0.05-0.07 s to
  0.13-0.17 s at four. The engine's row kernel holds up to 32 rows a launch.
- **Weights.** The release's bf16 is exact and is what this recipe serves. The engine also loads the CUDA page's MLX
  affine conversions: the 8-bit one decodes at 166 tokens/s (4.1 GiB a step) for a mean KL of 0.0021 against the
  release and greedy replies that part within a hundred tokens on two of four chats; 4-bit was not measured. Neither
  is needed on a Mac.
- **Context and memory (`context`, `load_limit_gb`).** 36 KiB a token in the nine full layers plus 270 MiB of rings a
  stream: 64 GB holds four 262,144-token streams or one of 1,048,576. Cold prompts run 9.4k-10.9k tokens/s to 37k
  tokens, 5.0k at 149k and about 1.0k averaged over a million (the full layers' attention grows with the prompt).
- **Prompt cache (`prompt_cache_gib`).** A resumed turn (5,499 of 5,624 tokens kept) starts in 0.065 s against 0.59 s
  cold, with the same reply; kept states give way to streams that need the room.
- **Copy drafts (`drafts`).** On by default from the lane core, replies equal to `drafts = false`; they accepted 7-15%
  of their tokens on the harness's prompts and gave no measurable decode gain there. The family has no draft head.
- **Thinking.** On by default (the template's); it lengthens replies, not the cost a token; prompt speed with it on is
  within 1% of off.
- **Not offered.** No int8 or fp8 cache, no fast-math, no reduced-precision sums.

**Quality.** No quality suite was run for this recipe (neither thinking off nor `--plan thinking`).

## Notes

- **Tool calls** use the template's form, which the server parses into OpenAI `tool_calls` and constrains for
  `tool_choice: required` or a named tool.
- Only the M5 Ultra was checked; a smaller-RAM Mac needs a lower `load_limit_gb` and fewer or shorter streams
  (the weights alone need 7.7 GiB).

## License

The recipe is MIT. Spark-X2.6-4B's weights are Apache-2.0 (XHToken); TensorFold is Apache-2.0.
