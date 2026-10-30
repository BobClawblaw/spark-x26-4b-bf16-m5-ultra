# Spark-X2.6-4B on one M5 Ultra (TensorFold native, the bf16 release)

Serves [Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B), XHToken's 4B dense model with a 1M-token
context, from its bf16 release as published, with TensorFold's native Metal engine, as an OpenAI-compatible server
with reasoning, tool calls, structured output, the prompt cache and several streams. Nothing is converted: the
weights are 7.7 GiB, which fits any Mac the engine runs on, so there was no reason for a lossy quantization.

The engine's `spark2_5` family is on TensorFold main: its CUDA side came with
[ashhart/TensorFold#1297](https://github.com/ashhart/TensorFold/pull/1297) and its Metal side with
[#1320](https://github.com/ashhart/TensorFold/pull/1320). This recipe builds a pinned commit of
[ashhart/TensorFold](https://github.com/ashhart/TensorFold)'s main.

## Quick start

Needs an M5 Mac (the M5's tensor units; M1-M4 are untested), Python 3.11+, git, `hf` and
[Zig 0.17.0](https://ziglang.org/download/).

```bash
git clone https://github.com/ashhart/TensorFold ~/github/TensorFold
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
| `variant` | bf16 | `bf16`: the release, exact; `8bit`: its MLX affine conversion ([8-bit variant](#8-bit-variant)) |
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

All on one M5 Ultra (256 GB, macOS 27), checkpoint revision 66e3ae7b. The checks in this section were made on the
Metal branch's commit cc840195 (and its earlier forms 878697c6 and 11f15133, byte-identical in logits, greedy replies and
logprobs) before #1320 merged. The pinned main commit abf82258 gives byte-identical replies to cc840195: ten greedy
chats, thinking on and off, top-2 logprobs at every token. The harness runs under Measured were repeated on abf82258.

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
| 1 | 105.6 | 104.7 | 104.6 | 105.0 | 0.05-0.07 s |
| 2 | 186.8 | 176.4 | 186.9 | 183.7 | 0.08-0.10 s |
| 4 | 317.9 | 293.6 | 320.5 | 313.5 | 0.13-0.17 s |

Tokens a second, all streams together (`evidence/bf16-mlx-speed`; `evidence/bf16-speed` repeats one and two streams).
One stream reads the 7.7 GiB of weights each step (9.6 ms), the Mac's memory bandwidth; four streams read them once for
all four rows (12.9 ms a round, 3.0 times one stream's tokens); an 8-bit conversion decodes at 166 tokens/s but costs a mean KL of 0.0021
against the release and parts from its greedy replies within a hundred tokens on two of four chats.

| cold prompt | 2,360 | 9,318 | 37,295 | 148,520 tokens |
|---|---:|---:|---:|---:|
| tokens a second | 9,042 | 10,723 | 8,946 | 4,955 |
| time to first token | 0.26 s | 0.87 s | 4.17 s | 30.0 s |

A fresh server, each prompt starting with words no earlier one did (`evidence/bf16-prefill`; with thinking on, within
1% from 9k tokens up, and from -1% to +9% at 2.4k, where a quarter-second prompt varies run to run). A multi-turn chat's
third turn, resumed from 5,499 of its 5,624 tokens, starts in 0.066 s against 0.59 s cold (`evidence/bf16-quick`). The
long-context needles (254k and 1M) were not repeated on abf82258.

## Performance options

Each keeps the replies' bits; the numbers are this Mac's (`evidence/`).

- **Streams (`parallel`).** One stream decodes at 105 tokens/s, bound by reading the 7.7 GiB of weights each step.
  The weights are read once a round for every stream's row: two streams give 176-187 tokens/s in all, four 294-321
  (12.9 ms a round against 9.6), each reply equal to its run alone; the first token moves from 0.05-0.07 s to
  0.13-0.17 s at four. The engine's row kernel holds up to 32 rows a launch.
- **Weights.** The release's bf16 is exact and is the default. `variant = "8bit"` serves an MLX affine 8-bit
  conversion instead: 166 tokens/s a stream (4.1 GiB a step), 353-368 with four, for a measured loss against the
  release; the [8-bit variant](#8-bit-variant) section has the numbers. 4-bit was not measured.
- **Context and memory (`context`, `load_limit_gb`).** 36 KiB a token in the nine full layers plus 270 MiB of rings a
  stream: 64 GB holds four 262,144-token streams or one of 1,048,576. Cold prompts run 8.9k-10.7k tokens/s to 37k
  tokens, 5.0k at 149k and about 1.0k averaged over a million (the full layers' attention grows with the prompt).
- **Prompt cache (`prompt_cache_gib`).** A resumed turn (5,499 of 5,624 tokens kept) starts in 0.066 s against 0.59 s
  cold, with the same reply; kept states give way to streams that need the room.
- **Copy drafts (`drafts`).** On by default from the lane core, replies equal to `drafts = false`; they accepted 7-15%
  of their tokens on the harness's prompts and gave no measurable decode gain there. The family has no draft head.
- **Thinking.** On by default (the template's); it lengthens replies, not the cost a token; prompt speed with it on is
  within 1% of off.
- **Not offered.** No int8 or fp8 cache, no fast-math, no reduced-precision sums.

## 8-bit variant

`variant = "8bit"` (in `local.toml`, or `VARIANT=8bit`) serves an MLX affine 8-bit conversion of the release, groups of
64, made from the downloaded release by the engine's own converter:

```bash
python3 -m mac convert           # model_dir -> model_dir_8bit (~/models/Spark-X2.6-4B-8bit), 4.07 GiB, about 13 s
(cd ~/models/Spark-X2.6-4B-8bit && shasum -a 256 -c PATH_TO_THIS_REPO/checksums/spark-x2.6-4b-8bit.sha256)
VARIANT=8bit python3 -m mac up
```

The conversion is deterministic: the shards' hashes in `checksums/` came out the same from numpy 2.5.3 and 2.5.4, and
the engine serves the folder with the same kernels the CUDA family's checkpoints use.

**What it costs.** It is a lossy conversion of a release that fits every Mac in its own precision, so it is not the
default. Against the release through HF in fp32 over the same 1,500 positions: mean KL 0.0030 (0.0021 for the
conversion itself, HF to HF), top-1 97.8% (bf16: 0.00076 and 99.6%); two of the four greedy chats part from the
release's within a hundred tokens (at tokens 5 and 48), the other two equal it. One-row steps and windows still give
the same logits bit for bit, and every serving check passes (`evidence/8bit-quick`: structured 18/18, tools 5/5,
sampling 9/9, turns, soak 60/60, concurrent 12/12, multi-turn 8/8).

**What it gains**, measured on the same engine commit (`evidence/8bit-speed`, `8bit-mlx-speed`, `8bit-prefill`; the
needles on cc840195):

| | bf16 (default) | 8-bit |
|---|---:|---:|
| decode, 1 stream | 105 tok/s | 166 tok/s |
| decode, 2 streams, all | 176-187 | 259-268 |
| decode, 4 streams, all | 294-321 | 353-368 |
| cold prompt 2.4k / 9.3k / 37k / 149k | 9,042 / 10,723 / 8,946 / 4,955 tok/s | 8,258 / 9,266 / 7,766 / 4,577 |
| resumed turn, first token | 0.066 s | 0.05 s |
| 254k needle (depths 0.1 / 0.5 / 0.9) | found, 73-76 s | found, 78-79 s |
| 1M needle (`context 1048576`, `parallel 1`) | found, 1,035 s | found, 1,050 s |

Decode is faster because a step reads 4.1 GiB instead of 7.7; prompts are a little slower because the tensor-unit
matmul dequantizes on the way in.

**TensorFold leaderboard.** The board's community-v1 suite was run on `XHToken/Spark-X2.6-4B` with cc840195
(`python3 -m harness leaderboard spark-x26-4b-m5`) and could not be completed, so nothing was published: its `code`
fixture is an instruction sent as raw text, and at temperature 1 the model answers it with an immediate end token on
about one attempt in four (the release gives that token 22.8% of the first-token mass there, through HF in fp32; here
the warm-up and one of five repetitions were empty and one more ended at 251 of 256 tokens). The chat cells and the
greedy code cell complete at 105-106 tokens/s. A cell counts only when every repetition completes, and the run was
not repeated until one happened to.

**Quality.** No quality suite was run for this recipe (neither thinking off nor `--plan thinking`).

## Notes

- **Tool calls** use the template's form, which the server parses into OpenAI `tool_calls` and constrains for
  `tool_choice: required` or a named tool.
- Only the M5 Ultra was checked; a smaller-RAM Mac needs a lower `load_limit_gb` and fewer or shorter streams
  (the weights alone need 7.7 GiB).

## License

The recipe is MIT. Spark-X2.6-4B's weights are Apache-2.0 (XHToken); TensorFold is Apache-2.0.
