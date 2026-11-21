# Private PI0.5 scaffold for Reality Check

This private repository contains a LeRobot π0.5 inference adapter behind the
starter SDK's contract v0 / observation-action schema v2. The checkpoint is
loaded from the runner's read-only `/weights` mount. No policy weights or gated
tokenizer assets are in this source tree.

## Base model versus a submission checkpoint

The official `lerobot/pi05_base` checkpoint is a fine-tuning base. Its config
expects three 224×224 views, 32 state dimensions, and a 32-dimensional action
feature. It is not tuned to the Reality Check UR12e, camera geometry, state
normalization, or joint/gripper semantics. The `32` is padding capacity, not a
documented 32-channel robot command. The checkpoint defines no named actuator
ordering. The LeRobot PI05 docs show a seven-dimensional LIBERO action, but that
does not establish the UR12e order.

A useful submission needs a complete, embodiment-specific LeRobot checkpoint
with `config.json`, weights, and saved pre/postprocessor JSON files. This
adapter does not resolve a base model plus a PEFT/LoRA-only adapter. Its
six-target-plus-gripper action conversion is valid only if the fine-tuned
checkpoint was trained with that exact action order and semantics. Mounting
`lerobot/pi05_base` can exercise model loading and inference, but its output is
not established as meaningful or safe for the task.

The benchmark's `three_view` images are `wrist`, `left`, and `right`; PI0.5's
base config expects `base_0_rgb`, `left_wrist_0_rgb`, and `right_wrist_0_rgb`.
The scaffold maps `left → base`, `wrist → left wrist`, `right → right wrist`.
Those camera meanings do not match; train/fine-tune with this mapping or change
the map and checkpoint together. The adapter supports the published
`three_view` input only. It ignores the goal image because standard PI0.5
preprocessing accepts live images, state, and language only.

## Build the local private Spark image

The saved PI0.5 preprocessor names the gated tokenizer
`google/paligemma-3b-pt-224`. An authorized HF account must accept Google's
terms and download the tokenizer. The tokenizer files are stored outside this
source tree at `/tmp/rc-private-pi05-tokenizer-cache`; they are not copied into
GitHub.

Build the private base image from the already-installed local `rc-lerobot:train`
image, then build the policy image:

```sh
docker buildx build \
  --build-context pali-tokenizer=/tmp/rc-private-pi05-tokenizer-cache \
  -f Dockerfile.base.local -t rc-pi05-base:local .
docker build -t rc-private-pi05 .
```

`Dockerfile.base.local` copies tokenizer files into `/opt/pali-tokenizer` and
sets `RC_PI05_TOKENIZER_PATH`. The default `Dockerfile` uses that private local
tag and does not need a build secret. This setup works only on a Spark runner
with `rc-lerobot:train` and `rc-pi05-base:local` available. Do not upload or
publish the local base image because it contains gated tokenizer assets. No
tokens are embedded in the Dockerfiles or source. Network access stays enabled.

At runtime, mount a complete, immutable fine-tuned PI0.5 snapshot at
`/weights`:

```sh
docker run --rm --gpus all -p 8080:8080 \
  -v /path/to/fine-tuned-pi05:/weights:ro \
  rc-private-pi05
```

`RC_PI05_DTYPE` optionally selects `float32` or `bfloat16`; default is the
checkpoint's saved dtype. LeRobot PI05 0.7.0 accepts those two dtypes. Verify
dtype support and latency on the actual Spark. Check `GET /readyz`, then send a
real schema-v2 `/act` request with three camera images, an instruction, and six
finite `proprio.joint_pos` values. The SDK self-test checks only the HTTP
envelope; it does not load this checkpoint or validate its action semantics.

## Size and runtime budget

At Hub revision `22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0`,
`lerobot/pi05_base` reports 3,616,757,520 F32 parameters and 14,467,165,872
stored bytes (about 14.47 GB decimal / 13.47 GiB). That is the policy snapshot,
excluding CUDA PyTorch, LeRobot, and the tokenizer. Tokenizer assets fetched for
this scaffold total about 22 MB; tokenizer.json alone is 17.5 MB. Policy
weights are mounted separately, outside the Git build context.

Spark policy containers should use the documented 100 GB memory cap; the
runner's 16 GB default leaves little headroom beyond 14.47 GB of F32 weights.
The simulator and policy share the Spark's 128 GB unified memory. The evaluator
allows up to 60 seconds of scenario time per repetition, but this is not a
latency guarantee. PI0.5's base config uses 10 flow-matching inference steps and
a 50-action queue. Measure cold load, warmup, and per-call latency on GB10.

## Inspected artifacts

The official model card, config, and saved processor JSON can be inspected at
https://huggingface.co/lerobot/pi05_base. The config specifies `google/paligemma-3b-pt-224`; the π0.5 model repo itself is
public, but this tokenizer repo is gated/manual. LeRobot 0.7.0's official
processor and policy implementations are used.

This private repository is an organizer test scaffold. It has not yet been
validated with the full model or submitted through the Reality Check dashboard.
The current dashboard accepts only public repositories, so this private repo
cannot currently be submitted there.
