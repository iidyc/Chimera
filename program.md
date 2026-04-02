# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `/include/*` — implementation. The files of interest are `gpu_index.cuh` and `gpu_kernels.cuh`. Modify the implementation in these two files if needed.
   - `/expr/*` — main experiment files, the only one that will be used is `expr/gpu_search.cu`. Do not modify files under this folder.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs the executable `build/gpu_search`. To luanch the experiment, enter `build` folder, compile using `make -j` if you modified the code, then simply run `./gpu_search`.

**What you CAN do:**
- Modify `gpu_index.cuh` and `gpu_kernels.cuh` (and `gpu_config.cuh` if needed) — this is the only three files you edit. Everything is fair game: optimize the implementation while keeping the logic unchanged to obtain better performance.
- Add printed profiling statistics to help you understand the behavior of the code.

**What you CANNOT do:**
- Modify files under `expr` folder. It is read-only. It contains the fixed data loading and evaluation.
- Modify `CMakeLists.txt` and invoke `cmake`.
- Try to use sudo related operations. The machine does not have super user permission.
- Modify the search procedure logic and search hyper-parameters, i.e., `nprobe = 128`, `k_rank_cluster = 1800` and `k_rank_all_tokens = 300`. You have to make sure every experiment run has the same recall within random perturbation (in the current setting, 0.952 +- 0.002).
- Switch to the usage of `stage1_binary_ip_kernel_v2` and `stage2_binary_ip_kernel_v2` kernels. You should stick to using LUT-based binary IP approach and try to improve its performance.

**The goal is simple: get the lowest query time.** Since the search logic is fixed, you don't need to worry about recall — it's always 0.95. Everything is fair game: look into the implementation and try to propose approaches to optimize the performance.

**The first run**: Your very first run should always be to establish the baseline, so you will run the `gpu_search` as is.

## Output format

Once the run finishes it prints results like this:

```
...
[PROFILE] Phase C: wait_d2h=0.090 ms, topk=0.063 ms, identify=0.071 ms, gpu_extract=0.060 ms, cpu_ip_ex=1.346 ms, wait_extract=0.008 ms, combine=0.258 ms (total=2.006 ms, 300 docs)
[PROFILE] Mode: Persistent Stage 2+3 (streaming top-k + system fence)
[PROFILE] Stage 1 time: 1.81965 ms
[PROFILE]   1. CAGRA search            : 0.769024 ms
[PROFILE]   2. GPU IVF expansion       : 0.155648 ms
[PROFILE]   3. Binary IP kernel        : 0.17408 ms
[PROFILE]   4. Aggregation + tracking  : 0.247808 ms
[PROFILE]   5. Sum doc scores (sparse) : 0.245312 ms
[PROFILE]   6. Top-k sort (sparse)     : 0.164864 ms
[PROFILE]   7. D2D copy top-k doc IDs  : 0.004096 ms
[PROFILE]   8. Memset (overlapped 1-3) : 0.685888 ms (not in critical path)
[PROFILE]   Sum accounted              : 1.76083 ms
[PROFILE] Total search time           : 4.02022 ms
[0.0182818 s] GPU search time for 5 queries.
Recall@100: 0.95
```

You can extract the key metric from the log file:

```
grep "^GPU search time" run.log
```
and
```
grep "^Recall@100:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	time	recall	status	description
```

1. git commit hash (short, 7 chars)
2. time achieved in s (e.g. 0.0182818) — use 0.000000 for crashes
3. recall achieve — if deviate too much from 0.95, and the statu should be `discard`
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	time	recall	status	description
a1b2c3d	4.02022	0.95	keep	baseline
b2c3d4e	4.20153	0.95	keep	some description
c3d4e5f	4.51434	0.01	discard	some description
d4e5f6g	3.81415	0.95	keep	some description
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune the gpu search related code with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment (should enter `build` folder and compile with `make -j` before start): `./gpu_search > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^:\|^GPU search time" run.log` and `grep "^:\|^Recall@100:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If time improved (lower), you "advance" the branch, keeping the git commit
9. If time is equal or worse or recall is not normal, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!