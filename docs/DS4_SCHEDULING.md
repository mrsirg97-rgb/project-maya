# DeepSeek: scheduling several conversations

With `--parallel N` (N > 1) the DeepSeek engine answers several conversations at once on a layer split. What a
user feels then is less how fast a token is than **what runs next**: one prompt's read takes every GPU, so a
250-token question behind a 16,000-token prompt waits for all of it, and a nearly finished answer waits for a read
that has just begun. This note is how the engine decides, and the measurements each rule came from - including the
ones that did not work. The settings are in [SETTINGS.md](SETTINGS.md#deepseek-several-conversations-at-once-parallel).

All numbers: a 3-GPU split (2x RTX 5090 + RTX 3090), DeepSeek-V4.1-Flash GSQ-RCO IQ2_XXS (~138 GB, about half the
experts in VRAM, the rest on the CPU lane from RAM and the disk), `--parallel 4`, greedy, 32-token answers unless
said otherwise. `tools/ds4/schedule_bench.py` runs them.

## The loop: events in (priority, id) order

The serve loop is one event queue (`src/ds4/ds4_evt.hpp`, the [evt](https://github.com/mrsirg97-rgb/evt) engine
of [rig](https://github.com/mrsirg97-rgb/rig), ported): an event has a priority and an id from a counter, the queue
pops the highest priority first and, among equals, the lowest id, and one thread - the loop's - runs them one at a
time. Anything may add an event; only the loop's thread touches the conversations' jobs, so nothing in them locks.

| Event | Added by | Priority |
| --- | --- | --- |
| a line (GEN, STOP, QUIT) | the reader thread | highest |
| choose the next prompt to read | a line that admitted a request, a request ended | middle |
| a conversation's token is ready | a host callback on the head GPU's stream after the window's token (`Model::multi_on_done`) | lowest |

So every request that came is admitted before a read is chosen among them, and a token is taken the moment it is
ready - in the order the GPUs finish them, not the order they were issued in.

**Owning the GPUs.** A prompt's read, and the expert cache's look at its tiers, need every GPU and nothing of the
decode in flight. They ask for them (`want_devices`): from then on a conversation whose token comes back has its
next one parked, the ones already issued come back as events, and once none is out the owner runs; then the parked
ones are issued again. An owner asked for inside another (a boundary during a read's gap) runs after it. This one
shape replaced the loop's flags and its `drain()` - which waited for each conversation's last event on the head GPU
and assumed the GPUs were idle then.

**A request past the N states** waits for a lane (it used to be answered with ERR).

## Which prompt is read next: the highest response ratio

One read at a time; the next is the **highest response ratio**: `(wait + read) / read`, `read` the prompt's tokens
left at the last read's measured rate. Waiting alike, the shortest goes first - which minimises the mean wait - and a
long prompt's ratio grows as it waits, so a stream of short ones does not starve it. No constant: the rate is
measured, the wait is the clock's.

Four requests sent at once, longest first, `MAYA_DS4_READ_ORDER=arrival` against the default (same build):

| Prompt (tokens) | First come, first served: first token / done (s) | Default: first token / done (s) |
| ---: | ---: | ---: |
| 250 | 28.4 / 30.3 | 2.9 / 3.9 |
| 1,000 | 25.8 / 30.2 | 7.7 / 8.9 |
| 4,000 | 22.1 / 30.1 | 13.4 / 14.4 |
| 16,000 | 11.0 / 15.2 | 24.4 / 25.4 |
| mean | 21.8 / 26.4 | 12.1 / 13.2 |

All four done after 30.3 and 25.4 s. The long one waits for the short ones - the trade.

## Between a read and the answers: shortest remaining work first

The same question comes up at every point a read could start or go on: read now, or let the running answers
decode? The engine answers it one way everywhere - **the shortest remaining work first**:

- the read's remaining time: its tokens left at its measured rate (the read's own so far, from its progress);
- an answer's: its expected tokens left at a decode round's measured time (a round: a token for every conversation
  answering). Its expected length is the mean of the answers ended so far (its `max_new` at most; none ended: its
  `max_new`), or - once it has passed that - as many tokens again as it has (the longer it has run, the longer it is
  likely to run);
- the answers go first while the soonest of them would end before the read does.

Before each read the answers get a round (which measures one); at a read's chunk boundaries the same rule makes the
**decode gaps** (a round at least, each round measuring them again, 4000 ms at most). With the 250-token request's
answer at 200 tokens:

| Gaps | Mean first token (s) | Mean done (s) | All done (s) |
| --- | ---: | ---: | ---: |
| fixed 50% (`MAYA_DS4_PREFILL_GAPS=50`) | 10.9 | 24.9 | 28.8 |
| shortest remaining work first (default) | 16.2 | 18.6 | 30.7 |

The rule minimises the mean time to a whole answer - an agent waits for the whole answer before it acts on it -
at some cost to the first token of the prompts after a long answer.

## A short prompt that comes during a long read

A request that arrives while a read is paused at a boundary is admitted; a prompt of one chunk may be read in its own
lane in windows of 16 right there - but windows read a prompt several times slower than chunks, so only where that is
**measured** quicker than its turn: its windows at what a prompt window took here (not measured yet: its first window,
which measures one), against the rest of the read and its own read at the read's rate. A prompt waiting that fits a
chunk asks again at every boundary.

| | 4,000-token prompt sent during a 16,000-token read | 256-token prompt sent during a 100,000-token read |
| --- | --- | --- |
| before (windows whenever it fits a chunk) | read in windows: 54 s | - |
| default | read after it in chunks: first token 24.6 s from the long one's send | read in windows: all 16 answer tokens (33 tok/s) before the read was through |

## What did not work

Kept here because each one looked right:

| Tried | What happened | Why |
| --- | --- | --- |
| Cutting short reads into 4 chunks (so they have gap points, as GLM's `MAYA_GLM_PREFILL_MIN_CHUNKS`) | a 4,000-token read ~12 s instead of ~4 s; four 2,000-token reads 27 s instead of 16 | a chunk streams every expert its rows route to - about as many for 1,000 rows as for 4,000 - so where they do not all fit in VRAM a cut read streams them again a chunk. Off by default (`MAYA_DS4_PREFILL_MIN_CHUNKS`); where every expert is in VRAM a chunk's cost follows its rows |
| Fixed gap shares of 50, 20, 10% | the reads took as long at every share; lower shares only starved the answers (a 250-token answer done at 13, 31 s) | the cost was the chunks', not the gaps' |
| Shortest first, aged by "waited longer than its own read" | prompts arriving during a long read were read in arrival order | every short prompt behind a long read has waited longer than its own read: all aged, so all first come, first served. The response ratio replaced it |
| Choosing the read when a request is admitted | shortest first made no difference: the first request's read began before the others were admitted | the choice is an event of its own now, below the lines |

## Bugs found on the way

- **The PCIe lane's pulls numbered per conversation.** The fetch kernel's flags are its GPU's and wait for a value
  equal to the pull's number; with several conversations each numbered its requests from 1, so two on one GPU found
  the flags already set and copied experts before they were read (NaN rows, an illegal address a few tokens later -
  only when two prompts arrived together, while both numbers were small). A pull's number is now
  `seq * conversations + conversation`, one conversation's `seq` exactly.
- **A 512-2047-token prompt through windows at a sixth of the speed** on a fresh engine (one conversation too): the
  choice of path modelled a window from its experts' bytes and left out its fixed costs; it now uses what a full
  window measured.

## Reproducing

    python tools/ds4/parallel_test.py --pack <pack> --lanes 4             # exactness, stop, slots, reading
    python tools/ds4/parallel_test.py --pack <pack> --lanes 4 --measure   # + the throughput table
    python tools/ds4/schedule_bench.py --pack <pack>                      # the tables above
