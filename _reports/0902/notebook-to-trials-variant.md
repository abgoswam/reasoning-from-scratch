# Prompt: trials files for a chapter built from script variants and logs

The sibling prompt (`notebook-to-trials.md`) assumes the chapter teaches by
defining a function and demonstrating it, so a shared `trials_00.py` hub pays
for itself. Some chapters are not shaped that way. Chapter 7 is the worked
example: §7.2 hands you five finished CSVs in `02_logs/`, and §7.3-7.6 are four
complete training scripts in `03_rlvr_grpo_scripts_advanced/`, each a diff
against the one before.

## Which prompt to use

Run this test on the chapter before choosing:

| Check | Hub prompt | This prompt |
|---|---|---|
| Do 5+ concepts need the same model / tokenizer / dataset in memory? | yes | no |
| Is the chapter's library module a set of small independent helpers? | no | yes |
| Does the chapter ship reference logs or script variants as teaching material? | no | yes |
| Is each concept a function with a demo cell, or a diff between two scripts? | function | diff |

Mixed answers mean split it: hub prompt for the parts that share state, this one
for the rest.

---

## THE PROMPT

```
CHAPTER        = chNN
NOTEBOOK       = <path>/chNN/01_main-chapter-code/chNN_main.ipynb
LIBRARY MODULE = reasoning_from_scratch/chNN.py
REFERENCE LOGS = <path>/chNN/02_logs/            (if the chapter ships any)
SCRIPT VARIANTS= <path>/chNN/03_*/               (if the chapter ships any)
OUTPUT DIR     = <path>/chNN/01_main-chapter-code/

I did this for chNN-1 already: a trials_00.py mirroring the chapter's library
module, then trials_01..NN, one per concept, each runnable standalone so I can
set breakpoints. Read those files first — OUTPUT DIR's sibling chapter — so you
match their naming and header style.

But do NOT assume the same decomposition transfers.

### Step 1 — plan before you write anything

Read NOTEBOOK end to end, plus REFERENCE LOGS and SCRIPT VARIANTS. Understand
the flow of what the author is teaching and in what order. Only then, give me a
recommendation as a table:

    proposed file | notebook cells | section | what it teaches | needs GPU? | needs logs?

Include in the table any concept you decided NOT to give a file, and why.

State explicitly whether a trials_00.py hub is worth it here. It is worth it
only when several files need the same expensive shared state. If the chapter's
library module is a handful of small independent helpers, say so and skip the
hub — I would rather each file inline a verbatim copy of the one function it
studies than import through a mirror that buys nothing.

Wait for me to approve the table before writing files.

### Step 2 — write the files

    trials_01_<concept>.py ... trials_NN_<concept>.py

Numbered in the order the chapter teaches them, not the order of the library
module. Each file follows this shape:

    # Concept: <short phrase> (notebook cells N-M, section X.Y)
    #
    # <3-6 lines: the framing. What is this section really about, and what
    # would a reader get wrong? State the thesis, not a summary.>
    #
    # <one line on cost: what it reads, whether it needs a model or GPU>
    #
    #   Part 1: <claim>
    #   Part 2: <claim>
    #   ...

    <imports>

    <module-level fixtures: hardcoded tensors lifted from the notebook,
     paths to REFERENCE LOGS, constants>


    # ---------------------------------------------------------------------------
    # Part 1 -- <the claim, in prose. Why this is not obvious.>
    # ---------------------------------------------------------------------------
    def part1_<name>():
        print("=" * 74)
        print("PART 1  <claim>")
        print("=" * 74)
        ...


    def main():
        part1_<name>()
        part2_<name>()
        ...


    if __name__ == "__main__":
        main()

Rules:

- Prefer zero dependencies. Most files should run on CPU in under a second
  with no model loaded. Achieve that by hardcoding the notebook's own input
  vectors as module-level constants, with the interesting deltas annotated:

      OLD_LOGPS = torch.tensor([-10.9243, -20.3546, -14.6130, -23.3677])
      #                          ^ +3.0     ^ +0.2     ^ -2.0     ^ same

- When the file studies a function that exists in LIBRARY MODULE, copy that
  function verbatim into the file rather than importing it. These files are
  read as much as run, and the reader should not have to open another file.
  Do not improve the copy.

- Where the chapter ships REFERENCE LOGS, read them from disk. Do not
  re-download; note in the header that the notebook's download calls exist for
  readers who did not clone the repo.

- Where a concept is a diff between two SCRIPT VARIANTS, make the diff the
  subject: show the earlier behaviour, then the added mechanism, then what
  switch actually turns it on.

- Give the last Part of the first file a "compare your own run against the
  reference" section, so the trials connect to runs I do later.

- Each Part prints intermediate values and ends with a one-line takeaway
  (`-> ...`). No docstrings. No asserts.

- Number Parts only because they are a real reading order. If a file has one
  claim, give it one Part or none.

### Constraints

- Do not refactor the author's code.
- Do not invent behaviour the chapter does not show.
- Do not write tests.
- If a proposed file would be under ~60 lines, fold it into its neighbour.

### Deliver

The approved table, updated to final, plus the files. Then tell me which files
need a GPU and which are instant, so I know what I can run on a whim.
```

---

## Appendix: the second track

Ch07 also produced `submit_02_baseline.py` .. `submit_06_format_reward.py`.
That fork is worth repeating whenever a chapter ships runnable training
scripts, and it came from three decisions stated as constraints:

1. **Self-contained copies, not dynamic loading.** The rejected proposal was
   `stage = load_module("7_3_plus_tracking.py")`. Copying each variant into its
   own `submit_0X_<name>.py` means the diff between consecutive files *is* the
   section's contribution — worth more than the deduplication.
2. **Fail closed on logging.** Abort before the model loads if W&B is missing,
   unauthenticated, or unreachable, so a long run never trains with its metrics
   going nowhere. Keep a `--no_wandb` escape hatch for smoke tests.
3. **Every metric written twice** — to W&B and to a fresh
   `runs/<timestamp>-<wandb-run-id>/` directory, so the local files and the
   dashboard point at each other.

Mark every deviation from the upstream script with a single tag comment
(`# WANDB`) and list the sites in the file header, with a drift-check command
against the original.

---

## How ch07 actually happened

Reconstructed from the 23 Aug session. The opening prompt did three things the
ch06 one did not, and those three are why the output differs:

> *"similar to what we did in ch06, i want to create trials_XX files but this
> time for ch07 ... can u first confirm if this is correct ? can u give your
> recommendation in a tabular format ... make sure u go over the ch07 notebook.
> you should understand the flow of what the author is trying to teach. only
> then give your recommendation"*

It named the precedent, demanded a plan before any file was written, and made
reading the chapter a precondition of the plan. The decomposition was therefore
designed for ch07 rather than transplanted from ch06 — which is how the hub got
dropped and the logs became first-class material.

Two follow-ups did the rest of the shaping: *"do they rely on downloaded
logs?"* pinned the `02_logs` dependency, and *"i want trials_xx scripts for my
own learning ... submit_xx scripts where i can do my own training run"* created
the two-track split.

One thing nobody asked for: the `Part 1..N` structure. It carried over because
`trials_12`-`trials_14` were built earlier in the same conversation, so the
convention was already in context. It is written down here so it no longer
depends on that accident.
