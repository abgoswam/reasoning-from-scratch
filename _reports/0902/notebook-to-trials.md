# Prompt: split a chapter notebook into `trials_*.py` files

> **Two prompts live here.** This one assumes the chapter teaches by defining a
> function and then demonstrating it, so a shared `trials_00.py` hub pays for
> itself — the ch06 shape. If the chapter instead ships reference logs or a set
> of script variants where each concept is a *diff* between two scripts, use
> [`notebook-to-trials-variant.md`](notebook-to-trials-variant.md) — the ch07
> shape. That file carries the four-question test for choosing.

Paste the block below, filling in the four values at the top. It encodes the
conventions already established in `ch06/01_main-chapter-code/trials_00.py`
through `trials_10_checkpoints.py` — the output should be indistinguishable in
style from those files.

---

## THE PROMPT

```
CHAPTER          = chNN
NOTEBOOK         = <path>/chNN_main.ipynb
LIBRARY MODULE   = reasoning_from_scratch/chNN.py
OUTPUT DIR       = <path>/chNN/01_main-chapter-code/

I am working through "Build a Reasoning Model (From Scratch)". I read the
chapter notebook, but a notebook cell cannot be imported, breakpointed, or
diffed against upstream, and every cell depends on kernel state from the cells
above it. I want the notebook split into standalone Python scripts I can set
breakpoints in and run individually.

Do this in two steps.

### Step 1 — build trials_00.py, the shared library

trials_00.py is a local editable mirror of LIBRARY MODULE. Every other file
imports from it, so it is the one file I edit when experimenting.

Copy LIBRARY MODULE verbatim, then apply exactly these changes:

1. Rewrite relative imports to absolute:
   `from .qwen3 import X`  ->  `from reasoning_from_scratch.qwen3 import X`
   (trials_00.py is a loose script, not a package module.)

2. Append a section, under this exact comment banner:

   # ---------------------------------------------------------------------------
   # Notebook-only definitions: these appear in chNN_main.ipynb but not in
   # reasoning_from_scratch/chNN.py, because the chapter builds them up in stages.
   # ---------------------------------------------------------------------------

   Into it, put every function the notebook DEFINES but the shipped module
   DROPPED — the intermediate versions the chapter arrives at and then
   supersedes. These matter: they are what lets a trials file demonstrate that
   two spellings of the same computation agree. Find them by diffing the
   notebook's `def` names against the module's.

3. Append a second section:

   # ---------------------------------------------------------------------------
   # Shared setup used by the trials_XX_*.py scripts.
   # ---------------------------------------------------------------------------

   Into it, hoist the fixtures the notebook reuses across cells — the running
   example prompt, the hardcoded rollout list, a `load_base_model()` helper.
   Give them module-level UPPER_CASE names so every trials file references the
   same values.

4. Put this in the header, with the line offset computed to strip exactly the
   header comment block you wrote (count them — do not guess):

   # Local editable mirror of reasoning_from_scratch/chNN.py.
   # Every trials_XX_*.py imports from here, so this is the one file to edit
   # when experimenting. Check for drift with:
   #   diff <(sed -n '<first_code_line>,<last_mirrored_line>p' trials_00.py) \
   #        <(sed -n '<first_code_line>,$p' ../../reasoning_from_scratch/chNN.py)

   The check must isolate the mirrored region only, so the appended sections in
   (2) and (3) do not show up as permanent phantom drift.

trials_00.py has NO main() and NO __main__ guard. It is a library; importing it
produces no output.

### Step 2 — one file per concept

Walk the notebook top to bottom. Each time the author defines a function and
then demonstrates it, or builds one idea across a run of cells, that is one
concept and one file:

    trials_01_<concept>.py
    trials_02_<concept>.py
    ...

Number in notebook order. Name the concept in snake_case, one or two words
(`dataset`, `rollouts`, `rewards`, `advantages`, `logprobs`, `pg_loss`,
`grpo_loss`, `training`, `checkpoints`).

Every file follows this shape exactly:

    # Concept: <short phrase> (notebook cells N-M)   [cost tag, if any]
    #
    # <2-5 lines: why this concept is non-obvious, or what to watch for when
    # it runs. State the gotcha the notebook glosses over. Not a restatement
    # of what the code does.>

    <stdlib imports>

    <torch>

    <from reasoning_from_scratch.chXX import ...>
    <from trials_00 import ...>
    <from trials_NN_earlier import ...>


    def <glue_name>(...):        # only if this file introduces reusable glue
        ...


    def main():
        ...


    if __name__ == "__main__":
        main()

Rules for the body:

- The file holds the EXPERIMENT, not the definitions. Import every function
  under study from trials_00 rather than redefining it.
- Exception: when the notebook writes something as loose top-level cell code
  (a bare expression, an inline for-loop), give it a name and a `def` in the
  file that first needs it, so later trials can import it. Keep the notebook's
  own variable names and print formatting verbatim.
- Later files import that glue from earlier files by name.
- Print intermediate values. The point is watching numbers, not asserting.
- `main()` may return a value and may take default arguments.
- No docstrings. No comments inside `main()` unless a line is genuinely
  surprising.

Cost tags — put one on the Concept line whenever the file is not instant:

    [loads model]          needs the checkpoint in memory
    [model, very slow]     trains; minutes to hours
    [downloads]            fetches from the network

Files with no tag must run on CPU in seconds. I use the tags to decide what to
run on a whim versus what needs a free GPU.

### Constraints

- Do not refactor, improve, or modernize the author's code. Mirror it. The
  value is that trials_00.py can be diffed against upstream.
- Do not invent behavior the notebook does not show.
- Do not write tests. These are runnable explanations, not assertions.
- Skip cells that are pure prose or images.
- If two adjacent concepts are each under ~15 lines, merge them into one file
  rather than creating a stub.

### Deliver

1. The files, written to OUTPUT DIR.
2. A table: file -> notebook cells -> cost tag -> what it imports from trials_00.
3. Any notebook `def` you could not place, and why.

Then confirm: every file runs standalone, and the untagged ones finish in
seconds on CPU.
```

---

## Notes for next time

**Where this came from.** Reconstructed from the ch06 session on 19 Aug, where
the shape emerged over four messages: standalone-file-per-concept (rejecting
pytest), then `trials_00.py` as the importable hub. This prompt collapses that
negotiation into one pass.

**What was never specified last time** and is now pinned down here: the
`# Concept:` header format, the `(notebook cells N-M)` citation, the cost tags,
and the drift check. Those accreted as conventions — which is why the ch06
drift check shipped with a stale `sed 1,7d` offset that reports phantom drift.

**Not covered by this prompt:** chapters shaped like ch07 — see the variant
file above. Also the `trials_12`–`trials_14` style — the toy-model
tutorials with `Part 1..N` sections that explain a mechanism the book never
addresses. Those come from getting stuck at a breakpoint, not from reading the
notebook, so they cannot be generated up front. Ask for one when you hit the
confusion, with the constraint that made the existing three work: *keep it very
simple, small enough to put breakpoints anywhere, and verify the toy against the
real function.*
