# Latest additive reward report

Open comparison.html: all 24 saved trajectories, latest E/G/A/B/F components only.
There is no Original reward column. Previous reports and inputs are unchanged.

Formula: R=.15E+F(.20G+.45A+.15B+.05). B is binary full-reference completeness,
requires A=1, and leaves .15 headroom above otherwise perfect core correctness.
E is source sufficiency (0 none, .5 partial, 1 full), using all delivered receipts.
G checks material factual claims against valid final citations; subjective
conclusions belong in A. B checks material reference coverage, not grounding.

Judge: Inference Hub openai/openai/gpt-5.6-terra. Protocol:
workshop-rubric-v6-additive. The transport metadata retains its existing v5
client prompt-version label; the new protocol and complete prompt/payload are
included in cache identity. Raw judgments and question/reference rubrics are
preserved in each sidecar. Source hashes match all local originals.

## Unresolved reference conflict

22/24 scores resolved. Both 00684 records remain unresolved, not zero.
Gold says gross margins declined .8%, while cited displayed margins are 19.4%
and 18.5%, implying .9 percentage points. The judge cannot reliably award the
full-reference bonus under that conflict. Raw component judgments are available
in the report even though aggregate rewards are null. No target was changed.
Reported means explicitly exclude these two records (11 resolved per phase).

## Verification and limits

162 tests passed, 14 opt-in tests skipped, 43 subtests passed in isolated remote
SDK environment. The four additive live controls passed separately: incomplete
but core-correct, fully complete, wrong conclusion with supported number, and
uncited-but-retrieved evidence. The first qualitative-control wording was
ambiguous; it was clarified to assert the objective ratio-below-one comparison.

An initial judge pass accepted incorrect rounding of a volunteered current
ratio. The final prompt explicitly requires recomputation; final 00807 judgment
correctly rejects 1.43 for 15754/10936 (rounds to 1.44). The correct quick ratio
.96 retains factual support. Final SFT 00807: E=1, G=4/7, A=0, B=0, F=1,
reward=.314286. SFT 00499 reward=.619444 with A=1 and B=0.

These remain model judgments, not human-verified training targets. Claim
segmentation varies. In particular, final 00807 includes a general statement
about the relevance of quick ratio among unsupported claims; that classification
deserves review because general definitions and subjective interpretation should
not require separate filing citations. Do not treat the passing synthetic
controls as proving judge calibration across these real trajectories.

HTML structurally verified: 12 question articles, 24 trace panels, 24 latest-only
tables; no browser rendering verification. Formula recomputed for every resolved
record; original hashes and frozen IDs verified. No policy inference, training,
live reward deployment, teacher migration, Git commit or push occurred.
