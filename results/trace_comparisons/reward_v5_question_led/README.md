# Question-led reward v5 — saved dev24 rescore

Open `comparison.html`. It contains all 12 base and 12 SFT original trajectories,
their historical scores and the new component judgments. Non-applicable numeric
or semantic components display N/A. `summary.json` compares v4 with v5; the HTML's
Original column instead preserves the original pre-v4 harness scores.

## Corrected contract

1. Identify requirements from the question alone, without the reference/candidate.
2. Use the reference to answer only those requirements. Mode and claim IDs cannot
   expand. Gold supporting facts are not an exhaustive answer checklist.
3. Assess semantic equivalence and any requested final quantities. Optional
   numeric claims are checked under grounding; they do not activate numeric reward.
4. Ground only material candidate claims against authentic final citations.
   Wrong optional arithmetic reduces grounding, not semantic credit unless it
   changes or contradicts the required answer. Split conclusions from numbers.

The formula is unchanged: R = F*A*(0.5+0.5G); mixed A=.6N+.4S,
numeric-only A=N, semantic-only A=S. Conditional fallback explanations are not
additional mandatory branches alongside an otherwise valid numeric answer.

## Results

| Measure | Base | SFT |
|---|---:|---:|
| Saved records | 12 | 12 |
| Mean reward v4 → v5 | .344444 → .366667 | .590377 → .576389 |
| Correct under v5 | 5/12 | 8/12 |
| Fully grounded successes | 2/12 | 4/12 |
| Unresolved | 0 | 0 |

00499 is semantic-only: both get N=null, S=A=1. Base G=1/6 yields
R=.583333; SFT G=.2 yields R=.60 (previously .233333 and .342857).
Missing the reference's other two metrics causes no penalty. Unsupported
volunteered figures still reduce grounding.

00807 is also semantic-only: SFT's wrong healthy-liquidity conclusion receives
zero, replacing the .45 numeric partial credit under v4. The question does not
explicitly request a numerical result. Mixed questions that do request both a
number and an assessment still receive separate numeric/semantic credit.

00438 SFT receives S=1, G=.8, R=.9: the requested declining-margin conclusion
is correct; the unsupported explanation reduces grounding. 00540's primary
request is numeric; its conditional applicability fallback does not create an
extra mandatory semantic answer invented from outside the reference.

## Verification and limits

Final code verified in isolated SparkyOne copy: 149 tests passed, 10 opt-in live
tests skipped, 43 subtests passed. All 10 live tests separately passed against
Inference Hub. Controls cover qualitative/numeric/mixed/conditional applicability,
wrong final values/scales, wrong conclusions, missing citations and volunteered
wrong arithmetic with an otherwise correct qualitative conclusion.

All 24 records resolve, source hashes match local originals, and each phase
contains exactly the 12 frozen dev IDs. HTML structurally verified: 12 question
articles, 24 trace panels, and explicit N/A. Not browser-render verified.

Final judge: openai/openai/gpt-5.6-terra through Inference Hub. Scorer:
workshop-rubric-v5-question-led; prompt: component-judge-v4-question-led. A first
GLM rubric attempt failed schema validation and was not used. Two intermediate
Terra prompt revisions failed the new volunteered-arithmetic control, which
motivated the final separation and atomic-claim instructions. These intermediate
outputs are not the scores in this report; cached judgments/drafts remain intact.

No second-judge consensus is claimed for v5. Passing controls do not prove
universal judge accuracy: grounding segmentation and support decisions still need
calibration. Higher or lower reward is not evidence of model improvement—these
are the same saved outputs under a repaired rubric.

Local code updated and tested remotely in a scratch checkout only. No training,
fresh policy inference, production deployment or Git push. Original reports and
teacher data remain unchanged. The earlier teacher export still carries v4
scores and has not been requalified under v5.
