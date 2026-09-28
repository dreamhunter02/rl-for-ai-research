# Observed case studies

- **B1 development smoke:** 12 scheduled questions were evaluated with one rollout each. Two reached accepted terminal submissions, four stopped on length, six reached the turn limit, and one episode was unresolved; no accepted submission was correct or grounded. This is a small development result, not a final evaluation.
- **Training signal gate:** the one nominal train batch sampled eight groups × eight trajectories. All eight groups had zero reward and zero variance; four were marked unresolved and none were retained. The trainer therefore made zero optimizer calls and logged one skipped zero-signal batch.
- **Final checkpoint diagnostic:** the trainer still saved a final sampler checkpoint and evaluated it on the matched 12-question dev split, but it is labeled a zero-update smoke checkpoint, not R1.
- **Provider/infrastructure:** both Tinker sampling and checkpoint save completed; no infrastructure failures were recorded.
