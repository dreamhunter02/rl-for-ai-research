# Routine failed attempts preserved

- `provider_smoke_lr1e5/console.log`: the first smoke command created its nonempty run directory before invoking the trainer; the trainer correctly refused to overwrite it with `FileExistsError`. No model sampling or training occurred in that attempt.
- `provider_smoke_lr1e5_v2`: shell redirection targeted a file inside a not-yet-created run directory, so the shell failed before Python started. No model sampling or training occurred.
- `provider_smoke_lr1e5_v3`: corrected command completed Tinker sampling/checkpoint save and exposed the actual scientific gate: zero retained groups and zero optimizer updates.
