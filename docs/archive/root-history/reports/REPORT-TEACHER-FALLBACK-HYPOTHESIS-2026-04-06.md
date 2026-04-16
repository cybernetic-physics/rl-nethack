# Teacher Fallback Hypothesis

## Short Version

Rollout-time teacher fallback is now implemented and validated in the repo.

What the first disciplined probe showed is:

- fallback to auxiliary distillation teachers is harmful
- fallback to the exact trusted `0.9875` teacher materially improves late stability
- but fallback alone still does not produce a teacher-beating learned checkpoint

So the hypothesis is now narrower:

> teacher-as-base is a real stabilizer, but it needs a better improvement mechanism on top of it than raw confidence fallback.
