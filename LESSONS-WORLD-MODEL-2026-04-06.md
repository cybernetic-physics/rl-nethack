# Lessons: World Model Validation

## What We Learned

### 1. The old world-model eval was not good enough

Before this pass, the world model was mostly judged by a few generic metrics:

- feature MSE
- reward MAE
- done accuracy

That was not enough to tell whether it helped policy learning.

After this pass, we now measure:

- action accuracy
- action top-3 accuracy
- future-feature cosine similarity
- reconstruction cosine similarity
- latent dead-fraction
- common action mismatches
- downstream BC trace match from transformed features

This is a real improvement to the repo. We can now debug the world model as a policy-supporting module instead of a generic predictor.

### 2. There was a real eval bug

The old eval could not cleanly score a base `v4` world model against `v4+wm_*` trace files.

That failed because:

- the model expected `302`-dim inputs
- the augmented traces had `379`-dim or larger feature vectors

That is now fixed. The eval trims augmented features back to the base observation prefix when appropriate.

So one thing we learned is:

- some of our previous world-model evaluation surface was incomplete or fragile

### 3. The original world model was weak as a direct predictor

For the baseline model `/tmp/x100_v4_world_model.pt`, held-out direct metrics were poor:

- action accuracy: `0.25`
- action top-3 accuracy: `0.8594`
- feature cosine mean: `0.4439`

Its action head had effectively collapsed toward `north`.

So the baseline world model was not actually a strong learned dynamics model, even though it had been useful as a feature augmenter.

### 4. Improving direct world-model quality is possible

The retrained action-focused models improved a lot.

Best direct metrics:

- `horizon=4` action-focused model:
  - action accuracy: `0.875`
  - feature cosine mean: `0.7211`
- `horizon=2` action-focused model:
  - action accuracy: `0.9583`
  - feature cosine mean: `0.7366`

So the world model is not stuck. It can be made much better as a predictor.

### 5. Better prediction did not produce a better teacher

This is the most important lesson.

Even though the new world models were much better by direct metrics:

- downstream BC from the old baseline world model still reached `0.9375`
- downstream BC from the better `h4` world model only tied at `0.9375`
- downstream BC from the even better-direct `h2` world model got worse at `0.925`

So:

- better world-model prediction quality does **not** automatically mean better policy representation quality

### 6. Better world-model prediction also did not fix online RL

Short RL test from the new `h4 + concat` teacher:

- teacher: `0.9375`
- best learned checkpoint: `0.9125`

So the online learner still drifted below the teacher.

That means:

- the world model is not currently the main bottleneck in online improvement

### 7. The world model is still useful, but in a narrower role

The right interpretation is not:

- “world models were a waste”

The right interpretation is:

- the world model helps as a representation and diagnostic tool
- but improving its predictive loss is not enough to solve the repo’s main problem

The main problem remains:

- how to improve beyond the teacher without drifting away from the trusted metric

## What We Should Stop Assuming

1. Lower feature MSE means a better policy representation.
2. Better action prediction inside the world model will automatically produce a better BC teacher.
3. A better world model will automatically make APPO more stable.
4. `concat_aux` is always the best augmentation mode.

The current evidence does not support any of those assumptions strongly enough.

## What We Should Believe Now

1. World-model eval must include downstream policy usefulness, not just direct prediction metrics.
2. The promotion gate for a new world model should be:
   - downstream BC improvement first
   - then short RL stability
   - only then larger RL runs
3. The world model is a support module, not the main solver of the online drift problem.
4. If we continue world-model work, the next objectives should move closer to policy usefulness, not just generic prediction quality.

## Practical Next Rule

Do not scale a new world model into a larger RL run unless it passes both:

1. It improves downstream BC trace match over the current baseline.
2. A short teacher-replay APPO run stays at or above the teacher instead of regressing.

## Bottom Line

We learned that the world-model stack is now measurable, debuggable, and improvable.

We also learned that world-model quality and policy quality are not the same thing in this repo.

That is the key lesson to carry forward.
