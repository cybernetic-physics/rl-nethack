# RL-NetHack Rolling Research Thesis

This document is a living synthesis of the repo-local markdown trail through 2026-04-06. It is meant to function as a durable research narrative rather than another point-in-time handoff note.

The core goal is to explain:

- what problem this repo is trying to solve,
- how the system and benchmarks changed over time,
- what each major markdown artifact contributed,
- which conclusions held up,
- which conclusions were later weakened or invalidated,
- and what the current evidence actually says.

## Problem Statement

The repo began with a forward-model thesis:

- use NetHack trajectories to train a model that predicts action-conditioned state changes,
- use memory-augmented state descriptions rather than raw pixels,
- build deterministic evaluation and artifact manifests around that training path,
- and only later convert the learned knowledge into better control.

The repo then broadened into a stronger control thesis:

- build a high-quality teacher policy for short-horizon NetHack behavior,
- measure it with a trustworthy deterministic benchmark,
- and learn an online improver that can beat that teacher without drifting into reward-hacking or shallow repetitive behavior.

By 2026-04-06, the dominant question is no longer:

- "Can this repo run RL at all?"

It is:

- "How do we improve beyond a strong offline teacher without losing teacher-aligned behavior under online updates?"

## Stable Setup And Working Assumptions

Across the documents, the stable hardware and systems picture is:

- machine: `4x NVIDIA H200`,
- typical GPU split: `0,1` for local vLLM serving and `2,3` for training,
- main environment: NLE / NetHack through custom wrappers,
- main control stack over time:
  - forward-model SFT,
  - task-harness teacher control,
  - BC / behavior-regularized offline teachers,
  - Sample Factory APPO online improvement,
  - world-model feature augmentation,
  - later proxy-reward and improver-architecture work.

Stable conceptual assumptions that survived multiple documents:

- raw NetHack score is too sparse and too misaligned to be the main early objective,
- teacher quality matters more than blind RL throughput,
- evaluation quality matters more than another large run,
- better representation helps, but representation alone does not solve online drift,
- the repo should be judged by the strongest trustworthy benchmark available for each phase, not by raw training reward.

## Comparison Hygiene

Direct comparisons across the entire history are dangerous. The codebase changed meaningfully between runs.

Numbers are only safely comparable when all of the following match:

- observation version (`v2`, `v3`, `v4`, `v4 + wm_*`, etc.),
- trace dataset and whether it was generated before or after trace fixes,
- warm-start bridge implementation,
- policy family and whether it used RNN or non-RNN APPO,
- world-model augmentation mode,
- teacher artifact and held-out trace split.

Three especially important benchmark breaks happened:

1. Live seeded evaluation was found to be nondeterministic and was demoted to diagnostic-only use.
2. The original `v2` trace regime was replaced by a corrected `tracefix_v2` deterministic trace benchmark.
3. Later `v3` and `v4 + wm_*` teacher benchmarks materially changed the representation and the teacher artifact, so they are not numerically interchangeable with the older `tracefix_v2` line.

## Benchmark Lineage

### 1. Throughput and data-generation benchmarks

These were important early because the repo first needed a practical local data engine.

- Remote ZAI dataset v1:
  - `20 games x 50 steps = 1,000 pairs`
  - roughly `10 hours` sequential
  - about `$0.22`
- Local vLLM `Qwen2.5-0.5B-Instruct`:
  - `5,000` pairs in about `30s`
  - useful for throughput, but action quality was poor
- Local `Qwen2.5-3B-Instruct` with frontier-biased fallback:
  - `1,000` samples in `8.75s`
  - balanced directional mix and better reward than earlier local runs
- Two 1-GPU replica serving path:
  - `10,000` examples in `78.76s`
  - first local corpus judged good enough to keep scaling

Interpretation:

- serving throughput stopped being the main blocker,
- data quality and action-label quality became the bottleneck.

### 2. Task-harness closed-loop metrics

These became the first trustworthy behavior metrics before learned RL matured.

Representative results from the task harness:

- `explore`, seeds `42,43,44`, `10` steps:
  - `wall_avoidance`: avg task reward `-1.17`, unique tiles `36.33`
  - `task_greedy`: avg task reward `7.27`, unique tiles `93.00`
- `survive`, seeds `42,43,44`, `15` steps:
  - `wall_avoidance`: avg task reward `-1.63`
  - `task_greedy`: avg task reward `-0.37`

Interpretation:

- shaped task rewards were immediately useful as teacher/control signals,
- the task harness became the first good local teacher,
- but it was still one-step greedy and expensive.

### 3. Live seeded policy evaluation

This was used heavily in early RL reports, then demoted.

Why it was demoted:

- repeated evaluations of the same model on the same seeds produced materially different summaries,
- BC and APPO both diverged across repeats,
- raw NLE resets were not stable enough for trusted regression gating.

Status:

- diagnostic only,
- not valid as the source of truth after the determinism bug was established.

### 4. Deterministic trace-match benchmark

This was the single biggest measurement improvement in the repo.

Important trace-benchmark regimes:

- `data/validate_v2_explore_traces.jsonl`
  - useful transitional artifact,
  - later partially superseded because trace features were fixed afterward
- `data/tracefix_v2_explore_traces.jsonl`
  - corrected trusted benchmark for the repaired `v2` line
  - BC teacher baseline: `0.6395`
  - best non-RNN APPO before teacher regularization: `0.6240`
  - best teacher-reg APPO x100: `0.6318`
- later `v3` held-out trace regime:
  - cheap BC validation around `0.6500`
- later `v4 + wm_*` held-out trace regimes:
  - offline teacher artifacts around `0.9375`
  - short scheduled-replay APPO branch reached `0.95`
  - later documents sometimes cite either `0.9375` or about `0.95` as the practical baseline because the reference artifact itself changed across runs

Interpretation:

- deterministic trace match became the trusted objective,
- it repeatedly contradicted training reward,
- it exposed the real failure mode: online drift away from the teacher.

### 5. World-model direct metrics

World-model work added a second benchmark family:

- feature MSE / MAE,
- feature cosine similarity,
- reconstruction MSE / cosine,
- action accuracy / top-3 accuracy,
- reward error,
- latent dead-fraction,
- downstream BC trace match from transformed features.

Most important later result:

- direct world-model quality improved a lot,
- but better direct predictive metrics did not automatically improve downstream teacher quality or short online RL.

### 6. Proxy-reward offline metrics

The proxy-reward branch added another offline gate:

- held-out proxy action top-1,
- multi-head proxy decomposition sanity,
- calibrated and bounded live proxy score,
- short RL trace match under proxy-mixed reward.

Best early proxy result:

- held-out proxy action top-1 reached `0.8125`,
- but short proxy-mixed RL only matched `0.9375`,
- which still did not beat the best short teacher-replay branch at `0.95`.

## How The Repo Morphed

### Stage A: Forward-model and data-generation repo

Main shape:

- `src/` modules for state encoding, data generation, evaluation, manifesting,
- `scripts/generate_training_data.py`,
- memory-augmented forward-model idea,
- LoRA / Unsloth training path.

Main question:

- can we cheaply generate useful NetHack state-action-outcome data?

### Stage B: Teacher-harness repo

Main additions:

- task-shaped rewards,
- task harness,
- one-step counterfactual control,
- teacher traces.

Main question:

- can we define meaningful tasks and build a teacher stronger than raw random or wall-avoidance behavior?

### Stage C: Real RL repo

Main additions:

- `rl/` subtree,
- Sample Factory APPO backend,
- custom NetHack APPO env,
- BC warm start,
- options / scheduler / reward plumbing.

Main question:

- can we turn the teacher into a real learned online improver?

### Stage D: Evaluation-first RL repo

Main additions:

- deterministic trace evaluation,
- checkpoint ranking by trace match,
- disagreement reports,
- trace sharding,
- best-trace checkpoint preservation.

Main question:

- can we trust any policy improvement claim at all?

### Stage E: Teacher-centric improver repo

Main additions:

- teacher regularization,
- DAgger tooling,
- behavior-regularized offline training,
- world-model feature augmentation,
- scheduled replay,
- proxy reward modules,
- repo-morph plans treating the teacher as the primary artifact.

Main question:

- which online improver class is best once the teacher and benchmark are already strong?

## Chronological Research Ledger

This section walks the markdown trail in creation order. Each entry records:

- what the document added,
- what it changed in the project’s thinking,
- and how later evidence treated it.

### Phase 0: Initial Thesis, Forward Model, and Task Framing

1. `PLAN.md` (`2026-04-05 02:13 UTC`)
   - Established the original LoRA/manifest concept: deterministic input hashing, virgin benchmarks, before/after evaluation, and a forward-model-oriented workflow.
   - Strong early conclusion: infrastructure had improved enough that data quality was now the real bottleneck.
   - Still relevant as the root statement that the project should be benchmarked, manifested, and audited rather than run casually.

2. `IMPLEMENTATION-PLAN.md` (`2026-04-05 02:13 UTC`)
   - Expanded the forward-model thesis into a deterministic, testable pipeline: state encoder, data generator, trainer, evaluator, manifest builder.
   - This is more of a build spec than a scientific result document.
   - Later partially superseded by the RL-heavy direction, but still valuable because it formalized determinism and testability as first-class design rules.

3. `references/README.md` (`2026-04-05 02:13 UTC`)
   - Anchored the literature baseline: NLE, NetPlay, AutoAscend, language-wrapper, Sample Factory, BeBold, MiniHack.
   - Important early calibration numbers appeared here: random near-zero reward, TorchBeast around `120`, language-wrapper around `730`, local `0.5B` generation fast but poor.
   - This note remained useful throughout because later design pivots repeatedly matched those external lessons.

4. `RL-HARNESS-TASKS.md` (`2026-04-05 03:27 UTC`)
   - Defined the repo’s task-centric philosophy: `explore`, `survive`, `combat`, `descend`, `resource`.
   - Formalized the idea that trajectory-level evaluation mattered more than one-step prediction alone.
   - This document held up well. Later teacher, world-model, and proxy work all continued to rely on task decomposition.

5. `HANDOFF.md` (`2026-04-05 03:36 UTC`)
   - Captured the first coherent repo state after major data-generation and forward-model work.
   - Key results:
     - remote ZAI dataset v1 was slow but workable,
     - local `0.5B` data generation was fast but low-quality,
     - `3B` + frontier-biased fallback was the first healthy local action mix,
     - replica serving beat in-process batch serving for current settings.
   - Important because it records the point where local throughput stopped being the main problem.

6. `TODO.md` (`2026-04-05 03:36 UTC`)
   - Condensed the immediate execution priorities:
     - counterfactual task-value labels,
     - reward/value models,
     - golden replay overfitting,
     - forward-model LoRA,
     - larger local corpus.
   - This remains a good snapshot of priorities before the later stronger teacher/improver framing took over.

7. `MAESTROMOTIF-INTEGRATION.md` (`2026-04-05 03:42 UTC`)
   - Imported the key hierarchical lesson from MaestroMotif:
     - use skills/options,
     - use LLMs as judges rather than low-level actors,
     - build reward models and schedulers over skills.
   - This was an early conceptual pivot away from “just optimize one scalar reward.”
   - Later repo plans repeatedly echoed this document’s hierarchy and reward-modeling ideas.

8. `APPO-OPTIONS-OVERHAUL.md` (`2026-04-05 04:05 UTC`)
   - Declared APPO the intended low-level learner and options the intended temporal abstraction.
   - Proposed the `rl/` package layout that later actually appeared.
   - Historically important because it marks the moment the repo stopped being only a forward-model and teacher-harness project and started becoming a real RL codebase.

9. `output/pipeline_adapter/checkpoint-30/README.md`, `output/pipeline_adapter/checkpoint-60/README.md`, `output/pipeline_adapter/README.md` (`2026-04-05 04:59 UTC`)
   - These are autogenerated SFT model cards, not research analyses.
   - They still matter as evidence that the original adapter/SFT pipeline ran on `unsloth/Qwen2.5-0.5B-Instruct`.
   - Analytically weak; mostly historical artifacts of the earlier forward-model training path.

### Phase 1: RL Becomes Real, But The Measurement Story Is Weak

10. `RL-APPO-HANDOFF.md` (`2026-04-05 05:10 UTC`)
    - First strong claim that the repo had a real learned RL backend:
      - real NLE env,
      - learned actor-critic,
      - Sample Factory APPO,
      - recurrence, value learning, checkpointing.
    - At the same time, it admitted that policy quality still trailed the heuristic teacher badly.
    - This document is the operational birth certificate of the `rl/` subtree.

11. `UPGRADE-PLAN.md` (`2026-04-05 05:13 UTC`)
    - Reframed the problem after the first APPO build-out:
      - no BC-to-APPO bridge,
      - observation too weak,
      - reward learning too local,
      - options too shallow.
    - Strong diagnosis that the observation encoder and lack of BC initialization were major blockers.
    - Much of this aged well; later reports confirmed both points.

12. `RUN-REPORT-2026-04-05-V2-BC-APPO.md` (`2026-04-05 05:29 UTC`)
    - Validated the first `v2` end-to-end BC-to-APPO pipeline.
    - Key results:
      - `v2` traces: `14` episodes, `273` rows,
      - BC train accuracy: `0.6630`,
      - APPO training completed over `21,504` frames at about `174.7 FPS`,
      - APPO snapshot still had very high repetition and poor task reward.
    - Most important discovery: repeated evaluation on the same seeds was nondeterministic.
    - This report changed the project by showing that evaluation correctness itself was a blocker.

13. `FAST-ITERATION-PLAN.md` (`2026-04-05 05:35 UTC`)
    - Responded to the above by explicitly rejecting long end-to-end runs as the default debugging loop.
    - Proposed cheap deterministic loops, slice analysis, DAgger, checkpoint selection by trustworthy metrics, and feature-first BC debugging.
    - This plan largely held up and became the repo’s later working style.

14. `RUN-REPORT-2026-04-05-FAST-LOOP-VALIDATION.md` (`2026-04-05 05:49 UTC`)
    - Validated the new debug commands and bounded short loop.
    - Key results:
      - fresh `v2` traces: `119` rows,
      - BC train accuracy: `0.7647`,
      - APPO `51,200` frames at about `190.7 FPS`,
      - APPO still repetitive and still below the teacher.
    - Confirmed again that live-seed evaluation remained nondeterministic.
    - This report strengthened the decision to stop trusting raw seeded eval.

15. `README.md` (`2026-04-05 06:05 UTC`)
    - Consolidated the operator story for the whole repo:
      - AutoAscend traces,
      - LLM agent,
      - forward-model pipeline,
      - trace training,
      - reward / scheduler training,
      - BC,
      - APPO,
      - evaluation.
    - It is important because it records the project’s self-conception after the first RL expansion.

### Phase 2: Deterministic Trace Evaluation Replaces Live Eval

16. `RUN-REPORT-2026-04-05-TRACEFIX-X10.md` (`2026-04-05 06:33 UTC`)
    - This is one of the most important reports in the repo.
    - It validated the corrected trace-feature generation and established the repaired deterministic `tracefix_v2` benchmark.
    - Key results:
      - corrected trace file: `258` rows,
      - BC deterministic trace match: `0.6395`,
      - x10 APPO deterministic trace match: `0.5853`,
      - APPO improved a lot relative to earlier runs but still trailed BC.
    - Historically, this report marks the shift from noisy live numbers to a stable benchmark that actually changed the project’s beliefs.

17. `FAST-DEBUG-LOOP-REPORT-2026-04-05.md` (`2026-04-05 06:44 UTC`)
    - Added confusion profiles and trace slicing.
    - Stable finding:
      - BC and APPO were not failing uniformly,
      - `east`, `south`, and especially `search` were weak,
      - APPO had a particular `south` problem,
      - both policies over-predicted `north`.
    - This sharpened the diagnosis from “collapse” to “specific directional ambiguity and drift.”

18. `RUN-REPORT-2026-04-05-X10-RERUN.md` (`2026-04-05 07:06 UTC`)
    - Re-ran x10 APPO after the fast-loop changes.
    - Operational result: stable training to `230,400` frames.
    - Scientific result: negative.
      - BC stayed at `0.6395`,
      - latest APPO fell to `0.4651`,
      - best-by-reward checkpoint fell to `0.3837`,
      - the run drifted hard toward `west`.
    - This was decisive evidence that reward improvement and teacher alignment had diverged sharply.

19. `POSTMORTEM-NEXT-STEPS-2026-04-05.md` (`2026-04-05 07:08 UTC`)
    - Integrated the rerun with the literature:
      - DAgger for student-state distribution shift,
      - Kickstarting for persistent teacher supervision,
      - BRAC for behavior-constrained improvement.
    - This is an early form of the later “improver-limited, not infra-limited” thesis.

20. `FAST-RL-UPGRADE-PLAN-2026-04-05.md` (`2026-04-05 07:09 UTC`)
    - Turned the postmortem into a ladder:
      - deterministic trace gating,
      - directional slices,
      - BC-first debugging,
      - short RL drift tests,
      - only then larger RL.
    - Key conceptual shift:
      - stop spending budget on long blind sweeps,
      - treat teacher drift as the main failure class.

21. `RUN-REPORT-2026-04-05-X100-NORNN.md` (`2026-04-05 07:36 UTC`)
    - Validated the repaired non-RNN warm-start bridge.
    - Key results:
      - step-0 repaired non-RNN APPO: `0.6047`,
      - best APPO checkpoint: `0.6240`,
      - BC baseline remained `0.6395`.
    - This was a genuine milestone:
      - the old recurrent bridge had been invalid,
      - the repaired non-RNN bridge produced the first believable APPO improvement trajectory.

22. `FAST-RL-EXECUTION-PLAN-2026-04-05.md` (`2026-04-05 07:45 UTC`)
    - Reframed the next steps after the non-RNN repair:
      - teacher drift became the main bottleneck,
      - teacher regularization, DAgger, and behavior-regularized fine-tuning became the main families of interest.
    - This is a bridge document between “repair the harness” and “repair the learning rule.”

23. `RUN-REPORT-2026-04-05-X10-VALIDATE.md` (`2026-04-05 07:57 UTC`)
    - Validated the repaired non-RNN x10 path under the trace benchmark.
    - Key results:
      - best deterministic checkpoint at `46k` frames: `0.6318`,
      - later drift back toward `~0.61`,
      - still below BC at `0.6395`.
    - Important engineering change:
      - best-by-trace checkpoint preservation became necessary because retention deleted the true best checkpoint.

### Phase 3: Teacher-Regularized RL And Alignment-Centric Thinking

24. `TEACHER-REG-APPO-PLAN.md` (`2026-04-05 08:47 UTC`)
    - Proposed the next minimal change with the best theoretical backing:
      - keep teacher supervision active during RL,
      - optionally add an episodic explore bonus.
    - The key idea here was correct; later evidence supported teacher regularization and rejected the bonus.

25. `RUN-REPORT-2026-04-05-X100-TEACHER-REG-COMPARISON.md` (`2026-04-05 09:06 UTC`)
    - Compared teacher-regularized APPO against teacher-reg plus state-hash bonus.
    - Key results:
      - teacher-reg only x100 best: `0.6318`,
      - teacher-reg + bonus best: `0.6085`,
      - BC baseline: `0.6395`.
    - Stable conclusion:
      - teacher regularization was real and useful,
      - the novelty bonus path was not helping the trusted objective.

26. `ALIGNMENT-IMPROVEMENT-PLAN.md` (`2026-04-05 09:19 UTC`)
    - Declared alignment the primary bottleneck.
    - Proposed:
      - real DAgger schedule,
      - in-training best-trace checkpoint selection,
      - richer `v3` features,
      - teacher-reg as the default online improver,
      - behavior-regularized improvement as the likely next branch if APPO stalled.
    - This plan aged well and many pieces were later implemented.

27. `BEHAVIOR-REG-IMPROVEMENT.md` (`2026-04-05 09:36 UTC`)
    - Introduced the behavior-regularized branch as a controlled alternative if teacher-reg APPO plateaued.
    - At this stage it was still mostly a design note and experimental scaffold.
    - Later behavior-reg reports showed the code path was real but not yet stronger than the best teacher line.

28. `ALIGNMENT-IMPLEMENTATION-REPORT-2026-04-05.md` (`2026-04-05 09:45 UTC`)
    - Recorded that the alignment plan had largely been implemented:
      - DAgger schedule,
      - in-training trace checkpointing,
      - `v3` features,
      - teacher-reg default baseline,
      - behavior-reg experimental trainer.
    - Key results:
      - naive DAgger harmed held-out performance,
      - `v3` BC reached `0.6500`,
      - teacher-reg APPO remained below BC,
      - behavior-reg offline reached `0.6375` held-out on the cheap run.
    - This document is the clearest snapshot of the repo becoming a mature fast-loop experiment platform.

### Phase 4: Strategic Consolidation On 2026-04-06

29. `PROJECT-STATUS-AND-NEXT-STEPS-2026-04-06.md` (`2026-04-06 03:02 UTC`)
    - Strong project-wide diagnosis:
      - offline teacher strong,
      - warm-start bridge fixed,
      - deterministic trace evaluation trustworthy enough to gate experiments,
      - online RL still drifts.
    - Important claim:
      - the project is now objective-alignment and online-improver limited, not infrastructure limited.

30. `RESEARCH-NOTES-NETHACK-HORIZONS.md` (`2026-04-06 05:08 UTC`)
    - Recalibrated horizon assumptions against the literature.
    - Stable findings:
      - rollout length `32-64` was not the main problem,
      - episode horizon and teacher traces were still short,
      - total RL budget remained tiny relative to serious NetHack systems.
    - Useful because it prevented the repo from over-focusing on the wrong timescale parameter.

31. `RESEARCH-NOTES-WORLD-MODELS-NETHACK.md` (`2026-04-06 05:11 UTC`)
    - Answered the question: how could world models help this specific repo?
    - Best early answer:
      - learned representation and short-horizon predictive structure were promising,
      - full Dreamer-style replacement was not the right first move.

32. `WORLD-MODEL-PLAN-FOR-RL-NETHACK.md` (`2026-04-06 05:12 UTC`)
    - Narrowed the world-model agenda to a practical build plan:
      - skill-conditioned latent model,
      - short-horizon predictive heads,
      - use first as encoder pretraining and auxiliary losses.
    - Later world-model validation partly supported this support-role framing.

33. `CURRENT-RL-SYSTEM.md` (`2026-04-06 05:41 UTC`)
    - Clarified what the repo actually trains:
      - forward-model SFT,
      - BC / behavior-reg teachers,
      - world-model training,
      - APPO RL.
    - Important because it prevented conceptual confusion between data generation, teacher training, world-model training, and actual online RL.

34. `wagmi.md` (`2026-04-06 06:38 UTC`)
    - This is the longest and most comprehensive decision memo in the repo.
    - Central thesis:
      - the project had crossed from infrastructure-limited to improver-limited,
      - representation and teacher preservation had improved,
      - actual online teacher-beating improvement had not happened.
    - The evidence table and experimental ledger inside `wagmi.md` are among the clearest high-level summaries of the entire project.

35. `plan-wagmi.md` (`2026-04-06 06:41 UTC`)
    - Converted the `wagmi.md` diagnosis into a multi-phase decision plan:
      - freeze baseline,
      - scheduled replay,
      - prioritized replay,
      - targeted DAgger,
      - explicit APPO go/no-go,
      - new improver branch if necessary.
    - This document is important because it formalized promotion criteria and stop conditions for future work.

36. `report-wagmi.md` (`2026-04-06 06:54 UTC`)
    - Reported on the scheduled-replay world-model branch.
    - Key interpretation:
      - scheduled replay was a real stabilizer,
      - short run hit `0.95`,
      - larger run still plateaued early and did not produce later improvement.
    - This document helped split “preservation” from “improvement” as separate phenomena.

37. `REPORT-TWO-OPTIONS-2026-04-06.md` (`2026-04-06 07:50 UTC`)
    - Compared two serious next-step families:
      - teacher-replay / on-policy distillation,
      - behavior-regularized offline-to-online RL.
    - Strategic conclusion:
      - distillation was the best incremental next step,
      - behavior-regularized offline-to-online RL looked like the better long-term answer.

38. `REPORT-BEHAVIOR-REG-2026-04-06.md` (`2026-04-06 08:21 UTC`)
    - Evaluated the first behavior-regularized branch directly.
    - Key results:
      - offline behavior-reg teachers reached `0.9000`, `0.9250`, then `0.9125` after the masked-prior fix,
      - short online continuation from these teachers did not beat the best current branch.
    - Important bug discovery:
      - the original KL regularizer ignored row-wise action masks.
    - Stable conclusion:
      - the behavior-reg path was real but weaker than the best teacher line.

39. `REPO-MORPH-PLAN-2026-04-06.md` (`2026-04-06 08:23 UTC`)
    - Proposed the strongest structural repo-shape change:
      - center the repo around teacher construction, teacher evaluation, teacher-data refinement, and interchangeable improver modules.
    - This is arguably the clearest architectural statement of where the repo wants to go next.

40. `PLAN-PROXY-REWARD-OVERHAUL-2026-04-06.md` (`2026-04-06 08:52 UTC`)
    - Proposed the proxy-reward pivot:
      - move away from hand-shaped scalar reward,
      - learn short-horizon, teacher-derived, interpretable proxy heads from trace data,
      - gate RL promotion on offline proxy quality first.
    - Important because it treated reward design as a learned, inspectable artifact rather than a scalar guess.

41. `REPORT-PROXY-REWARD-OVERHAUL-2026-04-06.md` (`2026-04-06 10:03 UTC`)
    - Reported that the proxy branch was an engineering success but not yet a promotion success.
    - Key results:
      - proxy dataset only `200` train / `80` held-out rows,
      - calibrated held-out proxy action top-1 reached `0.8125`,
      - short proxy-mixed RL could match `0.9375` but not beat `0.95`.
    - Stable lesson:
      - the proxy path is promising,
      - but current data volume and label richness are too small.

42. `REPORT-WORLD-MODEL-VALIDATION-2026-04-06.md` (`2026-04-06 10:18 UTC`)
    - This is the clearest world-model empirical report.
    - Key findings:
      - old eval surface was too weak and had a dimension mismatch bug,
      - baseline world model had poor direct action accuracy (`0.25`) but still supported a good downstream teacher,
      - retrained action-focused world models improved direct metrics dramatically,
      - but did not improve downstream teacher quality or short online RL.
    - This report sharply narrowed the role of world models in the repo.

43. `LESSONS-WORLD-MODEL-2026-04-06.md` (`2026-04-06 10:19 UTC`)
    - Condensed the world-model report into durable lessons:
      - better world-model prediction is not the same thing as better policy representation,
      - world models are support modules and diagnostics, not the main fix for online drift.
    - This is a good “what to believe now” note for future maintainers.

## What Held Up Across The Whole Trail

These conclusions remained robust despite code churn:

- task decomposition is appropriate for NetHack in this repo,
- deterministic evaluation matters more than raw run count,
- the teacher path is the repo’s strongest asset,
- BC and later behavior-regularized offline training produce better aligned policies than unconstrained online RL,
- the warm-start bridge had real bugs and fixing it mattered,
- teacher regularization helps more than generic novelty bonuses,
- the current hand-shaped reward is good for debugging but too weak as the center of the learning story,
- world-model augmentation can help representation and teacher quality, but not enough by itself to solve online drift.
- the strongest world-model contribution so far is now:
  - world-model-backed offline teacher construction,
  - followed by teacher-logit distillation into a cheap base-`v4` student,
  - and then multi-teacher compression into a deeper cheap student,
  - not direct transformer-conditioned online rollouts.

## What Was Invalidated, Demoted, Or Narrowed

- Live seeded evaluation as a benchmark:
  - demoted to diagnostic-only use after nondeterminism was demonstrated.
- Reward-best checkpoint selection:
  - repeatedly shown to disagree with the trusted trace objective.
- Generic novelty / state-hash bonus as the main missing ingredient:
  - explicitly weakened by the teacher-reg versus bonus comparison.
- “More APPO scale will probably solve it”:
  - repeatedly contradicted by x10 and x100 runs.
- “Better world-model direct metrics should unlock better online RL”:
  - contradicted by the world-model validation pass.
- “Simple masked-prior behavior regularization is the next mainline”:
  - weakened by the behavior-reg report; the path was coherent but not stronger than the best teacher line.

## Current Best Read As Of 2026-04-06

The strongest integrated interpretation is:

1. The repo can now build a strong offline teacher.
2. The repo can evaluate that teacher with a trustworthy deterministic benchmark.
3. The repo can warm-start the online learner faithfully from the teacher.
4. The online learner still does not reliably improve beyond the teacher.
5. Therefore the main bottleneck is the online improver and its objective, not basic infrastructure.

More specifically:

- the teacher pipeline is stronger than the improver pipeline,
- the benchmark pipeline is stronger than the reward pipeline,
- the world model is useful as a support module,
- the best current world-model line is:
  - `distilbert` world-model teacher at `0.9625`,
  - cheap base-`v4` student distilled from that teacher at `0.975`,
  - two-teacher ensemble over the best cheap students at `0.9875`,
  - a single deeper cheap student distilled from that ensemble at `0.9875`,
  - after fixing the remaining APPO bridge bugs, the same deeper `0.9875` student now survives at step 0 with exact `0.9875` warm-start trace match,
  - but the first learned checkpoint still drops to `0.975` and later short-run checkpoints fall further to `0.925` and `0.9125`,
  - after fixing a CLI bug that had been silently dropping zero-valued teacher coefficients, a valid no-teacher APPO control still reached the same `0.975` best learned score,
  - that means the current teacher CE/replay settings are not the main reason the short bridge reaches `0.975`,
  - small offline attempts to improve the `0.9875` teacher with a supervised term, east-weighted supervision, or a wider pure-distill student all regressed below the baseline,
- the proxy-reward branch is implemented but not mature enough to replace the current best teacher-replay branch,
- prompt-conditioned BC teachers are now implemented with both stable-hash and frozen-transformer text encoders,
  - under teacher distillation, both tied the current best numeric teacher at `0.9875`,
  - neither fixed the remaining held-out `east -> south` miss,
  - a prompt-conditioned frozen-`distilbert` teacher trained directly on supervised trace labels regressed to `0.95`,
  - so missing prompt text is not the leading explanation for the last gap in the current teacher family,
- larger offline BC scaling is now implemented with explicit GPU training and deterministic held-out checkpoint selection,
  - scaling to `2048 x 4` MLP teachers on both a clean `680`-row set and a larger `1400`-row merged set could recover `0.9875`,
  - but only when selecting checkpoints by held-out trace match,
  - the same scaled runs regressed to `0.975` or worse at late epochs if judged by final weights alone,
  - so larger models and longer offline runs are viable, but only under benchmark-aware checkpoint selection,
  - and scale by itself still did not beat the `0.9875` cheap-teacher baseline,
- exact step-0 hard-case mining confirmed that the remaining `0.9875 -> 1.0` gap is a real slice-coverage problem,
  - existing train sets had zero rows for the held-out local geometry `north=monster_*`, `south=floor`, `east=monster_*`, `west=floor`,
  - an off-heldout miner over fresh seeds found only `3` exact rows, with action split `east=1`, `south=2`,
  - augmenting the base train set with the single off-heldout `east` example was enough to fix the original held-out `east` miss,
  - but every such augmentation simply moved the lone benchmark error to a different row, yielding `0.9875` again,
  - so the current best explanation is not “the model cannot represent the fix,” but “the repo lacks enough breadth in that failure-family slice,”
- broader reset-slice mining strengthened and then narrowed that diagnosis,
  - a new reset-slice miner over `30k` fresh seeds found `25` off-heldout rows for the looser geometry `north=monster_*`, `south=floor`, `east=monster_*`, `west=floor`,
  - but their teacher-action mix was `north=13`, `east=7`, `south=5`, which already showed that adjacency alone was not enough context,
  - held-out-selected teachers trained on that broader slice, including mixed distill variants, all regressed to `0.975`,
  - the exact held-out species pair `north=monster_f`, `east=monster_o` appeared only once in `30k` fresh seeds, and even oversampling that exact off-heldout `east` row still regressed to `0.975`,
  - so the last `0.9875 -> 1.0` gap is now best understood as too thin and too branch-specific for simple hard-case oversampling to be a reliable mainline teacher-improvement strategy,
- a later audit found a real improver bug in the online branch:
  - teacher CE / KL had been using unmasked teacher and student logits, even though this repo’s low-level control is action-mask-sensitive,
  - after making teacher regularization mask-aware, the comparable short APPO probe improved from a best learned checkpoint of `0.975` back up to `0.9875`,
  - but a longer `4k`-step cheap gate still tied the teacher only briefly and then drifted to `0.8875`,
  - so the mask-aware fix is a real stabilizer, not yet a teacher-beating improver,
- the first teacher-as-base fallback probes narrowed the next online direction:
  - a new rollout-time teacher prior / fallback path is now implemented directly in the APPO actor path,
  - fallback to the auxiliary distillation teachers tied the teacher briefly and then collapsed badly, down to `0.7625` and `0.7125` on the retained late checkpoints,
  - fallback to the exact trusted `0.9875` teacher did not improve the best learned checkpoint, which fell to `0.975`,
  - but it substantially improved late stability, with retained late checkpoints at `0.9625` and `0.95` instead of the previous `0.8875` / `0.8875`,
  - this means "teacher as rollout-time base" is a real stabilizer only when the fallback base is the exact trusted teacher artifact,
  - but raw confidence fallback alone is still preservation-oriented and not yet a real improver,
- splitting the rollout fallback base from the supervision teacher sharpened that conclusion:
  - a new `teacher_prior_bc_path` now allows the exact trusted `0.9875` teacher to remain the rollout fallback base while separate auxiliary teachers drive replay / CE supervision,
  - under that split-base configuration, the short `4k` probe recovered a best learned checkpoint of `0.9875` at `512`,
  - more importantly, a retained late checkpoint at `4352` also matched `0.9875`, which is better than both the no-fallback mask-aware branch and the single-teacher fallback branch,
  - but the final retained checkpoint still fell to `0.975`, and no learned checkpoint exceeded the trusted teacher,
  - so decoupled teacher base plus auxiliary supervision is the strongest stabilizer variant yet, but it is still not a teacher-beating improver,
- a follow-up disagreement-only gate weakened that line again:
  - a new `teacher_policy_disagreement_margin` gate was added so the student could only keep a disagreeing override when it beat the teacher-preferred action by a minimum probability margin,
  - with confidence fallback disabled and disagreement-margin gating alone enabled, the short split-base probe still tied `0.9875`, but only at `1024`,
  - retained late checkpoints then collapsed to `0.7875` and `0.7625`,
  - so disagreement-only gating is strong negative evidence: the confidence-based teacher anchor was doing essential stabilization work, and replacing it with a weaker disagreement-only rule makes the branch much worse,
- a direct teacher-base blend probe also failed:
  - the improver-report path now records top-level best/final trace metadata and explicit teacher-policy config, which closes a real reporting gap for constrained improver branches,
  - using that path, a split-base blend probe with `teacher_policy_blend_coef=0.15` and confidence fallback `0.55` still tied `0.9875` at `512`,
  - but retained late checkpoints fell to `0.8125` and `0.7875`,
  - so simple probability blending is not the right residual parameterization either; it is materially worse than the plain split-base confidence-fallback branch,
- a follow-up logit-residual probe also fell short:
  - a new `teacher_policy_logit_residual_scale` path now supports true logit-space interpolation `teacher + s * (student - teacher)`, which is a cleaner teacher-base residual than probability blending,
  - using `s=0.3` with the same split-base confidence fallback still tied `0.9875`, but only at `256`,
  - retained late and final checkpoints both settled at `0.9625`,
  - so logit-space residual mixing is substantially healthier than probability blending, but it is still weaker than the plain split-base confidence-fallback baseline and is not yet a real improver,
- a replay audit plus targeted replay-weight probe sharpened the next bottleneck:
  - the current trusted replay source `/tmp/x100_v4_train_traces.jsonl` has only `200` rows and no `behavior_action`, loop-risk, or failure annotations,
  - that means current replay modes like `disagreement` and `mixed` are much weaker than they look; on this dataset they mostly collapse to static teacher-action weighting,
  - a new `teacher_replay_action_boosts` path was added and tested with `east=2.0,south=2.0`, because the best split-base fallback branch only drifts on a tiny held-out `east <-> south` confusion pair,
  - that branch still only tied `0.9875`, now at `1280`, and then fell to `0.9625` on both retained late checkpoints,
  - so static teacher-action replay weighting is not enough; the next replay / DAgger gains need richer relabeled student-state data, not just different weights on the small teacher-only replay file,
- selective DAgger infrastructure is now in place, but the first probes were strongly negative:
  - `dagger_row_policy` and `dagger_keep_match_ratio` now let the repo filter relabeled student rows before merging them back into BC retraining,
  - relabeling `32 x 40` student steps from the drifted split-base final checkpoint and retraining against the trusted teacher did not help,
  - `hard_only` collapsed back to full-state DAgger because every harvested row was flagged as loop-risk, then fell to `0.8625`,
  - `disagreement` kept only `221 / 1280` rows and still fell to `0.85`,
  - so targeted DAgger remains a live direction in principle, but the current row flags are not yet selective enough to support it; future DAgger work needs stricter failure-family triggers rather than broad student-rollout relabeling,
- a follow-up exact confusion-pair probe made the DAgger story sharper:
  - a new `dagger_confusion_pairs` selector now supports explicit `behavior_action -> teacher_action` filtering on relabeled student rows,
  - broad disagreement harvests were mostly irrelevant confusions like `east -> west` and `north -> south`; the exact held-out `east <-> south` family was present but rare,
  - harvesting `128 x 40` student steps and keeping only exact `east->south,south->east` disagreement rows produced `144` selected rows and recovered to `0.975` held-out trace match,
  - that is much safer than broad DAgger, but it still does not beat the trusted `0.9875` teacher and the remaining held-out errors are still `east -> south`,
  - so exact student-harvested failure-family relabeling is a better selector than broad disagreement, but still not sufficient by itself; the next data-refinement step likely needs offline hard-case teacher data rather than more generic online-rollout relabeling,
- the most plausible next frontier is a more teacher-aware and behavior-constrained online improver.

## Practical Research Rules Going Forward

Future updates to this document should preserve a few rules that the markdown trail learned the hard way:

1. Every new experiment must name the exact teacher artifact and held-out trace split it uses.
2. Every serious branch should report:
   - step-0 teacher clone,
   - best learned checkpoint,
   - final checkpoint.
3. No large run should be promoted unless a short run beats the current baseline on the trusted benchmark.
4. World-model changes should be promoted only if they improve downstream teacher quality first.
5. Proxy-reward changes should be promoted only if they improve offline reranking and then short-run RL before any large-scale run.
6. Live seeded eval can support diagnosis, but not promotion decisions.

## Suggested Structure For Future Additions

To keep this document useful as a long-lived thesis, future additions should append a new dated section with:

- repo state at the time of the run,
- benchmark regime and why it is comparable or not comparable to earlier ones,
- exact artifact names,
- what changed in code,
- what changed in belief,
- whether the result is:
  - infrastructure progress,
  - measurement progress,
  - teacher progress,
  - improver progress,
  - or a dead end.

## Bottom Line

The repo’s markdown history shows a real scientific progression, not just scattered note-taking.

The progression is:

- from forward-model and data-generation infrastructure,
- to task-harness teacher design,
- to real APPO-based RL,
- to deterministic trace-based evaluation,
- to teacher-centered offline training,
- to world-model and proxy-reward support modules,
- and finally to a cleaner conclusion:
  - the strongest remaining problem is constrained online improvement beyond the teacher.

That is a much narrower and more valuable problem than the repo had at the start.
