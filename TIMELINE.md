# Estimated Timeline for MDN-C-CS

## Phase 1: Setup (1-2 days)

| Activity | Estimated Time | Status |
|----------|----------------|--------|
| Activate UNIMORE HPC account | 1 day | Pending |
| Install Python and create virtual environment | 1 hour | Pending |
| Install dependencies | 2-3 hours | Pending |
| Configure SSH and GPU access | 1 hour | Pending |
| Run a small model smoke test | 1 hour | Pending |

## Phase 2: Benchmark Preparation (1 day)

| Activity | Estimated Time | Status |
|----------|----------------|--------|
| Generate 100 trap questions | 2-3 hours | Pending |
| Generate synthetic test images | 1 hour | Pending |
| Validate benchmark format | 30 min | Pending |
| Optionally add MSCOCO or Visual Genome images | 2-4 hours | Optional |

## Phase 3: Experiments (1-2 days)

| Activity | Estimated Time | Status |
|----------|----------------|--------|
| Evaluate LLaVA-v1.5 on the benchmark | 30-90 min | Pending |
| Evaluate Qwen3-VL on the benchmark | 30-90 min | Pending |
| Run baseline vs strict-prompt mitigation | 1-3 hours | Pending |
| Save JSON outputs and metrics | 15 min | Pending |

## Phase 4: Analysis and Plots (1 day)

| Activity | Estimated Time | Status |
|----------|----------------|--------|
| Compute accuracy, hallucination rate, and POPE-style metrics | 30 min | Pending |
| Generate plots with matplotlib/seaborn | 1 hour | Pending |
| Analyze condition-level error patterns | 1-2 hours | Pending |

## Phase 5: Report Writing (2-3 days)

| Activity | Estimated Time | Status |
|----------|----------------|--------|
| Write abstract and introduction | 1 day | Pending |
| Document benchmark and methodology | 1 day | Pending |
| Present model results | 1 day | Pending |
| Discuss mitigation findings | 1 day | Pending |
| Final review and editing | 1 day | Pending |

## Phase 6: Submission (1 day)

| Activity | Estimated Time | Status |
|----------|----------------|--------|
| Export final PDF | 1 hour | Pending |
| Prepare code for submission | 1 hour | Pending |
| Push to GitHub if required | 30 min | Pending |

## Total Time Summary

| Phase | Time |
|-------|------|
| Setup | 1-2 days |
| Benchmark | 1 day |
| Experiments | 1-2 days |
| Analysis | 1 day |
| Report | 2-3 days |
| Submission | 1 day |
| **Total** | **6-10 days** |

## Acceleration Tips

1. Write the report background while GPU jobs are queued.
2. Use the provided report template to save formatting time.
3. Run small smoke tests before launching full cluster jobs.
4. Keep baseline and mitigation outputs in separate result folders.

## Suggested Milestones

- **Day 3**: Setup complete and benchmark generation working.
- **Day 5**: Benchmark complete and first model results available.
- **Day 7**: Analysis complete and report writing started.
- **Day 9**: Report draft complete.
- **Day 10**: Final review and submission.

Note: These estimates assume GPU access is already active. Add 1-2 days if the HPC account still needs to be activated.
