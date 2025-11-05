# Items You Need to Complete

This document lists all the items marked with **[TODO]** in REPORT.md that require your input.

## Personal Information
- [ ] Your full name
- [ ] Your student ID
- [ ] Your HKUST email

---

## Part 1: Single-turn Loss Masking
- [ ] Screenshot showing validation passed (run `bash scripts/check_exercises_1.sh`)

## Part 2: Multi-turn Loss Masking
- [ ] Screenshot showing validation passed (run `bash scripts/check_exercises_2.sh`)

## Part 3: Reverse Loss Masking - Single-turn
- [ ] Screenshot showing validation passed (run `bash scripts/check_exercises_3.sh`)

## Part 4: Reverse Loss Masking - Multi-turn
- [ ] Screenshot showing validation passed (run `bash scripts/check_exercises_4.sh`)

## Part 5: Cross-Entropy Loss
- [ ] Screenshot showing all test cases passed (run `python loss_functions_checker.py`)

---

## Part 6: Supervised Fine-Tuning

**Before you can complete this section, you need to:**
1. Run `bash setup.sh` to install dependencies and download the model
2. Run `bash scripts/sft.sh` to train the model

**After training, provide:**
- [ ] GPU Type used
- [ ] Batch size per device (BSZPERDEV)
- [ ] Gradient accumulation steps
- [ ] Effective batch size (TOTALBSZ)
- [ ] Learning rate
- [ ] Number of epochs
- [ ] Total training steps
- [ ] Max sequence length
- [ ] Warmup ratio
- [ ] LR scheduler type
- [ ] Screenshot of training loss curve (from W&B or console logs)
- [ ] Final checkpoint path

---

## Part 7: IFEval Evaluation & Hyperparameter Tuning

**Before you can complete this section:**
1. Evaluate base model: `cd ifeval && bash run.sh SmolLM2-135M results/SmolLM2-135M`
2. Evaluate fine-tuned model: `bash run.sh /path/to/checkpoint results/finetuned`
3. If accuracy < 22%, tune hyperparameters and re-train
4. Document all experiments

**After evaluation, provide:**

### Performance Table
- [ ] Base model strict accuracy
- [ ] Base model loose accuracy
- [ ] Fine-tuned model strict accuracy (must be > 22%)
- [ ] Fine-tuned model loose accuracy
- [ ] Improvement percentages

### Hyperparameter Tuning Table
For each experiment, record:
- [ ] Learning rate
- [ ] Number of epochs
- [ ] Batch size
- [ ] Warmup ratio
- [ ] Scheduler type
- [ ] Final training loss
- [ ] Strict accuracy
- [ ] Loose accuracy
- [ ] Notes about the experiment

### Example Outputs (2-3 examples)
For each example:
- [ ] IFEval prompt
- [ ] Base model output
- [ ] Fine-tuned model output
- [ ] Analysis of what improved

### Analysis Sections
- [ ] Which instruction-following capabilities improved?
- [ ] Instruction types where the model still struggles?
- [ ] Why certain hyperparameters helped?
  - Learning rate impact analysis
  - Epoch count impact analysis
  - Batch size impact analysis
  - LR scheduler impact analysis
- [ ] Overall insights from tuning process

### Conclusion
- [ ] Final strict accuracy achieved
- [ ] Key learnings from the assignment

---

## Quick Commands Reference

```bash
# Validation (Parts 1-5)
bash scripts/check_exercises_1.sh
bash scripts/check_exercises_2.sh
bash scripts/check_exercises_3.sh
bash scripts/check_exercises_4.sh
python loss_functions_checker.py

# Training (Part 6)
bash setup.sh  # Run first to setup environment
bash scripts/sft.sh

# Evaluation (Part 7)
cd ifeval
bash run.sh SmolLM2-135M results/base
bash run.sh /path/to/your/checkpoint results/finetuned
```

---

## Tips for Hyperparameter Tuning

If your initial model doesn't achieve > 22% strict accuracy, try:

1. **Learning Rate:** Try 1e-5, 2e-5, 5e-5
2. **Epochs:** Try 3, 4, 5 epochs
3. **Batch Size:** Try different TOTALBSZ values (64, 128, 256)
4. **Warmup:** Try different warmup ratios (0.05, 0.1, 0.15)
5. **Scheduler:** Try "cosine", "linear", "constant"

Document EVERY experiment you run - you need this for the report!
