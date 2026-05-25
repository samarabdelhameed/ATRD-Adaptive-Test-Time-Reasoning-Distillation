# 📚 Incremental Training Guide

## Overview
This guide explains how to add new training examples and retrain the model without losing previous progress.

---

## 🎯 Strategy 1: Incremental Training (Recommended)

### Step 1: Prepare New Examples

Create a new file with your examples:

```bash
# File: data/new_examples.jsonl
```

Example format:
```json
{"question": "Your new question", "thinking_trace": "<<thinking>>\nYour reasoning...\n</thinking>>", "answer": "\\boxed{answer}", "_source": "custom"}
```

### Step 2: Merge with Existing Dataset

```bash
# Backup original dataset
cp data/final_train_dataset.jsonl data/final_train_dataset.backup.jsonl

# Append new examples
cat data/new_examples.jsonl >> data/final_train_dataset.jsonl

# Verify
wc -l data/final_train_dataset.jsonl
# Should show: 2572 + (number of new examples)
```

### Step 3: Update Statistics

```bash
python3 << 'EOF'
import json

# Count new total
with open('data/final_train_dataset.jsonl') as f:
    total = sum(1 for line in f if line.strip())

# Update p1_stats.json
with open('data/p1_stats.json', 'r') as f:
    stats = json.load(f)

stats['final_total'] = total

with open('data/p1_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"Updated total: {total} examples")
EOF
```

### Step 4: Upload to Kaggle

```bash
# Create a new dataset version on Kaggle
kaggle datasets version -p data/ -m "Added X new examples for [topic]"
```

### Step 5: Run Incremental Training

In your Kaggle notebook:

```python
# Load the previous checkpoint
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "nvidia/Nemotron-3-Nano-30B",
    quantization_config=bnb_config
)

# Load previous LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "/kaggle/input/your-previous-checkpoint/sft/final_adapter"
)

# Continue training with new data
trainer = SFTTrainer(
    model=model,
    train_dataset=new_combined_dataset,
    # ... other params
)

trainer.train()  # Continues from previous weights
```

---

## 🎯 Strategy 2: Full Retraining

### When to Use:
- You have many new examples (>500)
- You want to rebalance the dataset
- You have time and GPU credits

### Steps:

```bash
# 1. Merge datasets
cat data/final_train_dataset.jsonl data/new_examples.jsonl > data/combined_dataset.jsonl

# 2. Shuffle for better training
shuf data/combined_dataset.jsonl > data/final_train_dataset_v2.jsonl

# 3. Upload to Kaggle
kaggle datasets version -p data/ -m "Full dataset v2 with new examples"

# 4. Run full training (12 hours)
# Use the same notebook as before, but with new dataset
```

---

## 🎯 Strategy 3: Fine-tuning on New Data

### When to Use:
- You have few new examples (<100)
- You want quick iteration
- You're testing new example types

### Steps:

```python
# In Kaggle notebook

# 1. Load previous model
model = PeftModel.from_pretrained(
    base_model,
    "/kaggle/input/previous-checkpoint/sft/final_adapter"
)

# 2. Train ONLY on new examples
new_dataset = load_dataset("json", data_files="new_examples.jsonl")

trainer = SFTTrainer(
    model=model,
    train_dataset=new_dataset,
    num_train_epochs=1,  # Just 1 epoch!
    # ... other params
)

trainer.train()
```

---

## 📊 Comparison Table

| Strategy | Time | GPU Cost | Quality | Use Case |
|----------|------|----------|---------|----------|
| **Incremental** | 6-8h | Medium | Best | Adding 100-500 examples |
| **Full Retrain** | 12h | High | Excellent | Major dataset changes |
| **Fine-tune** | 2-3h | Low | Good | Quick iterations |

---

## ⚠️ Important Notes

### 1. Backup Your Checkpoints!

```bash
# Before any training, backup:
cp -r checkpoints/sft checkpoints/sft_backup_$(date +%Y%m%d)
```

### 2. Version Your Datasets

```bash
# Keep track of dataset versions
data/
  final_train_dataset_v1.jsonl  # Original (2,572 examples)
  final_train_dataset_v2.jsonl  # + 100 new examples
  final_train_dataset_v3.jsonl  # + 200 more examples
```

### 3. Monitor for Catastrophic Forgetting

After training, test on old examples:

```python
# Test on original examples
old_examples = load_dataset("json", data_files="final_train_dataset_v1.jsonl")
test_accuracy = evaluate_model(model, old_examples)

if test_accuracy < 0.8:
    print("⚠️ Warning: Model forgot old examples!")
```

---

## 🔧 Practical Example

### Scenario: Adding 50 Discrete Math Examples

```bash
# 1. Create new examples
cat > data/discrete_math_examples.jsonl << 'EOF'
{"question": "Prove by induction that 1+2+...+n = n(n+1)/2", "thinking_trace": "...", "answer": "\\boxed{...}", "_source": "discrete_math"}
{"question": "Find the number of subsets of a set with n elements", "thinking_trace": "...", "answer": "\\boxed{2^n}", "_source": "discrete_math"}
# ... 48 more examples
EOF

# 2. Merge with existing
cat data/discrete_math_examples.jsonl >> data/final_train_dataset.jsonl

# 3. Update stats
python3 scripts/update_stats.py

# 4. Upload to Kaggle
kaggle datasets version -p data/ -m "Added 50 discrete math examples"

# 5. Run incremental training (6 hours)
# Use Strategy 1 approach in Kaggle notebook
```

---

## 📈 Expected Results

### After Incremental Training:

- **Old examples**: 95%+ accuracy (maintained)
- **New examples**: 90%+ accuracy (learned)
- **Training time**: 6-8 hours
- **Total examples**: 2,572 + 50 = 2,622

---

## 🚀 Quick Start Commands

```bash
# Add new examples
cat new_examples.jsonl >> data/final_train_dataset.jsonl

# Update stats
python3 << 'EOF'
import json
with open('data/final_train_dataset.jsonl') as f:
    total = sum(1 for _ in f)
with open('data/p1_stats.json') as f:
    stats = json.load(f)
stats['final_total'] = total
with open('data/p1_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print(f"Total: {total}")
EOF

# Upload to Kaggle
kaggle datasets version -p data/ -m "Added new examples"

# Run training on Kaggle
# (Use your existing notebook with updated dataset)
```

---

## ✅ Checklist

Before starting incremental training:

- [ ] Backup current checkpoint
- [ ] Backup current dataset
- [ ] Prepare new examples in correct format
- [ ] Verify new examples have `<<thinking>>` traces
- [ ] Update `p1_stats.json`
- [ ] Upload to Kaggle
- [ ] Test on sample before full training
- [ ] Monitor training logs
- [ ] Evaluate on both old and new examples

---

## 📞 Troubleshooting

### Problem: Model forgot old examples

**Solution**: Use Strategy 2 (Full Retraining) with shuffled data

### Problem: Training too slow

**Solution**: Use Strategy 3 (Fine-tuning) with fewer epochs

### Problem: New examples not learning

**Solution**: Increase learning rate slightly (2e-4 → 3e-4)

---

**Date**: 2026-05-25  
**Project**: ATRD  
**Author**: Samar
