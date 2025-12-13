# Qwen2.5-Coder-14B Fine-tuning on Code Benchmarks

Fine-tuning Qwen2.5-Coder-14B model on HumanEval and MBPP datasets using LoRA and Unsloth for efficient Python code generation.

## Overview

- **Model**: Qwen2.5-Coder-14B-Instruct (4-bit quantized)
- **Method**: LoRA (Low-Rank Adaptation)
- **Framework**: Unsloth
- **Datasets**: HumanEval (164 problems) + MBPP (374 train, 90 test)
- **Hardware**: 14GB GPU (Google Colab T4)

## Installation

```bash
# Dependencies auto-installed in notebook
pip install unsloth transformers==4.56.2 trl==0.22.2 datasets==4.3.0
```

## Configuration

### Model Settings
```python
MODEL_NAME = "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
```

### LoRA Parameters
```python
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

### Training Hyperparameters
```python
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 5
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 8
MAX_GRAD_NORM = 0.3
WARMUP_RATIO = 0.03
```

## Dataset Format

```
<|im_start|>system
You are an expert Python programmer. Write clean, efficient, and correct code.<|im_end|>
<|im_start|>user
[Problem description]<|im_end|>
<|im_start|>assistant
[Solution code]<|im_end|>
```

## Usage

### 1. Run All Sections in Order

**Section 0**: Install dependencies (run once)

**Section 1**: Test base model (optional, for comparison)

**Section 2-3**: Setup environment and mount Google Drive

**Section 4**: Load and format datasets

**Section 5**: Load model with LoRA adapters

**Section 6**: Train model (~2-3 hours on T4)

**Section 7**: Save model to local and Google Drive

**Section 8**: Test fine-tuned model

### 2. Quick Start

```python
# Just run all cells sequentially
# The notebook handles everything automatically
```

## Training Details

- **Total Steps**: ~230 steps (5 epochs)
- **Evaluation**: Every 100 steps
- **Checkpointing**: Saves top 3 models
- **Early Stopping**: Patience = 5
- **Memory Usage**: ~12-13 GB VRAM

## Output Structure

```
./qwen-coder-checkpoints/          # Training checkpoints
./qwen-coder-final/                # Final trained model
/content/drive/MyDrive/qwen-coder-humaneval/  # Google Drive backup
```

## Test Examples

The notebook tests 5 scenarios:

1. Sum of even numbers in list
2. Palindrome checker
3. First non-repeating character
4. Merge two sorted lists
5. Prime number checker

## Memory Requirements

- **Minimum**: 14 GB GPU
- **Recommended**: Google Colab Pro with T4
- **Base model**: ~8 GB
- **During training**: ~13 GB

## Troubleshooting

### Out of Memory
```python
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
```

### Training Too Slow
- Check GPU is active (not CPU)
- Verify T4 allocation in Colab

### Dataset Load Fails
- HumanEval is optional (continues with MBPP only)
- Check internet connection

## Key Features

✅ Memory efficient (4-bit + LoRA)  
✅ Fast training (Unsloth optimized)  
✅ Auto-save to Google Drive  
✅ Early stopping enabled  
✅ Pre/post training tests  
✅ Best model selection  

## Important Notes

1. Use Google Colab Pro for reliable T4 access
2. Don't close browser during training
3. Ensure 10GB free space on Google Drive
4. Compare Section 1 (base) vs Section 8 (fine-tuned) outputs

## Expected Results

After fine-tuning:
- Better code structure
- Improved problem understanding
- Cleaner solutions
- Better edge case handling

## License

- Qwen2.5-Coder: Apache 2.0
- HumanEval: MIT
- MBPP: Apache 2.0

## Credits

- Unsloth for optimized training
- Qwen Team for base model
- OpenAI for HumanEval
- Google Research for MBPP