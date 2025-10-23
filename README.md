# TinyStories Language Model

A Transformer-based language model for conditional story generation with tag-based control, trained on the TinyStories dataset.

## Overview

Custom autoregressive-style model using Transformer Encoder layers to generate children's stories conditioned on narrative tags.

**Model Statistics:**
- Parameters: ~35 million
- Architecture: 12 Transformer Encoder layers
- Context Length: 50 tokens
- Vocabulary: 50,257 tokens (GPT-2 tokenizer + special tokens)

## Supported Tags

- `BadEnding` - Stories with unexpected or unhappy conclusions
- `Conflict` - Stories involving challenges or problems
- `Dialogue` - Stories with character conversations
- `Foreshadowing` - Stories with hints about future events
- `MoralValue` - Stories teaching life lessons
- `Twist` - Stories with surprising plot turns

## Performance

**Quantitative Results:**
- Average Cross-Entropy: ~350 nats/story
- Training: 90,000 stories
- Validation: 10,000 stories

**Qualitative Analysis:**

| Metric | Approach 1 | Approach 2 |
|--------|-----------|-----------|
| Tag Adherence | Low | High |
| Creativity | High | Medium |
| Hallucinations | High | Medium |

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- `torch>=2.0.0`
- `numpy>=1.24.0`
- `transformers>=4.30.0`
- `datasets==3.6.0`
- `torchinfo>=1.8.0`

## Usage

### Generate a Story

```python
from tinystories_model import generate_text, model

# Generate a story with specific tags
generate_text(model, tags=["MoralValue", "Twist"])
```

### Example Output

```
Generating story with ['MoralValue', 'Twist']
Once upon a time, there was a little boy named Tim. Tim was a very tired boy. 
Every day, he would sit down and dream. He would dream about the sea and the waves. 
His mom and dad were very proud of him. One day, Tim saw a big fish in the water. 
He wanted to watch it. He put the fish in a cage and watched it go into the water. 
The fish swam fast, but Tim was happy. But then, something unexpected happened. 
The big fish turned into a fairy! The fairy thanked Tim for helping. She wished 
for the whole swamp to be beautiful again. Tim learned that being persistent and 
good can be good, even with a big fish. And the fairy flew away, as a thank you 
for the pretty swamp.
```

### Train Your Own Model

```python
# Training configuration:
# - Epochs: 10
# - Optimizer: Adam (lr=0.0001)
# - Context Length: 50 tokens
# - Batch Size: 1

python tinystories_model.py
```

## Model Architecture

```
Embedding (vocab_size=50257, emb_dim=256)
LearnedPositionalEmbedding (max_length=50, dim=256)

TransformerEncoder (3 layers, d_model=256, nhead=8, dim_feedforward=768)
Dropout (0.1)

TransformerEncoder (3 layers, d_model=256, nhead=8, dim_feedforward=1024)
Dropout (0.1)

TransformerEncoder (3 layers, d_model=256, nhead=8, dim_feedforward=1024)
Dropout (0.1)

TransformerEncoder (3 layers, d_model=256, nhead=8, dim_feedforward=768)
Dropout (0.1)

SelectLastToken()
Linear (256 -> vocab_size)
```

## Dataset

Model trained on [TinyStories-GPT4 dataset](https://huggingface.co/datasets/skeskinen/TinyStories-GPT4):
- Total Stories: 2,745,100
- Training Subset: 90,000 stories
- Average Story Length: 197.97 tokens

## Technical Details

**Tokenization:** GPT-2 BPE with special tokens `[PAD]`, `[BOS]`, `[EOS]`, `[TAG_1]` through `[TAG_6]`

**Training Strategy:** Sliding window of 50 tokens over each story with two tag insertion approaches tested

**Text Generation:** Top-p sampling (p=0.9) with dynamic EOS bias after 200 tokens, max length 350 tokens

## License

[Choose your license]

## Acknowledgments

- TinyStories dataset by Eldan & Li (Microsoft Research)
- Hugging Face Transformers library
- GPT-2 tokenizer by OpenAI
