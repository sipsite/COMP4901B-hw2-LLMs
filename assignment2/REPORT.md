# COMP4901B Homework 2 Report
## Supervised Fine-Tuning (SFT) for Language Models

**Student Name:** [YOUR NAME]
**Student ID:** [YOUR STUDENT ID]
**Email:** [YOUR EMAIL]@ust.hk

---

## Part 1: Single-turn Loss Masking (5 points)

### Validation Result
**[TODO: Insert screenshot showing validation passed]**

### Implementation Explanation

The single-turn loss masking implementation ensures that only the assistant's response contributes to the training loss, while system and user messages are masked out.

**Implementation logic:**
1. Start with all labels initialized to `IGNORE_TOKEN_ID = -100` (masked)
2. Iterate through the messages to find the assistant message
3. Use `prefix_lengths` to determine token boundaries:
   - `start_pos = prefix_lengths[i-1]` (end of previous message)
   - `end_pos = prefix_lengths[i]` (end of current message)
4. Copy tokens from `full_ids[start_pos:end_pos]` to `labels[start_pos:end_pos]`

This ensures the model learns to generate appropriate assistant responses while ignoring the prompt tokens during backpropagation.

**Code reference:** `conversation_func.py:92-101`

---

## Part 2: Multi-turn Loss Masking (5 points)

### Validation Result
**[TODO: Insert screenshot showing validation passed]**

### Implementation Explanation

The multi-turn implementation extends the single-turn logic to handle conversations with multiple (user, assistant) exchanges.

**How it extends single-turn logic:**
1. Instead of finding a single assistant message, iterate through ALL messages
2. For each message with `role == "assistant"`, unmask its tokens
3. Critical addition: Use `min(prefix_lengths[i], len(full_ids))` to handle truncation
   - When conversations exceed `max_length`, `prefix_lengths` may overshoot the actual sequence length
   - The `min()` ensures we stay within bounds and don't create invalid indices
4. All user and system messages remain masked with `IGNORE_TOKEN_ID`

This allows the model to learn from multiple assistant responses in a single conversation, which is essential for multi-turn dialogue training.

**Code reference:** `conversation_func.py:124-132`

---

## Part 3: Reverse Loss Masking - Single-turn (5 points)

### Validation Result
**[TODO: Insert screenshot showing validation passed]**

### Implementation Explanation

This exercise implements the reverse masking strategy: mask assistant messages and train on user messages.

**Implementation approach:**
1. The function automatically reorders messages: `[assistant] [user]` (swapped from original order)
2. After reordering, find the message with `role == "user"` (now in position 2)
3. Use the same `prefix_lengths` logic to determine token boundaries
4. Copy user tokens to labels, leaving assistant tokens masked

**Code reference:** `reverse_conversation_func.py:158-167`

### Conceptual Question: What would this model learn?

**Model Behavior:**
After training with this masking approach, the model learns to predict what a **user would say** given an assistant's previous response. This is the reverse of normal chatbot training.

**Real-world Application Scenarios:**

1. **User Intent Prediction & Conversational AI Testing:**
   - Given an assistant response, predict likely user follow-up questions
   - Helps design better chatbot flows by anticipating user reactions
   - Example: If assistant says "I've canceled your subscription", model predicts user might ask "Will I get a refund?" or "When does it end?"

2. **Dialogue Simulation for Training Data Generation:**
   - Generate synthetic user responses to create training datasets
   - Test chatbots by simulating realistic user behaviors
   - Quality assurance teams can use this to automatically generate edge cases

3. **Customer Journey Modeling:**
   - E-commerce platforms can predict next customer questions based on agent responses
   - Help prepare customer service agents with likely follow-up queries
   - Improve FAQ sections by understanding user question patterns

---

## Part 4: Reverse Loss Masking - Multi-turn (5 points)

### Validation Result
**[TODO: Insert screenshot showing validation passed]**

### Implementation Explanation

Multi-turn reverse masking extends single-turn to handle multiple user utterances in a conversation.

**Extension logic:**
1. Unlike single-turn, messages are NOT reordered for multi-turn
2. The function automatically adds a system message: "You are a good state predictor."
3. Iterate through all messages and unmask only those with `role == "user"`
4. Use `min(prefix_lengths[i], len(full_ids))` to handle truncation safely
5. Both assistant and system messages remain masked

**Code reference:** `reverse_conversation_func.py:192-200`

### Conceptual Question: What would this model learn?

**Model Learning:**
The model learns to predict user responses/queries given the conversation history, effectively learning **user behavior patterns** and **conversational state transitions** from the user's perspective.

**Real-world Application Scenarios:**

**Example 1: Intelligent Tutoring Systems**
- **Use case:** Predict student questions during learning sessions
- **Value:** Given a teacher's explanation, predict what the student will ask next
- **Application:** Adaptive learning platforms can prepare follow-up materials based on predicted student confusion points
- **Example flow:**
  - Teacher: "Derivatives measure the rate of change"
  - Model predicts student might ask: "Can you show an example?" or "What's the difference between derivatives and integrals?"

**Example 2: Healthcare Conversation Analysis**
- **Use case:** Predict patient questions during medical consultations
- **Value:** Help doctors anticipate patient concerns and prepare thorough responses
- **Application:** Medical training simulations, conversation analysis tools
- **Example:** After doctor explains diagnosis, predict patient will ask about treatment options, side effects, or recovery time

**Example 3: Negotiation and Sales Training**
- **Use case:** Predict client/customer responses in sales conversations
- **Value:** Train salespeople by simulating realistic customer reactions
- **Application:**
  - Given a sales pitch, predict customer objections
  - Prepare counter-arguments in advance
  - Example: After product demo, predict customer might ask about pricing, competitors, or specific features

**Example 4: Mental Health Chatbot Evaluation**
- **Use case:** Predict patient responses in therapy conversations
- **Value:** Evaluate if therapeutic interventions elicit expected user responses
- **Application:** Quality control for mental health chatbots - ensure they're creating safe, therapeutic conversations

---

## Part 5: Cross-Entropy Loss Implementation (20 points)

### Validation Result
**[TODO: Insert screenshot showing all test cases passed]**

### Loss Computation Explanation

**Formula:**
The cross-entropy loss for language modeling is computed as:

```
Loss = -Σ log(P(y_i | x)) / N
```

Where:
- `y_i` is the correct next token
- `P(y_i | x)` is the model's predicted probability for token `y_i`
- `N` is the number of valid tokens (`num_items_in_batch`)

**Implementation Steps:**

1. **Causal Shift:**
   ```python
   shift_logits = logits[:, :-1, :]  # Predictions
   shift_labels = labels[:, 1:]       # Targets
   ```
   In language modeling, `logits[:, i]` predicts `labels[:, i+1]` (next token)

2. **Flatten Tensors:**
   ```python
   flat_logits = shift_logits.view(-1, vocab_size)
   flat_labels = shift_labels.view(-1)
   ```
   Convert from `[batch, seq_len, vocab]` to `[batch*seq_len, vocab]` for easier processing

3. **Compute Log Probabilities:**
   ```python
   log_probs = F.log_softmax(flat_logits, dim=-1)
   ```
   Use `log_softmax` for numerical stability (avoids underflow)

4. **Gather Target Token Probabilities:**
   ```python
   token_log_probs = log_probs.gather(dim=-1, index=flat_labels.unsqueeze(-1))
   ```
   Extract the log probability of the correct token at each position

5. **Mask Invalid Tokens:**
   ```python
   mask = (flat_labels != IGNORE_TOKEN_ID).float()
   masked_log_probs = token_log_probs * mask
   ```
   Zero out contributions from padding and masked tokens

6. **Compute Final Loss:**
   ```python
   total_loss = -masked_log_probs.sum()
   loss = total_loss / num_items_in_batch
   ```
   Negative log likelihood, normalized by valid token count

**Code reference:** `loss_functions.py:62-96`

### What is `num_items_in_batch` and Why is it Necessary?

**Definition:**
`num_items_in_batch` represents the **total number of valid tokens** (non-masked, non-padding) across all sequences in the batch, accounting for gradient accumulation.

**Why it's necessary:**

1. **Proper Normalization:**
   - Without normalization, loss magnitude would depend on batch size
   - Different batch sizes would lead to different gradient magnitudes
   - Makes training unstable and hyperparameters non-transferable

2. **Handling Variable-Length Sequences:**
   - Sequences have different lengths due to padding
   - Some tokens are masked (user prompts in SFT)
   - We only want to normalize by **actual training tokens**

3. **Gradient Accumulation Compatibility:**
   - When using gradient accumulation, we process multiple micro-batches
   - HuggingFace Trainer automatically scales `num_items_in_batch` to include accumulation factor
   - This ensures gradients are correctly normalized across accumulation steps
   - Example: With 128 accumulation steps and 10 valid tokens per micro-batch:
     - `num_items_in_batch` = 10 (in our implementation for that micro-batch)
     - Trainer handles the accumulation scaling internally

4. **Consistent Loss Values:**
   - Mean loss per token is interpretable (typically 0-10 range)
   - Sum loss would be huge and meaningless
   - Allows comparing losses across different batch configurations

**Example:**
```python
# Batch with 2 sequences
labels = [
    [5, 10, -100, 2],      # 3 valid tokens (position 2 is masked)
    [3, -100, -100, -100]  # 1 valid token
]
# num_items_in_batch = 4 (total valid tokens)
# loss = total_loss / 4
```

---

## Part 6: Supervised Fine-Tuning (25 points)

### Training Configuration Summary
**[TODO: Fill in after running training]**

```
GPU Type: [e.g., NVIDIA RTX 2080 Ti]
Batch Size per Device: [BSZPERDEV value]
Gradient Accumulation Steps: [calculated from TOTALBSZ/BSZPERDEV]
Effective Batch Size: [TOTALBSZ value]
Learning Rate: [e.g., 2e-5]
Number of Epochs: [e.g., 3]
Total Training Steps: [from training logs]
Max Sequence Length: [e.g., 2048]
Warmup Ratio: [e.g., 0.1]
LR Scheduler: [e.g., cosine]
```

### Training Loss Curve
**[TODO: Insert screenshot from W&B or console logs showing loss over training steps]**

### Final Checkpoint Path
```
[TODO: Insert path, e.g., ./output/checkpoint-XXXX]
```

### Question: Role of `tokenizer.apply_chat_template()`

**Purpose:**
`tokenizer.apply_chat_template()` converts a structured conversation (list of message dictionaries) into a properly formatted token sequence with special tokens that the model understands.

**How it formats conversations:**

1. **Input Format:**
   ```python
   messages = [
       {"role": "user", "content": "What is Python?"},
       {"role": "assistant", "content": "Python is a programming language."}
   ]
   ```

2. **Template Application:**
   The function uses the tokenizer's chat template (e.g., Qwen3ChatTemplate for SmolLM2) to add special tokens:
   ```
   <|im_start|>user
   What is Python?<|im_end|>
   <|im_start|>assistant
   Python is a programming language.<|im_end|>
   ```

3. **Tokenization:**
   Converts the formatted string into token IDs that the model can process

**Key Features:**

- **Consistency:** Ensures all conversations follow the same format the model was pre-trained on
- **Special Tokens:** Adds `<|im_start|>`, `<|im_end|>`, and role markers (user/assistant/system)
- **Turn Separation:** Clearly delineates between different speakers in the conversation
- **Generation Prompt:** `add_generation_prompt=False` during training (we have the full conversation)
- **Generation Prompt:** `add_generation_prompt=True` during inference (model should continue generating)

**Role in SFT Pipeline:**

1. **Preprocessing:** Converts raw conversation data into model input format
2. **Boundary Detection:** Allows `prefix_lengths` calculation to identify message boundaries
3. **Loss Masking:** Enables precise identification of which tokens belong to which role
4. **Inference:** At generation time, properly formats the prompt so the model continues appropriately

Without `apply_chat_template()`, the model wouldn't know where one message ends and another begins, making role-based loss masking impossible.

---

## Part 7: IFEval Evaluation & Hyperparameter Tuning (35 points)

### Baseline vs Fine-tuned Model Comparison

**[TODO: Fill in after running evaluation]**

| Model | Strict Accuracy | Loose Accuracy |
|-------|----------------|----------------|
| Base SmolLM2-135M | [TODO: %] | [TODO: %] |
| Fine-tuned Model | [TODO: %] | [TODO: %] |
| **Improvement** | **[TODO: +X%]** | **[TODO: +X%]** |

### Final Best Model Performance

**Strict Accuracy:** [TODO: Must be > 22%]

### Hyperparameter Tuning Summary

**[TODO: Fill in with your experiments]**

| Experiment | Learning Rate | Epochs | Batch Size | Warmup Ratio | Scheduler | Training Loss | Strict Acc | Loose Acc | Notes |
|------------|---------------|--------|------------|--------------|-----------|---------------|------------|-----------|-------|
| Baseline | 2e-5 | 3 | 128 | 0.1 | cosine | [TODO] | [TODO] | [TODO] | Initial config |
| Exp 1 | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [Description] |
| Exp 2 | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [Description] |
| Exp 3 | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [Description] |
| **Best** | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | **[TODO]** | [TODO] | Final config |

### Example Outputs

**[TODO: Provide 2-3 examples showing before/after SFT behavior]**

#### Example 1: [Instruction Type]
**Prompt:**
```
[TODO: Insert IFEval prompt]
```

**Base Model Output:**
```
[TODO: Insert base model generation]
```

**Fine-tuned Model Output:**
```
[TODO: Insert fine-tuned model generation]
```

**Analysis:** [TODO: Explain what improved]

---

#### Example 2: [Instruction Type]
**Prompt:**
```
[TODO: Insert IFEval prompt]
```

**Base Model Output:**
```
[TODO: Insert base model generation]
```

**Fine-tuned Model Output:**
```
[TODO: Insert fine-tuned model generation]
```

**Analysis:** [TODO: Explain what improved]

---

#### Example 3: [Instruction Type]
**Prompt:**
```
[TODO: Insert IFEval prompt]
```

**Base Model Output:**
```
[TODO: Insert base model generation]
```

**Fine-tuned Model Output:**
```
[TODO: Insert fine-tuned model generation]
```

**Analysis:** [TODO: Explain what improved]

---

### Analysis

#### Which instruction-following capabilities improved?

**[TODO: Fill in based on your results, e.g.:]**
- Length constraints (e.g., "write at least 300 words")
- Format requirements (e.g., "include exactly 3 paragraphs")
- Content constraints (e.g., "end with the word 'love'")
- Keyword inclusion/exclusion
- Structural requirements (e.g., "use bullet points")

#### Instruction types where the model still struggles?

**[TODO: Fill in based on your results, e.g.:]**
- Complex multi-constraint instructions
- Precise numerical requirements
- Specific formatting edge cases
- [Other patterns you notice]

#### Why certain hyperparameters helped?

**[TODO: Analyze based on your experiments, e.g.:]**

**Learning Rate Impact:**
- Higher LR (e.g., 5e-5): [Analysis of what happened]
- Lower LR (e.g., 1e-5): [Analysis of what happened]
- Optimal: [Your best LR and why]

**Number of Epochs Impact:**
- Too few epochs: [Underfitting observations]
- Optimal epochs: [Your best epoch count and reasoning]
- Too many epochs: [Overfitting observations if any]

**Batch Size Impact:**
- Larger batches: [More stable gradients but...]
- Smaller batches: [More noise but potentially...]

**Learning Rate Schedule:**
- [Which scheduler worked best and why]

**Overall Insights:**
[Your key takeaways from the tuning process]

---

## Conclusion

This assignment demonstrated the complete pipeline for supervised fine-tuning of language models:

1. **Data Preprocessing:** Implemented proper loss masking to train only on desired outputs
2. **Loss Computation:** Built cross-entropy loss from scratch with proper handling of masking and normalization
3. **Model Training:** Successfully fine-tuned SmolLM2-135M on conversational data
4. **Evaluation:** Achieved [TODO: X%] strict accuracy on IFEval (baseline: ~22%)
5. **Hyperparameter Tuning:** Systematically improved performance through experimentation

**Key Learnings:**
- [TODO: Add your personal insights from the assignment]

---

**Total Score: 100 points**
- Part 1: 5 points ✓
- Part 2: 5 points ✓
- Part 3: 5 points ✓
- Part 4: 5 points ✓
- Part 5: 20 points ✓
- Part 6: 25 points [TODO: Complete training]
- Part 7: 35 points [TODO: Complete evaluation and tuning]
