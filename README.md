## MyLLM ViT → GPT2 Captioning (`MyLLM_Vit_gpt2_captioning.py`)

This project implements **image captioning** by combining:

- **ViT encoder** (pretrained, from `timm`): converts an image into a sequence of visual tokens (patch tokens + CLS token).
- **GPT-style decoder** (MyLLM building blocks): generates a caption autoregressively (next-token prediction).
- **Cross-attention**: injects image information into the text stream (**Q=text**, **K/V=image tokens**).

It is essentially **GPT-2 decoder fine-tuning conditioned on images**, using a pretrained ViT as the visual backbone.

### What’s in this folder

- `MyLLM_Vit_gpt2_captioning.py`: end-to-end training + validation + demo (caption generation).
- `diagram_architecture.png` / `.dot`: architecture and flow diagrams.

### How it extends `MyLLM`

The script reuses “from scratch” Transformer components from the earlier project:

- `MyLLM/ch3.py`: `MultiHeadAttention` (used for **causal self-attention** over caption tokens)
- `MyLLM/ch4.py`: `FeedForward`, `LayerNorm` (used inside the decoder blocks)

Compared to `MyLLM/MyLLM_nominal.py` (text-only), this project adds:

- **a ViT image encoder**
- **cross-attention** so the text stream can attend to the image token sequence

### Architecture diagram

![VisionGPT2 architecture](diagram_architecture.png)

### Data assumptions (Flickr30k-style)

The code expects this layout under the repo `data/` folder:

- `data/results.csv`
- `data/flickr30k_images/` (image files referenced by `results.csv`)

Internally it loads via:

- `data_dir = REPO_ROOT / "data"`
- `image_dir = data_dir / "flickr30k_images"`

### How to run (minimal)

From repository root:

```bash
source venv/bin/activate
python MyLLM_Vit_gpt2_captioning/MyLLM_Vit_gpt2_captioning.py
```

Optional: choose a different HuggingFace GPT-2 checkpoint repo:

```bash
export HF_GPT2_REPO=openai-community/gpt2
```

### Outputs

- **Checkpoint**: best model saved to `captioner/captioner.pt`
- **Plots**:
  - training/validation loss
  - training/validation perplexity (\(\exp(\text{loss})\))
- **Demo loop**: samples images and prints/plots generated captions

---

## Code-level documentation (functions, classes, and flow)

Everything below describes what exists in `MyLLM_Vit_gpt2_captioning.py` in detail.

### Globals and key conventions

- **`REPO_ROOT`**: absolute path to the repository. Used to locate `data/` and to add modules to `sys.path`.
  - If you move the repo, update this constant near the top of the script.
- **`HF_GPT2_REPO`**: HuggingFace model id (env var override) used to load GPT-2 weights/tokenizer.
- **`tokenizer` (global)**:
  - Created in `__main__` via `GPT2TokenizerFast.from_pretrained(HF_GPT2_REPO)`.
  - Used implicitly by `Dataset.__getitem__` and `collate_fn`.
  - `pad_token` is set to the same id as `eos_token` so padding works with GPT-2 vocab.

### Helper functions

#### `_shift_labels(token_ids)`

- **Purpose**: create next-token prediction labels from a single token id sequence.
- **Input**: `token_ids` (Python list of ints).
- **Output**: `labels` list of ints with a left shift:
  - `labels[t] = token_ids[t+1]` for all but the last position
  - last label remains a copy of the last token id (it will typically be masked later if it’s padding)
- **Used by**: `Dataset.__getitem__` to build teacher-forcing labels.

#### `_pad_token_batch(tokenizer, token_id_batch)`

- **Purpose**: pad a list of variable-length token id sequences to a rectangular tensor.
- **Input**:
  - `tokenizer`: HuggingFace tokenizer configured with `pad_token`
  - `token_id_batch`: list of lists of ints (`[[...], [...], ...]`)
- **Output**: `torch.LongTensor` shaped `(B, T_max)` containing padded `input_ids`.
- **Used by**: `collate_fn` for both `input_ids` and `labels`.

#### `_load_flickr_dataframe(data_dir, image_dir)`

- **Purpose**: load the Flickr30k-style CSV and normalize column names/content.
- **Input**:
  - `data_dir`: directory containing `results.csv`
  - `image_dir`: directory containing images
- **Output**: `pd.DataFrame` with at least:
  - `image`: `Path` to the image file
  - `caption`: cleaned caption string (lowercased, stripped, NaNs replaced with `""`)
- **Notes**:
  - The CSV is read with `sep="|"` and `skipinitialspace=True`
  - Columns are stripped and `image_name → image`, `comment → caption`

#### `build_transforms(img_size=224, norm_mean=None, norm_std=None)`

- **Purpose**: build Albumentations augmentation pipelines for train/validation images.
- **Input**:
  - `img_size`: final spatial size (default `224×224` for ViT base patch16 224)
  - `norm_mean`, `norm_std`: normalization parameters (default `[0.5,0.5,0.5]`)
- **Output**: `(train_tfms, valid_tfms)` as `albumentations.Compose`.
- **Train pipeline** includes random augmentations (flip, color jitter, etc.) + resize + normalize + `ToTensorV2`.
- **Valid pipeline** includes only resize + normalize + `ToTensorV2`.

#### `collate_fn(batch)`

- **Purpose**: convert a list of dataset samples into a single batch.
- **Input**: `batch` is a list of tuples:
  - `(image_tensor, token_ids, labels)` from `Dataset.__getitem__`
- **Output**: `(images, input_ids, labels)`:
  - `images`: `torch.FloatTensor` `(B, 3, H, W)` stacked
  - `input_ids`: `torch.LongTensor` `(B, T_max)` padded
  - `labels`: `torch.LongTensor` `(B, T_max)` padded, with padding masked as `-100`
- **Important detail**:
  - Padding is masked using:
    - `pad_mask = (input_ids != tokenizer.pad_token_id)`
    - `labels[pad_mask == 0] = -100`
  - PyTorch `F.cross_entropy` ignores `-100` by convention.
- **Dependency**: uses the **global** `tokenizer`.

### Dataset

#### `class Dataset`

Simple dataset returning transformed image tensors and GPT-2 token sequences.

##### `Dataset.__init__(self, df, tfms)`

- **Stores**:
  - `df`: dataframe with `image` path and `caption` text
  - `tfms`: Albumentations transform pipeline

##### `Dataset.__len__(self)`

- Returns `len(self.df)`

##### `Dataset.__getitem__(self, idx)`

- **Loads**:
  - Image at `df.iloc[idx]["image"]` using PIL, converts to RGB, converts to numpy.
  - Applies Albumentations transforms → returns `image_tensor`.
- **Tokenizes caption**:
  - Appends `<|endoftext|>` to teach a stopping condition.
  - Uses the **global** `tokenizer(..., truncation=True)["input_ids"]`.
- **Builds labels**:
  - Uses `_shift_labels(token_ids)` for next-token prediction.
- **Returns**: `(image_tensor, token_ids, labels)` (token sequences are still Python lists here; padding happens in `collate_fn`).

### Model components

#### `class GPT2CrossAttention(nn.Module)`

Implements **cross-attention** where:

- **Queries** come from the text stream
- **Keys/Values** come from the image token sequence

This is the mechanism that “injects vision into language”.

##### `__init__(self, config)`

- Creates linear projections `q`, `k`, `v` and output projection `c_proj`.
- Uses `config.embed_dim`, `config.num_heads`, `config.attention_dropout`, `config.residual_dropout`.
- Initializes linear layers with a GPT-2-like normal init (`std=0.02`).

##### `forward(self, q, k, v)`

- **Inputs**:
  - `q`: text hidden states `(B, T_text, D)`
  - `k`, `v`: image token states `(B, T_img, D)`
- **Computation**:
  - Projects to Q/K/V, reshapes into heads `(B, H, T, Dh)`.
  - Computes attention weights `softmax(QK^T / sqrt(Dh))`.
  - Applies dropout to attention weights and output projection.
- **Output**:
  - Cross-attended text states `(B, T_text, D)` to be added via a residual connection.

#### `class GPT2Block(nn.Module)`

A single decoder block with three sublayers (each wrapped with LayerNorm + residual):

- **Causal self-attention** over text (`MultiHeadAttention` from `MyLLM`)
- **Cross-attention** over image tokens (`GPT2CrossAttention`)
- **MLP / feed-forward** (`FeedForward` from `MyLLM`)

##### `forward(self, hidden_states, encoder_states)`

- `hidden_states`: text stream `(B, T_text, D)`
- `encoder_states`: image token stream `(B, T_img, D)`
- Applies:
  - `hidden = hidden + self_attn(LN(hidden))`
  - `hidden = hidden + cross_attn(LN(hidden), encoder, encoder)`
  - `hidden = hidden + mlp(LN(hidden))`
- Returns updated `hidden_states`.

#### `class VisionGPT2Model(nn.Module)`

Full captioning model: **ViT encoder** + **GPT-style decoder** (with cross-attention).

##### `__init__(self, config)`

**ViT encoder path**

- Builds pretrained ViT via `timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)`.
- Reuses:
  - patch embedding (`patch_embed`)
  - CLS token (`cls_token`)
  - positional embeddings (`pos_embed`)
  - transformer blocks (`vit_model.blocks[:config.depth]`)

**GPT-style decoder path**

- Token embedding: `tok_emb` `(vocab_size, embed_dim)`
- Pos embedding: `pos_emb` `(seq_len, embed_dim)`
- Decoder stack: `trf_blocks` = `config.depth` instances of `GPT2Block`
- Output:
  - `final_norm` then `out_head` projecting to vocab
  - Weight tying: `out_head.weight` shares `tok_emb.weight`

##### `_pos_embed(self, patch_tokens)`

- Adds CLS token to patch tokens and adds ViT positional embeddings.
- **Input**: patch tokens `(B, N_patches, D)`
- **Output**: tokens with CLS + pos embed `(B, N_patches+1, D)`

##### `_encode_images(self, images)`

- Converts images `(B, 3, 224, 224)` → ViT token sequence `(B, T_img, D)`.
- Internally: patch embedding → `_pos_embed`.

##### `_build_text_embeddings(self, input_ids)`

- Builds decoder input embeddings:
  - `tok_emb(input_ids)` + `pos_emb(position_ids)`
- **Input**: `input_ids` `(B, T_text)`
- **Output**: embeddings `(B, T_text, D)`

##### `pretrained_layers_trainable(self, trainable=False)`

- Convenience method to freeze/unfreeze large parts of the model.
- Sets `requires_grad` on:
  - ViT token/patch/pos blocks
  - GPT token/pos embeddings, norms, head
  - GPT self-attn + MLP within each decoder block
- Prints total frozen parameter count.

##### `unfreeze_gpt_layers(self)`

- Unfreezes **only** GPT layers (decoder blocks’ norms/self-attn/MLP), leaving ViT frozen.
- Used for staged fine-tuning.

##### `_load_hf_gpt2_weights(self, sd_hf)`

- Loads HuggingFace GPT-2 weights into MyLLM-compatible layers:
  - Embeddings, final LayerNorm, attention Q/K/V projections and output projection, MLP weights.
- This is the “bridge” that makes the decoder start from GPT-2 pretrained weights.

##### `from_pretrained(cls, config)` (classmethod)

- Constructs the model and loads HF GPT-2 weights:
  - `GPT2LMHeadModel.from_pretrained(HF_GPT2_REPO)`
  - then `_load_hf_gpt2_weights(...)`
- Returns a ready-to-train `VisionGPT2Model`.

##### `forward(self, image, input_ids, labels=None)`

- **Encodes image** into `image_tokens`.
- **Embeds text** into `text_embeddings`.
- For `i in range(depth)`:
  - runs one ViT block on `image_tokens`
  - runs one decoder block on `text_embeddings` attending to `image_tokens`
- Applies `final_norm`.
- **If `labels` is provided**:
  - Computes full-token logits `(B, T_text, vocab)` and returns scalar `loss`.
- **If `labels` is not provided**:
  - Computes logits only for the last time-step `(B, 1, vocab)` for generation.

##### `generate(self, image, sequence, max_tokens=50, temperature=1.0, deterministic=False)`

Autoregressive decoding loop.

- **Inputs**:
  - `image`: image tensor batch `(1, 3, 224, 224)` in the demo helper
  - `sequence`: current token ids `(1, T)` (usually starting from BOS)
  - `temperature`: divides logits before softmax (lower → more deterministic)
  - `deterministic=True`: greedy argmax; otherwise multinomial sampling
- **Stops** early if EOS is produced (`tokenizer.eos_token_id`).
- **Returns**: generated token ids as a CPU 1D tensor.

### Training + evaluation

#### `class Trainer`

Orchestrates training/validation, checkpointing, and a simple caption-generation demo API.

##### `__init__(self, model_config, train_config, dls)`

- Builds model via `VisionGPT2Model.from_pretrained(...)` and moves to `train_config.device`.
- Freezes pretrained layers initially: `model.pretrained_layers_trainable(trainable=False)`.
- Creates tokenizer (again) for decoding during generation:
  - `self.tokenizer = GPT2TokenizerFast.from_pretrained(HF_GPT2_REPO)`
  - `pad_token = eos_token`
- Sets up:
  - AMP via `GradScaler()`
  - Adam optimizer (`lr / 25.` warm start)
  - `OneCycleLR` scheduler
  - metrics dataframe (loss + perplexity)
  - `gen_tfms` (simple resize+normalize) for inference-time preprocessing

##### `save_model(self)`

- Saves `self.model.state_dict()` to `train_config.model_path / 'captioner.pt'`.

##### `load_best_model(self, path=None)`

- Loads the saved checkpoint into `self.model`.
- If `path` is omitted, defaults to `train_config.model_path / 'captioner.pt'`.

##### `train_one_epoch(self, epoch)`

- Iterates over `train_loader` with tqdm.
- Runs forward under `torch.amp.autocast(...)`.
- Uses AMP scaling:
  - `scaler.scale(loss).backward()`
  - `scaler.step(optimizer)`
  - `scaler.update()`
- Steps scheduler each batch and zeroes grads.
- Logs epoch metrics:
  - `train_loss = mean(loss)`
  - `train_perplexity = exp(train_loss)`

##### `valid_one_epoch(self, epoch)`

- Same structure as training but under `@torch.no_grad()` and without backprop.
- Logs:
  - `val_loss`
  - `val_perplexity = exp(val_loss)`
- Returns `val_perplexity` (used for “best checkpoint” selection).

##### `clean(self)`

- Calls `gc.collect()` and `torch.cuda.empty_cache()` to reduce fragmentation between phases.

##### `run_training(self)`

Main loop (epochs):

- **Staged unfreezing schedule**:
  - when `epoch == freeze_epochs_gpt`: calls `unfreeze_gpt_layers()`
  - when `epoch == freeze_epochs_all`: calls `pretrained_layers_trainable(trainable=True)`
- For each epoch:
  - train → validate → track best perplexity → save checkpoint if improved
- Returns a dict with `best_perplexity` and `best_epoch`.

##### `generate_caption(self, image, max_tokens=50, temperature=1.0, deterministic=False)`

- Convenience wrapper for inference on a single image path.
- Loads and preprocesses the image using `self.gen_tfms`.
- Creates a starting sequence containing `bos_token_id`.
- Calls `self.model.generate(...)` and decodes tokens with `self.tokenizer.decode(..., skip_special_tokens=True)`.
- Returns caption string.

### `__main__` (how the script wires everything together)

When you run the file directly, it:

- Builds transforms (`build_transforms`)
- Creates the **global** tokenizer used by `Dataset` + `collate_fn`
- Loads and splits dataset dataframe (`_load_flickr_dataframe` + `train_test_split`)
- Builds `Dataset` + `DataLoader` (with `collate_fn`)
- Defines:
  - `model_config` (embed dim, heads, seq_len=1024, depth=12, dropout values, vocab size)
  - `train_config` (epochs, freeze schedule, lr, device, batch size, checkpoint dir)
- Trains using `Trainer.run_training()`
- Plots loss/perplexity curves
- Loads best checkpoint and runs a small qualitative demo loop

---

## Notes / limitations / gotchas

- **GPU strongly recommended**: ViT+GPT2 is compute-heavy.
- **Global tokenizer dependency**: `Dataset` and `collate_fn` use the module-level `tokenizer`.
  - If you import these in another script, make sure `tokenizer` is initialized before building DataLoaders.
- **Caption length**: `seq_len=1024` matches GPT-2 max context; real captions are much shorter, but padding/truncation behavior still depends on this config.
- **ViT depth usage**: `config.depth` controls both how many ViT blocks and how many decoder blocks are used (they’re stepped together in the forward loop).
- **Moving the repo**: update `REPO_ROOT` at the top of the script.


