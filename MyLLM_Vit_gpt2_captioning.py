import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timm import create_model
from types import SimpleNamespace
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm
import gc
import sys
import os

REPO_ROOT = Path('/home/rami/OpenUn/LLMs-from-scratch')
MYLLM_DIR = REPO_ROOT / "MyLLM"
for path in (REPO_ROOT, MYLLM_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from MyLLM.ch3 import MultiHeadAttention
from MyLLM.ch4 import FeedForward, LayerNorm

HF_GPT2_REPO = os.environ.get("HF_GPT2_REPO", "openai-community/gpt2")


def _shift_labels(token_ids):
    """Shift token ids left by one to build next-token prediction labels."""
    labels = token_ids.copy()
    labels[:-1] = token_ids[1:]
    return labels


def _pad_token_batch(tokenizer, token_id_batch):
    """Pad a batch of token id lists to the longest length in the batch."""
    return tokenizer.pad(
        {"input_ids": token_id_batch},
        padding="longest",
        return_attention_mask=False,
        return_tensors="pt",
    )["input_ids"]


def _load_flickr_dataframe(data_dir, image_dir):
    """Load Flickr30k CSV, normalize columns, and map image paths."""
    captions_df = pd.read_csv(data_dir / "results.csv", sep="|", skipinitialspace=True)
    captions_df.columns = [str(c).strip() for c in captions_df.columns]
    captions_df.rename({"image_name": "image", "comment": "caption"}, inplace=True, axis=1)
    captions_df["image"] = captions_df["image"].map(lambda x: image_dir / str(x).strip())
    captions_df["caption"] = (
        captions_df["caption"].fillna("").astype(str).str.strip().str.lower()
    )
    return captions_df


class Dataset:
    """Simple image+caption dataset with transforms and GPT-2 tokenization."""

    def __init__(self, df, tfms):
        self.df = df
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        image_path = row["image"]
        caption_text = row["caption"]

        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        image_tensor = self.tfms(image=image_array)["image"]

        caption_text = f"{caption_text}<|endoftext|>"
        token_ids = tokenizer(caption_text, truncation=True)["input_ids"]
        labels = _shift_labels(token_ids)
        return image_tensor, token_ids, labels


def build_transforms(img_size=224, norm_mean=None, norm_std=None):
    """Create train/valid Albumentations pipelines."""
    if norm_mean is None:
        norm_mean = [0.5, 0.5, 0.5]
    if norm_std is None:
        norm_std = [0.5, 0.5, 0.5]

    sample_tfms = [
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(),
        A.ColorJitter(),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),
        A.HueSaturationValue(p=0.3),
    ]
    resize_and_norm = [
        A.Resize(img_size, img_size),
        A.Normalize(mean=norm_mean, std=norm_std, always_apply=True),
        ToTensorV2(),
    ]
    train_tfms = A.Compose([*sample_tfms, *resize_and_norm])
    valid_tfms = A.Compose(resize_and_norm)
    return train_tfms, valid_tfms

if __name__ == "__main__":


    # Data augmentation for training + validation images.
    train_tfms, valid_tfms = build_transforms()

    tokenizer = GPT2TokenizerFast.from_pretrained(HF_GPT2_REPO)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token


    # flickr30k
    data_dir = REPO_ROOT / "data"
    image_dir = data_dir / "flickr30k_images"
    captions_df = _load_flickr_dataframe(data_dir, image_dir)
    captions_df.head()

    # sampled_df = df.sample(n=20)
    # fig, axs = plt.subplots(10, 2, figsize=(20, 30))

    # for i, row in enumerate(sampled_df.iterrows()):
    #     ax = axs[i // 2, i % 2]
    #     image_path = row[1]['image']
    #     caption = row[1]['caption']
    #     image = Image.open(image_path)
    #     ax.imshow(image)
    #     ax.axis('off')
    #     ax.set_title(caption)

    # plt.tight_layout()
    # plt.show()

    train_df, val_df = train_test_split(captions_df, test_size=0.1)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    print(len(train_df),len(val_df))

    train_ds = Dataset(train_df, train_tfms)
    val_ds = Dataset(val_df, valid_tfms)

    def collate_fn(batch):
        """Stack images and pad token sequences for a batch."""
        image_tensors = [item[0] for item in batch]
        input_id_seqs = [item[1] for item in batch]
        label_seqs = [item[2] for item in batch]

        images = torch.stack(image_tensors, dim=0)
        input_ids = _pad_token_batch(tokenizer, input_id_seqs)
        labels = _pad_token_batch(tokenizer, label_seqs)

        pad_mask = (input_ids != tokenizer.pad_token_id).long()
        labels[pad_mask == 0] = -100
        return images, input_ids, labels

    sample_loader = torch.utils.data.DataLoader(
        train_ds, shuffle=True, batch_size=2, collate_fn=collate_fn
    )
    _, sample_input_ids, sample_labels = next(iter(sample_loader))
    print(sample_input_ids[0])
    print(sample_labels[0])

    class GPT2CrossAttention(nn.Module):
            """Cross-attention: queries from text, keys/values from image tokens."""

            def __init__(self, config):
                super().__init__()
                self.embed_dim = config.embed_dim
                self.n_heads = config.num_heads
                assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
                self.head_size = self.embed_dim // self.n_heads
                self.seq_len = config.seq_len
                
                self.q = nn.Linear(self.embed_dim,self.embed_dim)
                self.k = nn.Linear(self.embed_dim,self.embed_dim)
                self.v = nn.Linear(self.embed_dim,self.embed_dim)
                self.scale = self.head_size ** -0.5
                
                self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
                
                self.attn_dropout = nn.Dropout(config.attention_dropout)
                self.resid_dropout = nn.Dropout(config.residual_dropout)
                
                self.apply(self._init_weights)
                
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                
                
            def forward(self, q, k, v):
                query_states, key_states, value_states = q, k, v
                batch_size, target_len, embed_dim = query_states.shape
                
                query = self.q(query_states)
                key = self.k(key_states)
                value = self.v(value_states)
                
                query = query.view(batch_size,query.size(1),self.n_heads,self.head_size).permute(0,2,1,3)
                key = key.view(batch_size,key.size(1),self.n_heads,self.head_size).permute(0,2,1,3)
                value = value.view(batch_size,value.size(1),self.n_heads,self.head_size).permute(0,2,1,3)
                
                attn_scores = (query @ key.transpose(-2,-1)) * self.scale
                attn_scores = F.softmax(attn_scores,dim=-1)
                attn_weights = self.attn_dropout(attn_scores)
                
                attn_output = attn_weights @ value # batch x n_heads x t x head_size
                attn_output = attn_output.permute(0,2,1,3).contiguous().view(batch_size,target_len,embed_dim)
        
                out = self.c_proj(attn_output)
                out = self.resid_dropout(out)
                
                return out
  
    # Decoder Block (MyLLM self-attn + FFN, VisionGPT2 cross-attn)
    class GPT2Block(nn.Module):
        """Decoder block: self-attn + cross-attn + MLP."""

        def __init__(self, config):
            super().__init__()
            self.embed_dim = config.embed_dim
            self.ln_1 = LayerNorm(self.embed_dim)
            self.attn = MultiHeadAttention(
                d_in=self.embed_dim,
                d_out=self.embed_dim,
                context_length=config.seq_len,
                dropout=config.attention_dropout,
                num_heads=config.num_heads,
                qkv_bias=getattr(config, "qkv_bias", True),
            )
            self.ln_2 = LayerNorm(self.embed_dim)
            self.cross_attn = GPT2CrossAttention(config)
            self.ln_3 = LayerNorm(self.embed_dim)
            self.mlp = FeedForward({"emb_dim": self.embed_dim})
            
        def forward(self, hidden_states, encoder_states):
            hidden_states = hidden_states + self.attn(self.ln_1(hidden_states))
            hidden_states = hidden_states + self.cross_attn(
                self.ln_2(hidden_states), encoder_states, encoder_states
            )
            hidden_states = hidden_states + self.mlp(self.ln_3(hidden_states))
            return hidden_states

    class VisionGPT2Model(nn.Module):
        """ViT encoder + GPT-2 style decoder with cross-attention."""

        def __init__(self, config):
            super().__init__()
            
            self.config = config
            
            vit_model = create_model('vit_base_patch16_224',pretrained=True,num_classes=0)
            self.patch_embed = vit_model.patch_embed
            num_patches = self.patch_embed.num_patches
            
            self.cls_token = vit_model.cls_token
            embed_len = num_patches + vit_model.num_prefix_tokens
            self.pos_embed = vit_model.pos_embed
            self.pos_drop = nn.Dropout(p=0.)
            
            self.blocks = nn.ModuleList([vit_model.blocks[i] for i in range(config.depth)])
            
            self.tok_emb = nn.Embedding(config.vocab_size,config.embed_dim)
            self.pos_emb = nn.Embedding(config.seq_len,config.embed_dim)
            self.drop_emb = nn.Dropout(config.emb_dropout)
            self.trf_blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.depth)])
            self.final_norm = LayerNorm(config.embed_dim)
            self.out_head = nn.Linear(config.embed_dim,config.vocab_size,bias=False)
            self.out_head.weight = self.tok_emb.weight
            
        def _pos_embed(self, patch_tokens):
            """Add CLS token and positional embeddings to patch tokens."""
            pos_embed = self.pos_embed
            patch_tokens = torch.cat(
                (self.cls_token.expand(patch_tokens.shape[0], -1, -1), patch_tokens), dim=1
            )
            patch_tokens = patch_tokens + pos_embed
            return self.pos_drop(patch_tokens)

        def _encode_images(self, images):
            """Convert images to ViT token sequence."""
            patch_tokens = self.patch_embed(images)
            return self._pos_embed(patch_tokens)

        def _build_text_embeddings(self, input_ids):
            """Token + positional embeddings for GPT inputs."""
            token_embeddings = self.tok_emb(input_ids)
            position_ids = torch.arange(0, input_ids.size(1)).to(input_ids.device)
            position_embeddings = self.pos_emb(position_ids)
            return self.drop_emb(token_embeddings + position_embeddings)
        
        def pretrained_layers_trainable(self, trainable=False):
            """Freeze/unfreeze all pretrained layers (ViT + GPT2)."""
            layers = [
                self.cls_token, self.patch_embed, self.pos_embed, self.blocks,
                self.tok_emb, self.pos_emb, self.final_norm, self.out_head
            ]
            gpt_layers = [[
                self.trf_blocks[i].ln_1,self.trf_blocks[i].ln_2,
                self.trf_blocks[i].attn,self.trf_blocks[i].mlp
            ] for i in range(self.config.depth)]
            for l in gpt_layers:
                layers.extend(l)
            
            for layer in layers:
                if not isinstance(layer,nn.Parameter):
                    for p in layer.parameters():
                        p.requires_grad = trainable
                else:
                    layer.requires_grad = trainable
                    
            total_frozen_params = sum([p.numel() for p in self.parameters() if not p.requires_grad])
            print(f'{total_frozen_params=}')
            
        def unfreeze_gpt_layers(self,):
            """Unfreeze only GPT layers (keep ViT frozen)."""
            gpt_layers = [[
                self.trf_blocks[i].ln_1,self.trf_blocks[i].ln_2,
                self.trf_blocks[i].attn,self.trf_blocks[i].mlp
            ] for i in range(self.config.depth)]
            flatten = []
            for l in gpt_layers:
                flatten.extend(l)
                
            for layer in flatten:
                if not isinstance(layer,nn.Parameter):
                    for p in layer.parameters():
                        p.requires_grad = True
                else:
                    layer.requires_grad = True
            
        def _load_hf_gpt2_weights(self, sd_hf):
            """Load HF GPT-2 weights into MyLLM-compatible modules."""
            with torch.no_grad():
                self.tok_emb.weight.copy_(sd_hf['transformer.wte.weight'])
                self.pos_emb.weight.copy_(sd_hf['transformer.wpe.weight'])
                self.final_norm.scale.copy_(sd_hf['transformer.ln_f.weight'])
                self.final_norm.shift.copy_(sd_hf['transformer.ln_f.bias'])
                self.out_head.weight = self.tok_emb.weight
                
                for i, block in enumerate(self.trf_blocks):
                    block.ln_1.scale.copy_(sd_hf[f'transformer.h.{i}.ln_1.weight'])
                    block.ln_1.shift.copy_(sd_hf[f'transformer.h.{i}.ln_1.bias'])
                    block.ln_2.scale.copy_(sd_hf[f'transformer.h.{i}.ln_2.weight'])
                    block.ln_2.shift.copy_(sd_hf[f'transformer.h.{i}.ln_2.bias'])
                    
                    c_attn_w = sd_hf[f'transformer.h.{i}.attn.c_attn.weight'].t()
                    c_attn_b = sd_hf[f'transformer.h.{i}.attn.c_attn.bias']
                    q_w, k_w, v_w = c_attn_w.split(self.config.embed_dim, dim=0)
                    q_b, k_b, v_b = c_attn_b.split(self.config.embed_dim, dim=0)
                    
                    block.attn.W_query.weight.copy_(q_w)
                    block.attn.W_query.bias.copy_(q_b)
                    block.attn.W_key.weight.copy_(k_w)
                    block.attn.W_key.bias.copy_(k_b)
                    block.attn.W_value.weight.copy_(v_w)
                    block.attn.W_value.bias.copy_(v_b)
                    
                    c_proj_w = sd_hf[f'transformer.h.{i}.attn.c_proj.weight'].t()
                    c_proj_b = sd_hf[f'transformer.h.{i}.attn.c_proj.bias']
                    block.attn.out_proj.weight.copy_(c_proj_w)
                    block.attn.out_proj.bias.copy_(c_proj_b)
                    
                    fc_w = sd_hf[f'transformer.h.{i}.mlp.c_fc.weight'].t()
                    fc_b = sd_hf[f'transformer.h.{i}.mlp.c_fc.bias']
                    proj_w = sd_hf[f'transformer.h.{i}.mlp.c_proj.weight'].t()
                    proj_b = sd_hf[f'transformer.h.{i}.mlp.c_proj.bias']
                    block.mlp.layers[0].weight.copy_(fc_w)
                    block.mlp.layers[0].bias.copy_(fc_b)
                    block.mlp.layers[2].weight.copy_(proj_w)
                    block.mlp.layers[2].bias.copy_(proj_b)
        
        @classmethod    
        def from_pretrained(self, config):
            """Build model and load GPT-2 decoder weights from HF."""
            model = VisionGPT2Model(config)
            gpt2_small = GPT2LMHeadModel.from_pretrained(HF_GPT2_REPO)
            model._load_hf_gpt2_weights(gpt2_small.state_dict())
            return model
        
        def forward(self, image, input_ids, labels=None):
            """Forward pass; returns loss if labels provided, else logits."""
            image_tokens = self._encode_images(image)
            text_embeddings = self._build_text_embeddings(input_ids)
            
            for i in range(self.config.depth):
                image_tokens = self.blocks[i](image_tokens)
                text_embeddings = self.trf_blocks[i](text_embeddings, image_tokens)
            
            text_embeddings = self.final_norm(text_embeddings)
            
            if labels is not None:
                lm_logits = self.out_head(text_embeddings)
                loss = F.cross_entropy(
                    lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1)
                )
                return loss
            
            lm_logits = self.out_head(text_embeddings[:,[-1],:])
            return lm_logits
        
        def generate(self, image, sequence, max_tokens=50, temperature=1.0, deterministic=False):
            """Autoregressive generation loop."""
            for _ in range(max_tokens):
                logits = self(image, sequence)
                logits = logits[:,-1,:] / temperature
                probs = F.softmax(logits,dim=-1)
                if deterministic:
                    next_token = torch.argmax(probs,dim=-1,keepdim=True)
                else:
                    next_token = torch.multinomial(probs,num_samples=1)
                sequence = torch.cat([sequence,next_token],dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
            return sequence.cpu().flatten()
    
    class Trainer:
        """Training loop, validation, and caption generation helpers."""

        def __init__(self,model_config,train_config, dls):
            
            self.train_config = train_config
            self.model_config = model_config
            self.device = self.train_config.device
            self.amp_device = "cuda" if "cuda" in str(self.device) else "cpu"
            
            self.model = VisionGPT2Model.from_pretrained(model_config).to(self.device)
            self.model.pretrained_layers_trainable(trainable=False)
            
            print(f'trainable parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}')
            
            self.tokenizer = GPT2TokenizerFast.from_pretrained(HF_GPT2_REPO)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.scaler = GradScaler()
            
            self.train_loader, self.val_loader = dls
            
            steps_per_epoch = len(self.train_loader)
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_config.lr / 25.)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.train_config.lr,
                epochs=self.train_config.epochs,
                steps_per_epoch=steps_per_epoch
            )
            
            
            self.metrics = pd.DataFrame()
            self.metrics[['train_loss','train_perplexity','val_loss','val_perplexity']] = None
            
            self.gen_tfms = A.Compose([
                A.Resize(224,224),
                A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],always_apply=True),
                ToTensorV2()
            ])
                
            
        def save_model(self,):
            """Save best model checkpoint to disk."""
            self.train_config.model_path.mkdir(exist_ok=True)
            state_dict = self.model.state_dict()
            torch.save(state_dict,self.train_config.model_path/'captioner.pt')
            
            
        def load_best_model(self, path=None):
            """Load the best saved checkpoint (optionally from a custom path)."""
            if path is None:
                checkpoint_path = self.train_config.model_path / 'captioner.pt'
            else:
                checkpoint_path = Path(path)
            state_dict = torch.load(checkpoint_path)
            self.model.load_state_dict(state_dict)
        
        
        def train_one_epoch(self,epoch):
            """Run one epoch of training and log loss/perplexity."""
            
            progress_bar = tqdm(self.train_loader,total=len(self.train_loader))
            
            running_loss = 0.
            
            for image, input_ids, labels in progress_bar:
                
                with torch.amp.autocast(self.amp_device):
                    image = image.to(self.device)
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)
                    
                    loss = self.model(image,input_ids,labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    running_loss += loss.item()
                    
                    progress_bar.set_description(f'train loss: {loss.item():.3f}')
                    
                del image, input_ids, labels, loss
                
            train_loss = running_loss / len(self.train_loader)
            train_perplexity = np.exp(train_loss)
            
            self.metrics.loc[epoch,['train_loss','train_perplexity']] = (train_loss,train_perplexity)
            
            
        @torch.no_grad()
        def valid_one_epoch(self,epoch):
            """Run one validation epoch and return perplexity."""
            
            progress_bar = tqdm(self.val_loader,total=len(self.val_loader))
            
            running_loss = 0.
            
            for image, input_ids, labels in progress_bar:
                
                with torch.amp.autocast(self.amp_device):
                    image = image.to(self.device)
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)
                    
                    loss = self.model(image,input_ids,labels)
                    running_loss += loss.item()
                    
                    progress_bar.set_description(f'valid loss: {loss.item():.3f}')
                    
                del image, input_ids, labels, loss
                
            val_loss = running_loss / len(self.val_loader)
            val_perplexity = np.exp(val_loss)
            
            self.metrics.loc[epoch,['val_loss','val_perplexity']] = (val_loss,val_perplexity)
            
            return val_perplexity
            
            
        def clean(self):
            """Force garbage collection and clear CUDA cache."""
            gc.collect()
            torch.cuda.empty_cache()
        
        
        def fit(self,):
            """Main training loop with checkpointing on best perplexity."""
            
            best_perplexity = 1e9
            best_epoch = -1
            epoch_bar = tqdm(range(self.train_config.epochs))
            
            for epoch in epoch_bar:
                
                if epoch == self.train_config.freeze_epochs_gpt:
                    self.model.unfreeze_gpt_layers()
                    print('unfreezing GPT2 entirely...')
                    
                if epoch == self.train_config.freeze_epochs_all:
                    self.model.pretrained_layers_trainable(trainable=True)
                
                self.model.train()
                epoch_bar.set_description('training')
                self.train_one_epoch(epoch)
                self.clean()
                
                self.model.eval()
                epoch_bar.set_description('validating')
                val_perplexity = self.valid_one_epoch(epoch)
                self.clean()
                
                print(self.metrics.tail(1))
                
                if val_perplexity < best_perplexity:
                    best_perplexity = val_perplexity
                    best_epoch = epoch
                    print('saving best model...')
                    self.save_model()
                    
            return {
                'best_perplexity': best_perplexity,
                'best_epoch': best_epoch
            }
            
            
        @torch.no_grad()
        def generate_caption(self,image,max_tokens=50,temperature=1.0,deterministic=False):
            """Generate a caption for a single image file path."""

            
            self.model.eval()
            
            image_path = image
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            image_tensor = self.gen_tfms(image=image_array)['image']
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            sequence = torch.ones(1,1).to(device=self.device).long() * self.tokenizer.bos_token_id
            
            caption = self.model.generate(
                image_tensor,
                sequence,
                max_tokens=max_tokens,
                temperature=temperature,
                deterministic=deterministic
            )
            caption = self.tokenizer.decode(caption.numpy(),skip_special_tokens=True)
            
            return caption

    model_config = SimpleNamespace(
        vocab_size = 50_257,
        embed_dim = 768,
        num_heads = 12,
        seq_len = 1024,
        depth = 12,
        attention_dropout = 0.1,
        residual_dropout = 0.1,
        mlp_ratio = 4,
        mlp_dropout = 0.1,
        emb_dropout = 0.1)

    train_config = SimpleNamespace(
        epochs = 10,
        freeze_epochs_gpt = 1,
        freeze_epochs_all = 2,
        lr = 1e-4,
        device = 'cuda',
        model_path = Path('captioner'),
        batch_size = 32)

train_dl = torch.utils.data.DataLoader(train_ds,batch_size=train_config.batch_size,shuffle=True,pin_memory=True,num_workers=2,persistent_workers=True,collate_fn=collate_fn)
val_dl = torch.utils.data.DataLoader(val_ds,batch_size=train_config.batch_size,shuffle=False,pin_memory=True,num_workers=2,persistent_workers=True,collate_fn=collate_fn)


trainer = Trainer(model_config,train_config,(train_dl,val_dl))
trainer.fit()

trainer.metrics
plt.plot(trainer.metrics['train_loss'],color='red',label='train loss')
plt.plot(trainer.metrics['val_loss'],color='orange',label='valid loss')
plt.title('loss, lower=better')
plt.legend()

plt.show()
plt.plot(trainer.metrics['train_perplexity'],color='blue',label='train perplexity')
plt.plot(trainer.metrics['val_perplexity'],color='lightblue',label='valid perplexity')
plt.title('perplexity, lower=better')
plt.legend()
plt.show()

# trainer.load_best_model('/home/rami/OpenUn/LLMs-from-scratch/MyLLM_Vit_gpt2_captioning/captioner.pt')
trainer.load_best_model('/home/rami/OpenUn/LLMs-from-scratch/captioner/captioner.pt')


for i in range(20):
    det = True
    test = val_df.sample(n=1).iloc[0]
    test_img = test["image"]
    test_caption = test["caption"]
    plt.figure()
    plt.imshow(Image.open(test_img).convert('RGB'))
    t = np.random.uniform(0.5,1.5)
    if i > 40:
        det = True
    gen_caption = trainer.generate_caption(test_img,temperature=t,deterministic=det)
    print(f"actual: {test_caption}\nmodel: {gen_caption}\ntemp: {t} deterministic generation: {det}")
    plt.title(f"actual: {test_caption}\nmodel: {gen_caption}\ntemp: {t} deterministic generation: {det}")
    plt.axis('off')
    plt.show()