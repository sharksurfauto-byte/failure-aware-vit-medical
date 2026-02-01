"""
Minimal Vision Transformer (ViT) Baseline for Malaria Classification

Architecture:
- Patch size: 16×16 (14×14 = 196 patches)
- Embed dim: 384
- Layers: 6
- Heads: 6
- ~5M parameters

Purpose: Enable attention-based explainability and uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """Embed image patches + add CLS token and positional embeddings"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 14*14 = 196
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images
        
        Returns:
            (B, num_patches + 1, embed_dim) patch embeddings with CLS token
        """
        B = x.shape[0]
        
        # Patch embedding: (B, 3, 224, 224) → (B, embed_dim, 14, 14) → (B, 196, embed_dim)
        x = self.proj(x)  # (B, embed_dim, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (B, 196, embed_dim)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self Attention with attention map extraction"""
    
    def __init__(self, embed_dim=384, num_heads=6, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for explainability
        self.attention_weights = None
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, N, embed_dim) input tokens
            return_attention: if True, return attention weights
        
        Returns:
            output: (B, N, embed_dim)
            attn_weights: (B, num_heads, N, N) if return_attention
        """
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Store for explainability
        self.attention_weights = attn.detach()
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        if return_attention:
            return x, attn
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block: MSA + MLP with residual connections"""
    
    def __init__(self, embed_dim=384, num_heads=6, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, N, embed_dim)
        
        Returns:
            x: (B, N, embed_dim)
        """
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBaseline(nn.Module):
    """
    Minimal Vision Transformer for binary malaria classification.
    
    Architecture:
        Patch embedding (16×16) → 6 Transformer blocks → CLS token → Classifier
    
    Features:
        - Attention map extraction for explainability
        - Dropout for MC Dropout uncertainty
        - Feature extraction before classification
    
    Total params: ~5M
    """
    
    def __init__(
        self, 
        img_size=224, 
        patch_size=16,
        in_channels=3,
        num_classes=2,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Dropout for MC Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, 3, 224, 224) input images
            return_attention: if True, return attention from all layers
        
        Returns:
            logits: (B, num_classes)
            attention_maps: list of (B, num_heads, N, N) if return_attention
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, 197, embed_dim)
        
        # Transformer blocks
        attention_maps = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attention_maps.append(attn)
            else:
                x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        # CLS token for classification
        cls_token = x[:, 0]  # (B, embed_dim)
        
        # Classify
        logits = self.head(self.dropout(cls_token))
        
        if return_attention:
            return logits, attention_maps
        return logits
    
    def get_features(self, x):
        """Extract CLS token features before classification"""
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token
    
    def get_attention_rollout(self, x):
        """
        Compute attention rollout for explainability.
        Aggregates attention across all layers.
        
        Returns:
            rollout: (B, num_patches) attention weights per patch
        """
        # Get attention from all layers
        _, attention_maps = self.forward(x, return_attention=True)
        
        # Rollout: multiply attention matrices
        # Start with identity
        B = x.shape[0]
        num_patches = self.patch_embed.num_patches + 1  # +1 for CLS
        rollout = torch.eye(num_patches, device=x.device).unsqueeze(0).expand(B, -1, -1)
        
        for attn in attention_maps:
            # Average over heads
            attn_avg = attn.mean(dim=1)  # (B, N, N)
            
            # Add identity (residual connection)
            attn_avg = 0.5 * attn_avg + 0.5 * torch.eye(num_patches, device=x.device)
            
            # Renormalize
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            
            # Multiply
            rollout = attn_avg @ rollout
        
        # Get attention to CLS token from all patches
        cls_attention = rollout[:, 0, 1:]  # (B, num_patches) excluding CLS
        
        return cls_attention


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = ViTBaseline()
    print(f"ViT Baseline")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test with attention
    logits, attn_maps = model(x, return_attention=True)
    print(f"\nAttention maps: {len(attn_maps)} layers")
    print(f"Attention shape per layer: {attn_maps[0].shape}")
    
    # Test attention rollout
    rollout = model.get_attention_rollout(x)
    print(f"\nAttention rollout shape: {rollout.shape}")
    
    # Test feature extraction
    features = model.get_features(x)
    print(f"Feature shape: {features.shape}")
