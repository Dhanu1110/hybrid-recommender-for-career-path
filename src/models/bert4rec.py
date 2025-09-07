#!/usr/bin/env python3
"""
BERT4Rec Model for Career Path Generation

This module implements a BERT-like transformer model for sequential recommendation
of career paths, adapted for the career recommendation domain.

Key components:
- Multi-head self-attention
- Position embeddings
- Masked language modeling for sequence prediction
- Configurable architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations and reshape
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output = self._attention(Q, K, V, mask)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(attention_output)
    
    def _attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute scaled dot-product attention."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)

class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward layers."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of transformer block."""
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class BERT4Rec(nn.Module):
    """
    BERT4Rec model for sequential career path recommendation.
    
    This model uses a bidirectional transformer to learn representations
    of career sequences and predict masked positions.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 256,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 d_ff: int = 1024,
                 max_seq_len: int = 50,
                 dropout: float = 0.1,
                 pad_token_id: int = 0,
                 mask_token_id: int = 1):
        """
        Initialize BERT4Rec model.
        
        Args:
            vocab_size: Size of the job vocabulary
            d_model: Model dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            pad_token_id: ID for padding token
            mask_token_id: ID for mask token
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of BERT4Rec.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Expand attention mask for multi-head attention
        # Shape: (batch_size, 1, seq_len, seq_len)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.expand(
            batch_size, 1, seq_len, seq_len
        )
        
        # Token embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Add positional encoding
        embeddings = self.position_encoding(embeddings.transpose(0, 1)).transpose(0, 1)
        
        # Pass through transformer blocks
        hidden_states = embeddings
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, extended_attention_mask)
        
        # Layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        return logits
    
    def predict_masked_tokens(self, 
                            input_ids: torch.Tensor,
                            attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict tokens at masked positions.
        
        Args:
            input_ids: Input token IDs with masked positions
            attention_mask: Attention mask
            
        Returns:
            Predicted token probabilities for masked positions
        """
        logits = self.forward(input_ids, attention_mask)
        
        # Get mask positions
        mask_positions = (input_ids == self.mask_token_id)
        
        # Extract logits for masked positions
        masked_logits = logits[mask_positions]
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(masked_logits, dim=-1)
        
        return probabilities
    
    def generate_next_jobs(self, 
                          input_ids: torch.Tensor,
                          top_k: int = 10,
                          temperature: float = 1.0) -> List[List[Tuple[int, float]]]:
        """
        Generate next job recommendations for given sequences.
        
        Args:
            input_ids: Input sequences of shape (batch_size, seq_len)
            top_k: Number of top recommendations to return
            temperature: Temperature for softmax (higher = more diverse)
            
        Returns:
            List of recommendations for each sequence
        """
        self.eval()
        with torch.no_grad():
            # Add mask token at the end of each sequence
            batch_size, seq_len = input_ids.size()
            
            # Find the last non-padding position for each sequence
            attention_mask = (input_ids != self.pad_token_id).float()
            seq_lengths = attention_mask.sum(dim=1).long()
            
            # Create new input with mask token at the end
            extended_input = torch.full((batch_size, seq_len + 1), 
                                      self.pad_token_id, 
                                      dtype=input_ids.dtype,
                                      device=input_ids.device)
            
            for i in range(batch_size):
                length = seq_lengths[i]
                extended_input[i, :length] = input_ids[i, :length]
                extended_input[i, length] = self.mask_token_id
            
            # Get predictions
            logits = self.forward(extended_input)
            
            # Extract logits for the mask positions
            recommendations = []
            for i in range(batch_size):
                length = seq_lengths[i]
                mask_logits = logits[i, length, :]
                
                # Apply temperature
                if temperature != 1.0:
                    mask_logits = mask_logits / temperature
                
                # Get top-k predictions
                probabilities = F.softmax(mask_logits, dim=-1)
                top_probs, top_indices = torch.topk(probabilities, top_k)
                
                # Convert to list of (token_id, probability) tuples
                seq_recommendations = [
                    (int(idx), float(prob)) 
                    for idx, prob in zip(top_indices, top_probs)
                ]
                recommendations.append(seq_recommendations)
            
            return recommendations
    
    def get_sequence_embeddings(self, 
                              input_ids: torch.Tensor,
                              attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get sequence-level embeddings by pooling token representations.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Sequence embeddings of shape (batch_size, d_model)
        """
        self.eval()
        with torch.no_grad():
            # Get hidden states from the last transformer layer
            embeddings = self.token_embedding(input_ids)
            embeddings = self.position_encoding(embeddings.transpose(0, 1)).transpose(0, 1)
            
            hidden_states = embeddings
            for transformer_block in self.transformer_blocks:
                hidden_states = transformer_block(hidden_states, attention_mask)
            
            hidden_states = self.layer_norm(hidden_states)
            
            # Mean pooling over non-padding tokens
            if attention_mask is None:
                attention_mask = (input_ids != self.pad_token_id).float()
            
            # Expand attention mask
            attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            
            # Sum and normalize
            sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
            sum_mask = torch.sum(attention_mask, dim=1)
            
            # Avoid division by zero
            sequence_embeddings = sum_embeddings / torch.clamp(sum_mask, min=1e-9)
            
            return sequence_embeddings

class BERT4RecConfig:
    """Configuration class for BERT4Rec model."""
    
    def __init__(self,
                 vocab_size: int = 10000,
                 d_model: int = 256,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 d_ff: int = 1024,
                 max_seq_len: int = 50,
                 dropout: float = 0.1,
                 pad_token_id: int = 0,
                 mask_token_id: int = 1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout,
            'pad_token_id': self.pad_token_id,
            'mask_token_id': self.mask_token_id
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'BERT4RecConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

def create_bert4rec_model(config: BERT4RecConfig) -> BERT4Rec:
    """
    Factory function to create BERT4Rec model from config.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized BERT4Rec model
    """
    return BERT4Rec(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        pad_token_id=config.pad_token_id,
        mask_token_id=config.mask_token_id
    )
