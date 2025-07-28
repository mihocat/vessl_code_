#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom Sentence Transformer module for jinaai/jina-embeddings-v3
This is a compatibility layer to handle the missing custom_st module
"""

from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer
import torch
from torch import nn


class JinaEmbeddingV3(nn.Module):
    """Compatibility wrapper for Jina Embeddings V3"""
    
    def __init__(self, model_name_or_path, **kwargs):
        super().__init__()
        # Use standard sentence transformer initialization
        self.auto_model = Transformer(model_name_or_path, **kwargs)
        self.pooling = Pooling(self.auto_model.get_word_embedding_dimension())
        
    def forward(self, features):
        # Standard forward pass
        trans_features = self.auto_model(features)
        pooled_features = self.pooling(trans_features)
        return pooled_features


# Make the module importable
__all__ = ['JinaEmbeddingV3']