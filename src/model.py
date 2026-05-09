"""
GPT-2 with classification head for sentiment analysis.
Add new models here and register them in MODEL_REGISTRY.
"""

import torch
import torch.nn as nn
from transformers import GPT2Model


class GPT2Classifier(nn.Module):
    """GPT-2 with a classification head on top."""

    def __init__(self, num_classes=5, dropout=0.1, freeze=True):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        hidden_size = self.gpt2.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        if freeze:
            for param in self.gpt2.parameters():
                param.requires_grad = False
            print('  Mode: FROZEN — Only classifier head will be trained')
        else:
            print('  Mode: FINE-TUNING — All GPT-2 weights will be updated')

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        last_token_idx = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        sentence_repr = last_hidden[batch_indices, last_token_idx]
        sentence_repr = self.dropout(sentence_repr)
        return self.classifier(sentence_repr)


# ─── Model registry ─────────────────────────────────────────────────────
# Add new models here:  'name': ModelClass
MODEL_REGISTRY = {
    'baseline': GPT2Classifier,
}


def get_model(model_name, num_classes, freeze):
    """Instantiate a model by name from the registry."""
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f'Unknown model "{model_name}". Available: {available}')
    cls = MODEL_REGISTRY[model_name]
    return cls(num_classes=num_classes, freeze=freeze)
