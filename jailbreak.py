import os
import json
import torch
import math
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from scipy.stats import pointbiserialr
import wandb

# Parameters
model_id = 'gpt2'
sae_model_path = 'jbloom/GPT2-Small-SAEs-Reformatted'
dataset_name = 'TrustAIRLab/in-the-wild-jailbreak-prompts'
target_layers = range(12)  # Layers 0 to 11
output_file = 'layerwise_feature_label_correlations.json'

# Initialize wandb
wandb.init(
    project="layerwise-feature-analysis",
    config={"model_id": model_id, "dataset_name": dataset_name, "layers": list(target_layers)},
)

# Load tokenizer and model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()

# Load dataset
def preprocess_dataset(dataset_name):
    regular = load_dataset(dataset_name, 'regular_2023_12_25', split='train')
    jailbreak = load_dataset(dataset_name, 'jailbreak_2023_12_25', split='train')
    combined = concatenate_datasets([regular, jailbreak])
    return combined.select_columns(['prompt', 'jailbreak'])

data = preprocess_dataset(dataset_name)

# Extract labels and texts
labels = []
texts = []
for example in tqdm(data, desc="Processing"):
    labels.append(example['jailbreak'])
    texts.append(example['prompt'])

# Binarize labels
lb = LabelBinarizer()
binary_labels = lb.fit_transform(labels).squeeze()

# Tokenize the dataset
tokens_list = []
for text in tqdm(texts, desc="Tokenizing"):
    encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=1024).to(device)
    tokens_list.append(encoding)

# Load SAE model
saes = []
for layer in tqdm(target_layers, desc="Loading SAEs"):
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        sae_model_path,
        f'blocks.{layer}.hook_resid_pre',
        device=device
    )
    saes.append(sae)

# Collect feature activations
feature_activations = []
for encoding in tqdm(tokens_list, desc=f"Collecting activations"):
    input_ids = encoding['input_ids']
    with torch.no_grad():
        outputs = model(**encoding, output_hidden_states=True)
        all_hidden_states = torch.stack(outputs.hidden_states[:len(saes)], dim=0)
        activations = torch.stack([sae.encode(hidden_state) for sae, hidden_state in zip(saes, all_hidden_states)], dim=0)  # (num_layers, seq_len, num_features)
        activations = activations.squeeze(1).cpu().numpy() # (num_layers, sequence_length, num_features)
        binary_activations = (activations > 0).astype(int)
        aggregated_activations = binary_activations.max(axis=1)  # Aggregate activations
        feature_activations.append(aggregated_activations) # (num_layers, num_features)

# Iterate through layers
all_layer_results = {}
for layer in tqdm(target_layers, desc="Processing layers"):
    print(f"Processing layer {layer}...")
    
    feature_activations = np.array(feature_activations)  # (num_samples, num_layers, num_features)
    num_features = feature_activations.shape[2]

    # Compute correlations
    correlations = []
    for feature_idx in tqdm(range(num_features), desc=f"Layer {layer} - Computing correlations"):
        feature_values = feature_activations[:, layer, feature_idx]
        corr, p_value = pointbiserialr(binary_labels, feature_values)
        correlations.append({
            'feature_index': feature_idx,
            'correlation': corr,
            'p_value': p_value
        })
    
    # Find top 5 features for the layer
    sorted_features = sorted(
        correlations, key=lambda x: 0 if math.isnan(x['correlation']) else abs(x['correlation']), reverse=True
    )
    top_features = sorted_features[:5]
    all_layer_results[f'layer_{layer}'] = top_features

    # Log top features for the layer to wandb
    wandb.log({f"layer_{layer}_top_features": top_features})

# Save results to JSON
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_layer_results, f, indent=2, ensure_ascii=False)
print(f"Correlation results saved to '{output_file}'.")

# Finish wandb run
wandb.finish()
