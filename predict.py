import os
import torch
import torch.nn as nn
import pandas as pd
import argparse
import glob
from typing import Dict, List, Iterable, Tuple

# -----------------------
# IMPORTS FROM YOUR REPO
# -----------------------
# We now import the Combined classes as well
from Networks import FixedMLP, FixedCombinedMLP, DynamicMLP, DynamicCombinedMLP

# ESMC Imports
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

# -----------------------
# Pooling helpers
# -----------------------
def mean_pool(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=0)

def median_pool(x: torch.Tensor) -> torch.Tensor:
    return x.median(dim=0).values

pooling_map = {
    "mean_pool": mean_pool,
    "median_pool": median_pool,
}

# -----------------------
# FASTA streaming utils
# -----------------------
def iter_fasta(filepath: str) -> Iterable[Tuple[str, str]]:
    with open(filepath, "r") as f:
        sid = None
        chunks: List[str] = []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if sid is not None:
                    yield sid, "".join(chunks)
                sid = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if sid is not None:
            yield sid, "".join(chunks)

def fasta_ids(filepath: str) -> List[str]:
    ids = []
    for sid, _ in iter_fasta(filepath):
        ids.append(sid)
    return sorted(ids)

def iter_fasta_in_chunks(filepath: str, chunk_size: int) -> Iterable[Dict[str, str]]:
    chunk: Dict[str, str] = {}
    for sid, seq in iter_fasta(filepath):
        chunk[sid] = seq
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = {}
    if chunk:
        yield chunk

# -----------------------
# ESMC encode (batched)
# -----------------------
@torch.inference_mode()
def esmc_encode_chunk(
    client: ESMC,
    seqs: Dict[str, str],
    device: torch.device,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    
    client = client.to(device)
    logits_cfg = LogitsConfig(sequence=True, return_embeddings=True)
    ids = list(seqs.keys())
    proteins = [ESMProtein(sequence=seqs[sid]) for sid in ids]
    out: Dict[str, torch.Tensor] = {}

    i = 0
    n = len(proteins)
    while i < n:
        j = min(i + batch_size, n)
        prots = proteins[i:j]
        ids_slice = ids[i:j]
        try:
            encoded = client.encode(prots)
            logits_out = client.logits(encoded, logits_cfg)
            embs = logits_out.embeddings
            if isinstance(embs, torch.Tensor):
                if embs.shape[0] != len(ids_slice): raise ValueError
            
            for k, sid in enumerate(ids_slice):
                emb = embs[k] if isinstance(embs, list) else embs[k]
                out[sid] = emb.squeeze().detach().to("cpu")
        except Exception:
            for sid, prot in zip(ids_slice, prots):
                enc = client.encode(prot)
                logits_out = client.logits(enc, logits_cfg)
                out[sid] = logits_out.embeddings.squeeze().detach().to("cpu")
        i = j
    return out

# -----------------------
# Model Loading Logic
# -----------------------
def load_models_from_dir(weights_dir: str, device: torch.device) -> List[Tuple[str, nn.Module]]:
    models = []
    # UPDATED: Look for .pth files (based on your directory listing)
    files = glob.glob(os.path.join(weights_dir, "*.pth"))
    
    if not files:
        print(f"No .pth files found in {weights_dir}")
        return []

    print(f"Found {len(files)} weight files. Analyzing...")

    for filepath in files:
        filename = os.path.basename(filepath)
        name_lower = filename.lower()

        # Skip min/max if explicitly present (unless they are part of a valid combo logic later)
        # But your requirements said only mean/median.
        if "max" in name_lower or "min" in name_lower:
            continue

        model = None
        pool_tag = ""
        
        # Determine if Dynamic or Fixed
        is_dynamic = "dynamic" in name_lower
        net_type = "Dynamic" if is_dynamic else "Fixed"

        # --- LOGIC UPDATE: Check Combined FIRST ---
        if "mean" in name_lower and "median" in name_lower:
            pool_tag = "mean+median"
            # Order matters in your CombinedMLP? 
            # Usually in your code: pooling_fns=[pool1, pool2]. 
            # Assuming standard order: mean then median.
            pooling_fns = [mean_pool, median_pool]
            
            if is_dynamic:
                model = DynamicCombinedMLP(input_dim=960, pooling_fns=pooling_fns)
            else:
                model = FixedCombinedMLP(input_dim=960, pooling_fns=pooling_fns)
                
        elif "mean" in name_lower:
            pool_tag = "mean"
            if is_dynamic:
                model = DynamicMLP(input_dim=960, pooling_fn=mean_pool)
            else:
                model = FixedMLP(input_dim=960, pooling_fn=mean_pool)
                
        elif "median" in name_lower:
            pool_tag = "median"
            if is_dynamic:
                model = DynamicMLP(input_dim=960, pooling_fn=median_pool)
            else:
                model = FixedMLP(input_dim=960, pooling_fn=median_pool)
        else:
            # File doesn't match known pooling types
            continue

        # Load Weights
        if model is not None:
            try:
                state = torch.load(filepath, map_location="cpu")
                model.load_state_dict(state)
                model.to(device).eval()
                
                # Create a clean column name
                # Removing extension for cleaner column header
                base_clean = os.path.splitext(filename)[0]
                model_id = f"{net_type}|{pool_tag}|{base_clean}"
                
                models.append((model_id, model))
                print(f"Loaded: {model_id}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

    return models

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="ESMC-300m Prediction Script")
    parser.add_argument("--fasta", type=str, required=True, help="Path to input FASTA file")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory containing .pth model weights")
    parser.add_argument("--output", type=str, required=True, help="Output CSV filename")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for ESMC embedding")
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of sequences to process in memory at once")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # 1. Load Models
    models = load_models_from_dir(args.weights_dir, device)
    if not models:
        print("No valid models loaded. Ensure files are .pth and contain 'mean'/'median'.")
        return

    # 2. Prepare Output Structure
    all_seq_ids = fasta_ids(args.fasta)
    preds_per_model: Dict[str, List[float]] = {name: [] for name, _ in models}

    # 3. Initialize ESMC
    print("Loading ESMC-300M...")
    esmc_client = ESMC.from_pretrained("esmc_300m")

    # 4. Stream and Predict
    print(f"Processing sequences from {args.fasta}...")
    for chunk in iter_fasta_in_chunks(args.fasta, args.chunk_size):
        chunk_ids = sorted(chunk.keys())
        
        embeddings = esmc_encode_chunk(esmc_client, chunk, device=device, batch_size=args.batch_size)

        with torch.inference_mode():
            for model_name, model in models:
                for sid in chunk_ids:
                    if sid not in embeddings:
                        preds_per_model[model_name].append(0.0)
                        continue
                        
                    emb = embeddings[sid].to(device=device, dtype=torch.float32)
                    out = model(emb)
                    
                    if isinstance(out, torch.Tensor):
                        logit = out.item() if out.ndim == 0 else float(out.view(-1)[0].item())
                    else:
                        logit = float(out)
                        
                    prob = float(torch.sigmoid(torch.tensor(logit)).item())
                    preds_per_model[model_name].append(prob)

        del embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 5. Save
    out_df = pd.DataFrame(
        data=[preds_per_model[name] for name, _ in models],
        index=[name for name, _ in models],
        columns=sorted(all_seq_ids),
        dtype=float
    )
    
    out_df.to_csv(args.output, index=True)
    print(f"Done. Predictions saved to {args.output}")

if __name__ == "__main__":
    main()