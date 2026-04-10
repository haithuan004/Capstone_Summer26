import os
import random
import torch
from pathlib import Path
from export_cvat_zips_to_stgcn import extract_samples, ACTION_CLASSES
from stgcn_model import STGCN

def load_stgcn_model(checkpoint_path, device="cpu"):
    print(f"Loading model from: {checkpoint_path}...")
    model = STGCN(in_channels=3, num_classes=4, dropout=0.0)
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
        
    model.to(device)
    model.eval()
    return model

def predict_sequence(model, sequence_tensor, device="cpu"):
    if sequence_tensor.dim() == 3:
        sequence_tensor = sequence_tensor.unsqueeze(0)
    
    sequence_tensor = sequence_tensor.to(device)
    
    with torch.no_grad():
        logits = model(sequence_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        
    pred_class_idx = torch.argmax(probs).item()
    pred_class_name = ACTION_CLASSES[pred_class_idx]
    confidence = probs[pred_class_idx].item()
    
    return pred_class_name, confidence

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "final_result/best_model.pth"
    model = load_stgcn_model(model_path, device)
    
    data_dir = Path("interpolated_zips")
    zip_files = list(data_dir.glob("*.zip"))
    
    all_extracted = []
    # Extract all samples from zips first
    for z in zip_files:
        try:
            samples = extract_samples(z, max_t=100) # (name, arr=(3, T, 17, 1), label_idx)
            all_extracted.extend(samples)
        except Exception as e:
            print(f"Failed to extract {z}: {e}")
            
    if len(all_extracted) == 0:
        print("No valid samples found.")
        return
        
    # Pick 10 random samples
    if len(all_extracted) > 10:
        sampled_list = random.sample(all_extracted, 10)
    else:
        sampled_list = all_extracted
        
    print("\n" + "="*50)
    print("      INFERENCE RESULT (10 Random Samples)")
    print("="*50)
    
    for i, (name, arr, label_idx) in enumerate(sampled_list, 1):
        # arr shape: (3, 100, 17, 1) -> target: (1, 3, 100, 17)
        tensor_data = torch.from_numpy(arr).squeeze(-1).unsqueeze(0) # (1, 3, 100, 17)
        
        pred_class, conf = predict_sequence(model, tensor_data, device)
        gt_class = ACTION_CLASSES[label_idx]
        
        match_str = "[MATCH]" if pred_class == gt_class else "[MISMATCH]"
        
        print(f"Sample {i:2d} | Source: {name}")
        print(f"  Ground Truth : {gt_class.upper()}")
        print(f"  Prediction   : {pred_class.upper()} ({conf*100:.1f}%) {match_str}")
        print("="*50)
    
    # Save the targets to JSON so plot_inference_results.py can read them
    import json
    targets_dict = {}
    for name, _, _ in sampled_list:
        base_name = "_".join(name.split("_track")[0].split("_"))
        zip_name = f"{base_name}.zip"
        if zip_name not in targets_dict:
            targets_dict[zip_name] = []
        targets_dict[zip_name].append(name)
        
    with open("selected_targets.json", "w") as f:
        json.dump(targets_dict, f)
        
if __name__ == "__main__":
    main()
