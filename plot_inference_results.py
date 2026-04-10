import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from export_cvat_zips_to_stgcn import extract_samples, ACTION_CLASSES
from predict_10_origin import load_stgcn_model, predict_sequence

# COCO-style skeleton edges
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # arms
    (5, 11), (6, 12), (11, 12), # torso
    (11, 13), (13, 15), (12, 14), (14, 16) # legs
]

def draw_normalized_sequence(seq_tensor, title, output_path):
    # seq_tensor: (3, T, 17) -> c, t, v
    c, t, v = seq_tensor.shape
    
    # find valid frames (where confidence > 0)
    conf = seq_tensor[2]
    valid_frames = np.where((conf > 0).any(axis=1))[0]
    if len(valid_frames) == 0:
        print(f"Skipping {title}, no valid frames")
        return
        
    # pick 10 evenly spaced frames from the valid range
    first, last = valid_frames[0], valid_frames[-1]
    actual_len = last - first + 1
    
    if actual_len < 10:
        indices = valid_frames[:10]
    else:
        indices = np.linspace(first, last, 10, dtype=int)
        
    num_frames = len(indices)
    cols = 5
    rows = (num_frames + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2), constrained_layout=True)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()  # y goes down
        ax.set_xticks([])
        ax.set_yticks([])
        
    for i, f_idx in enumerate(indices):
        ax = axes[i]
        x_pts = seq_tensor[0, f_idx]
        y_pts = seq_tensor[1, f_idx]
        c_pts = seq_tensor[2, f_idx]
        
        # draw bones
        for p1, p2 in SKELETON_EDGES:
            if c_pts[p1] > 0 and c_pts[p2] > 0:
                ax.plot([x_pts[p1], x_pts[p2]], [y_pts[p1], y_pts[p2]], color="#1f77b4", linewidth=2.0)
                
        # draw joints
        vis = c_pts > 0
        if np.any(vis):
            ax.scatter(x_pts[vis], y_pts[vis], c="#d62728", s=25, zorder=5)
            
        ax.set_title(f"Frame {f_idx}", fontsize=9)
        
        # limits
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(2.0, -2.0)
        
    for idx in range(num_frames, len(axes)):
        axes[idx].axis("off")
        
    fig.suptitle(title, fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_stgcn_model("final_result/best_model.pth", device)
    
    import json
    try:
        with open("selected_targets.json", "r") as f:
            targets = json.load(f)
    except FileNotFoundError:
        print("Run predict_10_origin.py first!")
        return
    
    data_dir = Path("interpolated_zips")
    
    for zip_name, track_names in targets.items():
        z_path = data_dir / zip_name
        if not z_path.exists():
            continue
            
        try:
            samples = extract_samples(z_path, max_t=100)
            for name, arr, label_idx in samples:
                if name in track_names:
                    tensor_data = torch.from_numpy(arr).squeeze(-1).unsqueeze(0) # (1, 3, 100, 17)
                    pred_class, conf = predict_sequence(model, tensor_data, device)
                    gt_class = ACTION_CLASSES[label_idx]
                    
                    status = "MATCH" if pred_class == gt_class else "MISMATCH"
                    
                    title = f"Source: {name}\nGT: {gt_class.upper()} | Pred: {pred_class.upper()} ({conf*100:.1f}%) [{status}]"
                    out_path = Path("final_result") / f"inference_vis_{name}.png"
                    
                    # shape is (1, 3, 100, 17), pass (3, 100, 17)
                    draw_normalized_sequence(arr.squeeze(-1), title, out_path)
                    print(f"Generated: {out_path}")
        except Exception as e:
            print(f"Failed {zip_name}: {e}")

if __name__ == "__main__":
    main()
