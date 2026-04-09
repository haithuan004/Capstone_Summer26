import json
import os
import shutil
from pathlib import Path

def main():
    checkpoint_dir = Path("checkpoints")
    result_dir = Path("final_result")
    result_dir.mkdir(exist_ok=True)
    
    results_file = checkpoint_dir / "kfold_results.json"
    if not results_file.exists():
        print("Could not find kfold_results.json")
        return
        
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    folds = data.get("per_fold", [])
    if not folds:
        print("Could not find fold metrics in json")
        return
        
    # Find best fold (based on Accuracy, break ties with Macro F1)
    best_fold_idx = 0
    best_acc = 0
    best_f1 = 0
    
    for i, m in enumerate(folds):
        acc = m.get("accuracy", 0)
        f1 = m.get("macro_f1", 0)
        if acc > best_acc or (acc == best_acc and f1 > best_f1):
            best_acc = acc
            best_f1 = f1
            best_fold_idx = i
            
    best_metrics = folds[best_fold_idx]
    
    # Generate report
    report = []
    report.append("="*60)
    report.append(f"BEST METRICS FOUND IN FOLD {best_fold_idx}")
    report.append("="*60)
    report.append(f"Accuracy:  {best_metrics['accuracy']:.4f}")
    report.append(f"Macro F1:  {best_metrics['macro_f1']:.4f}")
    report.append(f"Precision: {best_metrics['macro_precision']:.4f}")
    report.append(f"Recall:    {best_metrics['macro_recall']:.4f}")
    
    report.append("\nPer-class details:")
    for cls_name, cls_m in best_metrics["per_class"].items():
        report.append(f"  - {cls_name.capitalize():10s}: Precision = {cls_m['precision']:.3f}, Recall = {cls_m['recall']:.3f}, F1 = {cls_m['f1']:.3f}")
        
    report.append("\nConfusion Matrix:")
    cm = best_metrics["confusion_matrix"]
    classes = list(best_metrics["per_class"].keys())
    
    header = "            " + "".join(f"{c:>10s}" for c in classes)
    report.append(header)
    for i, cls in enumerate(classes):
        row = f"  {cls:>10s}" + "".join(f"{cm[i][j]:>10d}" for j in range(len(classes)))
        report.append(row)
        
    report_text = "\n".join(report)
    print(report_text)
    
    # Save to file
    report_path = result_dir / "best_metrics_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    # Copy best model
    best_model_src = checkpoint_dir / f"best_model_fold{best_fold_idx}.pth"
    best_model_dst = result_dir / "best_model.pth"
    
    if best_model_src.exists():
        shutil.copy2(best_model_src, best_model_dst)
        print(f"\n[OK] Copied model '{best_model_src.name}' to '{best_model_dst}'")
    else:
        print(f"\n[Error] Could not find model: {best_model_src}")
        
    print(f"[OK] Exported all final results to: {result_dir.absolute()}")

if __name__ == "__main__":
    main()
