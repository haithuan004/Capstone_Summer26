import torch
import numpy as np
from stgcn_model import STGCN

# Danh sách các action classes đã training
ACTION_CLASSES = ("standing", "walking", "sitting", "falling")

def load_model(checkpoint_path="best_model.pth", device="cpu"):
    """
    Load mô hình ST-GCN với trọng số tốt nhất.
    """
    print(f"Loading model from: {checkpoint_path}...")
    
    # Khởi tạo kiến trúc mô hình (3 channels: x, y, confidence | 4 classes)
    model = STGCN(in_channels=3, num_classes=4, dropout=0.0)
    
    # Load trọng số model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
        
    model.to(device)
    model.eval()  # Set model về chế độ inference (tắt dropout/batchnorm training mode)
    print("Model loaded successfully!")
    return model

def predict_single_sequence(model, sequence_tensor, device="cpu"):
    """
    Hàm test dự đoán trên 1 clip hành động duy nhất.
    Input:
        sequence_tensor: Tensor shape (C=3, T<=100, V=17) hoặc (1, 3, T, 17)
                         C: [x, y, confidence score]
                         T: số frames (thường là 100)
                         V: 17 joints chuẩn COCO
    """
    # Nếu batch size=1 bị thiếu, hãy thêm vào -> shape (1, 3, T, 17)
    if sequence_tensor.dim() == 3:
        sequence_tensor = sequence_tensor.unsqueeze(0)
        
    sequence_tensor = sequence_tensor.to(device)
    
    with torch.no_grad():
        # Lấy output thô rừ mô hình
        logits = model(sequence_tensor)
        
        # Chuyển đổi thành dạng phần trăm xác suất
        probs = torch.softmax(logits, dim=1)[0]
        
    pred_class_idx = torch.argmax(probs).item()
    pred_class_name = ACTION_CLASSES[pred_class_idx]
    confidence = probs[pred_class_idx].item()
    
    return pred_class_name, confidence, probs

if __name__ == "__main__":
    # --- DEMO MAIN SCRIPT ---
    
    # 1. Xác định thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load model
    model = load_model("best_model.pth", device)
    
    # 3. Giả lập một dữ liệu trích xuất từ OpenCV/Mediapipe (để text code)
    # Tensor shape thực tế: (3 kênh, 100 frames, 17 keypoints)
    # C=0: toạ độ x, C=1: toạ độ y, C=2: điểm tự tin của joint
    dummy_input = torch.randn(1, 3, 100, 17) 
    
    # 4. Chạy hàm dự đoán
    predicted_action, conf, all_probs = predict_single_sequence(model, dummy_input, device)
    
    # 5. In kết quả cuối cùng
    print("\n" + "="*40)
    print("        INFERENCE DEMO RESULT")
    print("="*40)
    print(f"[PREDICTION] Action : {predicted_action.upper()}")
    print(f"[CONFIDENCE] Score  : {conf*100:.2f}%\n")
    
    print("Class Probabilities:")
    for i, action in enumerate(ACTION_CLASSES):
        print(f"  - {action.capitalize():10s}: {all_probs[i]*100:>6.2f}%")
    print("="*40)
