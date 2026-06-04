import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

def run_diagnostics(model, test_loader, seq_test, headers_test, shape_test, class_names, device="cuda"):
    """
    Chạy chẩn đoán chi tiết mô hình đã huấn luyện:
    1. In báo cáo phân loại chi tiết (Classification Report).
    2. Phân tích ma trận nhầm lẫn (Confusion Matrix).
    3. Tìm các mẫu đoán sai nghiêm trọng (High-Confidence Errors).
    4. Phân tích sự hiện diện của motif GGGCGG (GC-box) trong các mẫu đoán sai.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    # ── 1. Thu thập dự đoán ──
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            shape_features = batch["shape_features"].to(device)
            labels = batch["labels"]
            
            logits = model(input_ids, attention_mask, shape_features)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy() if hasattr(labels, "cpu") else labels)
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    print("\n" + "="*80)
    print(" BÁO CÁO PHÂN LOẠI CHI TIẾT (VALIDATION SET)")
    print("="*80)
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))
    
    # ── 2. Phân tích Ma trận nhầm lẫn ──
    cm = confusion_matrix(all_targets, all_preds)
    print("\n" + "="*80)
    print(" MA TRẬN NHẦM LẪN (Số lượng mẫu thực tế -> dự đoán)")
    print("="*80)
    print(f"{'True \\ Pred':<12} | " + " | ".join([f"{name:<8}" for name in class_names]))
    print("-" * 55)
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<12} | " + " | ".join([f"{val:<8d}" for val in row]))
        
    # Tính tỉ lệ nhầm lẫn lớn nhất
    print("\n các cặp nhầm lẫn phổ biến nhất:")
    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((class_names[i], class_names[j], cm[i, j], cm[i, j] / cm[i].sum()))
    # Sắp xếp theo số lượng mẫu bị nhầm giảm dần
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    for true_cls, pred_cls, count, pct in confusion_pairs[:5]:
        print(f"  - Thực tế là [{true_cls}] bị đoán nhầm thành [{pred_cls}]: {count} mẫu ({pct:.2%})")

    # ── 3. Tìm các mẫu đoán sai nghiêm trọng (High-Confidence Errors) ──
    # Đây là các mẫu đoán sai nhưng mô hình cực kỳ tự tin (prob > 0.8)
    print("\n" + "="*80)
    print(" PHÂN TÍCH LỖI SAI VỚI ĐỘ TỰ TIN CAO (Probability > 80%)")
    print("="*80)
    
    high_conf_errors = []
    for idx in range(len(all_targets)):
        true_label = all_targets[idx]
        pred_label = all_preds[idx]
        prob = all_probs[idx][pred_label]
        
        if true_label != pred_label and prob >= 0.8:
            high_conf_errors.append({
                'idx': idx,
                'header': headers_test[idx],
                'sequence': seq_test[idx],
                'true_name': class_names[true_label],
                'pred_name': class_names[pred_label],
                'confidence': prob,
                'probs': all_probs[idx]
            })
            
    print(f"Tìm thấy {len(high_conf_errors)} mẫu đoán sai nhưng có độ tự tin >= 80%.")
    
    # In ra 5 mẫu tiêu biểu
    for i, err in enumerate(high_conf_errors[:5]):
        print(f"\nMẫu sai #{i+1} (Index: {err['idx']}):")
        print(f"  Header:    {err['header']}")
        print(f"  Sequence:  {err['sequence']}")
        print(f"  Thực tế:   {err['true_name']}  |  Đoán sai: {err['pred_name']} (Tự tin: {err['confidence']:.2%})")
        prob_str = ", ".join([f"{class_names[c]}: {err['probs'][c]:.2%}" for c in range(len(class_names))])
        print(f"  Phân bố Prob: [{prob_str}]")
        
        # Kiểm tra xem có chứa GC-box (GGGCGG hoặc CCGCCC) không
        seq_upper = err['sequence'].upper()
        has_gc = "GGGCGG" in seq_upper or "CCGCCC" in seq_upper
        print(f"  Có chứa motif GC-box chuẩn (GGGCGG/CCGCCC)? {'Có ✅' if has_gc else 'Không ❌'}")
        
    # ── 4. Thống kê Sinh học: Tần suất Motif GC-box trong các nhóm dự đoán đúng/sai ──
    print("\n" + "="*80)
    print(" PHÂN TÍCH SINH HỌC: SỰ HIỆN DIỆN CỦA GC-BOX (GGGCGG / CCGCCC)")
    print("="*80)
    
    correct_indices = np.where(all_preds == all_targets)[0]
    incorrect_indices = np.where(all_preds != all_targets)[0]
    
    def calc_gc_box_pct(indices):
        has_gc_count = 0
        for idx in indices:
            seq = seq_test[idx].upper()
            if "GGGCGG" in seq or "CCGCCC" in seq:
                has_gc_count += 1
        return has_gc_count / max(len(indices), 1)
        
    # Thống kê trên từng lớp
    for cls_idx, cls_name in enumerate(class_names):
        cls_indices = np.where(all_targets == cls_idx)[0]
        cls_correct = np.intersect1d(cls_indices, correct_indices)
        cls_incorrect = np.intersect1d(cls_indices, incorrect_indices)
        
        gc_all = calc_gc_box_pct(cls_indices)
        gc_correct = calc_gc_box_pct(cls_correct)
        gc_incorrect = calc_gc_box_pct(cls_incorrect)
        
        print(f"Lớp [{cls_name}] (Tổng {len(cls_indices)} mẫu):")
        print(f"  - Tỉ lệ chứa GC-box trên toàn bộ lớp: {gc_all:.2%}")
        print(f"  - Tỉ lệ chứa GC-box trên các mẫu đoán ĐÚNG ({len(cls_correct)} mẫu): {gc_correct:.2%}")
        print(f"  - Tỉ lệ chứa GC-box trên các mẫu đoán SAI ({len(cls_incorrect)} mẫu): {gc_incorrect:.2%}")
        
    # Nhận xét sinh học
    print("\n Nhận xét sinh học từ số liệu trên:")
    print("  1. Nếu tỷ lệ GC-box ở mẫu đoán SAI của lớp [Negative] rất cao, tức là Negative chứa chuỗi giống SP nhưng không liên kết thực tế, mô hình bị lừa bởi Sequence và cần dựa nhiều hơn vào DNAshape.")
    print("  2. Nếu tỉ lệ GC-box ở mẫu đoán ĐÚNG của SP1/2/4 vượt trội so với đoán SAI, nghĩa là mô hình đang dùng GC-box làm đặc trưng quyết định chính (có thể gây nhầm lẫn chéo giữa SP1, SP2, SP4 do cả 3 đều bám vào GC-box).")

    return all_preds, all_targets, all_probs, high_conf_errors

# Hàm hỗ trợ load lại model và data khi notebook bị restart
def load_and_diagnose(checkpoint_path="outputs_gcmab_safe/models/best_gcmab_safe.pt"):
    """
    Hàm này được gọi nếu user chạy trong cell mới sau khi restart kernel.
    Nó sẽ tự động load lại model, tokenizer, datasets, chia split theo seed 42 và chạy chẩn đoán.
    """
    print(f"Đang kiểm tra checkpoint tại {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Không tìm thấy checkpoint tại {checkpoint_path}. Hãy chắc chắn đường dẫn đúng.")
        return
        
    # Load checkpoint để kiểm tra cấu hình lưu kèm
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"✅ Load thành công checkpoint từ Epoch {checkpoint['epoch']} (Val Acc: {checkpoint['val_acc']:.2%})")
    
    # Đoạn này giả lập môi trường và cấu trúc dữ liệu của Script 19
    # Vì thế khuyên dùng là chạy TRỰC TIẾP trên các biến RAM của notebook cũ nếu chưa tắt.
    print("\n💡 LỜI KHUYÊN: Nếu bạn vừa chạy xong cell training và chưa tắt Kernel,")
    print("   hãy gọi trực tiếp hàm: run_diagnostics(model, test_loader, seq_test, headers_test, shape_test_norm, cfg.CLASS_NAMES)")
    print("   để tránh phải load lại toàn bộ dữ liệu.")
