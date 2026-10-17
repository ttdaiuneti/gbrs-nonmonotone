# Chẩn đoán rò rỉ nhãn (SteelPlates) và metric suy biến (PC3)

*Tài liệu này ghi lại quá trình kiểm tra cho Major #2 và Major #3. Mọi con số
đều tái lập được bằng `R1/code/diagnose_leakage.py`.*

## 1. SteelPlates — rò rỉ nhãn qua cột chỉ báo loại lỗi

**Triệu chứng.** Trong bảng gốc, NRS trên SteelPlates đạt kNN-3 = **1.0000 ±
0.0000** và SVM = **1.0000**, với `full_svm` = 1.0. Độ chính xác tuyệt đối trên
một bài phân lớp thực (n = 1941) gần như luôn là dấu hiệu của target leakage.

**Cơ chế.** Bộ `steel-plates-fault` (OpenML v1) là phiên bản **nhị phân** (2
lớp) của UCI Steel Plates Faults. Bản gốc UCI có 27 đặc trưng liên tục **cộng 7
cột chỉ báo one-hot loại lỗi** (Pastry, Z_Scratch, K_Scratch, Stains, Dirtiness,
Bumps, Other_Faults). Khi rút gọn về nhị phân, các cột chỉ báo này vẫn còn trong
tập đặc trưng. Vì mỗi tấm thép có đúng một loại lỗi, **nhãn là một hàm tất định
của các cột chỉ báo**.

**Bằng chứng định lượng.**
- Sau MinMax, có đúng **8 cột nhị phân {0,1}**: V12, V13, V28, V29, V30, V31,
  V32, V33.
- Reduct mà NRS chọn là **{V28, V29, V30, V31, V32, V33}** — toàn bộ là cột chỉ
  báo, γ* = 1.0000.
- Nhóm các mẫu theo mẫu bit của 8 cột nhị phân: có **14 nhóm, cả 14 đều thuần
  lớp** → **100.00% mẫu nằm trong nhóm bit thuần lớp**. Tức chỉ cần các cột chỉ
  báo là đã phân lớp hoàn hảo.

**Kiểm chứng bằng cách sửa.** Loại 8 cột nhị phân rồi chạy lại:

| Cấu hình | NRS \|R\| | kNN-3 | SVM |
|---|---|---|---|
| Có cột chỉ báo (rò rỉ) | 6 | **1.0000** | **1.0000** |
| Đã loại 8 cột chỉ báo | 5 | **0.6533** | **0.6574** |

Độ chính xác sụp từ 1.0000 xuống ~0.65 — xác nhận điểm hoàn hảo hoàn toàn do rò
rỉ. **Khuyến nghị:** dùng phiên bản đã loại cột chỉ báo (đề xuất) hoặc loại bộ
SteelPlates khỏi bảng phân lớp; giữ nó trong bảng audit đơn điệu là được vì audit
đo tính chất của γ, không phụ thuộc chất lượng nhãn.

## 2. PC3 — accuracy suy biến về bộ dự báo lớp đa số

**Triệu chứng.** SVM của **cả bốn** phương pháp (GBRS/DEG-GB/HTBS/NRS) bằng nhau
đúng **0.8976** trên PC3.

**Cơ chế.** PC3 (defect prediction) mất cân bằng nặng. Tỉ lệ lớp đa số =
**0.8976** — trùng khít accuracy của SVM. SVM-RBF với cấu hình mặc định dự đoán
toàn bộ về lớp đa số ⇒ accuracy = tỉ lệ lớp đa số, **không đo được năng lực phân
biệt**.

**Bằng chứng định lượng.** Với reduct GBRS (|R| = 1):
- Accuracy (kNN-3) = 0.8631
- **Balanced accuracy (SVM) = 0.5000** (đúng mức ngẫu nhiên của bài nhị phân)

Một reduct 1 thuộc tính "đạt gần accuracy đầy đủ" thực chất chỉ đang đoán lớp đa
số. **Khuyến nghị:** bổ sung balanced accuracy / F1-macro / MCC cho toàn bảng
(Major #3), và đánh dấu rõ các ô mà accuracy suy biến.
