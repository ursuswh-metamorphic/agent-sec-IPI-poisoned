# Dense top-100 rồi BM25

- CEM lọt top-5: **14/100; BM25 chấm trong 100 ứng viên dense, dùng thống kê toàn corpus.**
- [Code dùng chung](../shared/cascade_defense.py): `reverse_consensus, subset_bm25`. Không nhân bản code giữa các phương pháp.
- [Kết quả riêng](results.json): trích nguyên chỉ số từ lượt local đã chạy, không chạy thí nghiệm mới khi đóng gói.
- Kết quả đã lưu: [results.json](results.json). Đây là replay local; dữ liệu Kaggle gốc không nằm trong thư mục rút gọn.

Chạy từ root repo sau khi cài dependencies:

```bash
python defenses/shared/evaluate_cascade.py --root <thu_muc_du_lieu_kaggle> --output <thu_muc_ket_qua_moi>
```

Script chạy cả nhóm phương pháp để giữ đối chứng cùng điều kiện. Dùng thư mục output mới nếu đã chạy trước đó. Mỗi query có một target; không chạy lại CEM hay embedding.
