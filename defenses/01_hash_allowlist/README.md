# Danh sách duyệt + hash

- CEM lọt top-5: **0/100; giả định toàn bộ corpus clean được duyệt và tài liệu mới chưa được duyệt.**
- [Code dùng chung](../shared/defenses.py): `trust_reasons, content_hash`. Không nhân bản code giữa các phương pháp.
- [Kết quả riêng](results.json): trích nguyên chỉ số từ lượt local đã chạy, không chạy thí nghiệm mới khi đóng gói.
- Kết quả đã lưu: [results.json](results.json). Đây là replay local từ điểm vector đã lưu; dữ liệu Kaggle gốc không nằm trong thư mục rút gọn.

Chạy từ root repo sau khi cài dependencies:

```bash
python defenses/shared/run_kaggle.py --root <thu_muc_du_lieu_kaggle> --output <thu_muc_ket_qua_moi> --approve-clean-benchmark
```

Script chạy cả nhóm phương pháp để giữ đối chứng cùng điều kiện. Dùng thư mục output mới nếu đã chạy trước đó. Mỗi query có một target; không chạy lại CEM hay embedding.
