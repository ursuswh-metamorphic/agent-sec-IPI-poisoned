# Dense + BM25 qua RRF

- CEM lọt top-5: **25/100; hằng số 60, cửa sổ 1.000 mỗi nhánh, trọng số bằng nhau.**
- [Code dùng chung](../shared/hybrid_defense.py): `reciprocal_rank_fusion`. Không nhân bản code giữa các phương pháp.
- [Kết quả riêng](results.json): trích nguyên chỉ số từ lượt local đã chạy, không chạy thí nghiệm mới khi đóng gói.
- Kết quả đã lưu: [results.json](results.json). Đây là replay local; dữ liệu Kaggle gốc không nằm trong thư mục rút gọn.

Chạy từ root repo sau khi cài dependencies:

```bash
python defenses/shared/evaluate_hybrid.py --root <thu_muc_du_lieu_kaggle> --output <thu_muc_ket_qua_moi>
```

Script chạy cả nhóm phương pháp để giữ đối chứng cùng điều kiện. Dùng thư mục output mới nếu đã chạy trước đó. Mỗi query có một target; không chạy lại CEM hay embedding.
