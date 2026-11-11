# AGENT.md

## Mục đích của repository

Repository này nghiên cứu **retrieval poisoning / indirect prompt injection (IPI)** trong hệ thống RAG. Luồng thí nghiệm chính là:

1. Dùng một bộ truy xuất dense để xếp hạng tài liệu theo query.
2. Dùng CEM để tối ưu một prefix cho tài liệu mục tiêu, nhằm làm tài liệu đó tăng điểm retrieval.
3. Đo xem tài liệu bị đầu độc có lọt vào top-k hay không.
4. Thử các cơ chế phòng thủ ở bước retrieval hoặc ngay trước bước RAG.

Đây là repository nghiên cứu/thử nghiệm. Không nên hiểu các kết quả hiện có là bằng chứng hệ thống đã chống được mọi attacker thích nghi hoặc bảo đảm câu trả lời của LLM đúng.

## Hai nhánh hiện có

Hai nhánh remote không có commit gốc chung và phục vụ hai hướng khác nhau.

### `main`

File chính là [`ipi-def-v2.ipynb`](./ipi-def-v2.ipynb). Đây là notebook Kaggle end-to-end:

- Tải và kiểm tra artifact gốc từ Zenodo.
- Chuẩn bị workflow retrieval, CEM và RAG.
- Chạy baseline sạch và baseline có tài liệu CEM.
- Đo ranking, recall, nDCG, MRR và có thể chạy RAG bằng FLAN-T5-small.
- Có phần thử nghiệm `Dual-Filter Chunk Consistency` ở sau retrieval, trước RAG.

Dual-Filter gồm:

- Filter 1: chia tài liệu thành chunk và kiểm tra `max(cosine) - min(cosine) > 0.4`.
- Filter 2: nếu qua Filter 1, tính perplexity GPT-2 trên từng chunk; trung bình lớn hơn `200` thì gắn nhãn đáng ngờ.

Các cell defense trong notebook ghi rõ là phần thử nghiệm mới và chưa có kết quả mặc định được xác nhận. Ngưỡng chưa được calibration trên tập dữ liệu độc lập.

### `origin/hybrid_retriever`

Nhánh này là gói replay/evaluation đã chạy trên output Kaggle cũ. Nội dung chính nằm trong thư mục [`defenses/`](./defenses/) và báo cáo [`report_ket_qua_chay_hybrid_retriever.md`](./report_ket_qua_chay_hybrid_retriever.md).

Nó so sánh:

- allowlist + hash (`trusted_only`);
- rule detector;
- BM25 đơn lẻ;
- hybrid RRF;
- BM25 top-100 rồi dense;
- dense top-100 rồi BM25;
- budgeted RRF trên 100 ứng viên dense.

Kết quả mẫu dùng 100 query SciFact và một tài liệu CEM mục tiêu cho mỗi query. Dense baseline có tài liệu mục tiêu trong top-5 ở `100/100` query. Các kết quả JSON trong từng thư mục `defenses/*` là kết quả đã lưu/replay, không tự động có nghĩa là benchmark sẽ được chạy lại khi clone repo.

## Kết quả tham khảo đã ghi lại

| Phương pháp | Target CEM còn trong top-5 | Recall@5 | nDCG@5 |
|---|---:|---:|---:|
| Dense baseline | 100/100 | 0.818 | 0.540 |
| Allowlist + hash | 0/100 | 0.820 | 0.735 |
| Hybrid RRF | 25/100 | 0.862 | 0.781 |
| BM25 top-100 rồi dense | 38/100 | 0.843 | 0.690 |
| Dense top-100 rồi BM25 | 14/100 | 0.810 | 0.726 |
| Budgeted RRF | 22/100 | 0.842 | 0.782 |
| BM25 đơn lẻ | 11/100 | 0.805 | 0.708 |
| Rule detector | 100/100 | 0.818 | 0.540 |

Diễn giải đúng các số liệu:

- `target_hits_at_k` thấp nghĩa là ít tài liệu CEM lọt top-k trong kịch bản đã thử.
- Recall/nDCG đo chất lượng tài liệu liên quan, không phải độ chính xác câu trả lời của LLM.
- Allowlist + hash đạt `0/100` nhờ giả định corpus sạch đã được duyệt còn tài liệu CEM mới chưa được đăng ký; đây không phải detector nội dung tổng quát.
- Rule detector trong replay này không chặn được tài liệu CEM.

## Cách làm việc với repository

Kiểm tra nhánh và trạng thái trước khi sửa:

```bash
git status --short --branch
git branch -avv
```

Để xem nhánh hybrid mà không làm thay đổi working tree hiện tại:

```bash
git show origin/hybrid_retriever:<path>
git diff main..origin/hybrid_retriever
```

Nếu cần làm việc trực tiếp trên nhánh hybrid, tạo local branch từ remote sau khi chắc chắn working tree sạch:

```bash
git switch -c hybrid_retriever --track origin/hybrid_retriever
```

### Chạy notebook `main`

- Mở `ipi-def-v2.ipynb` trong Kaggle.
- Bật Internet và GPU nếu muốn chạy đầy đủ workflow.
- Giữ profile `demo` để kiểm tra nhanh; dùng `nfcorpus` cho thí nghiệm lớn hơn.
- Chạy các cell từ đầu để khôi phục biến và model state.
- Không bỏ qua checksum Zenodo hoặc tự đổi các checksum đã khai báo.
- Output được ghi vào thư mục kết quả riêng; không ghi model weights/cache vào repository.

### Chạy replay ở nhánh hybrid

Code replay cần thư mục dữ liệu Kaggle đã có sẵn, thường gồm `usenix_1226_AE` và `usenix_1226_AE_work_gte`. Xem README tương ứng trước khi chạy:

- [`defenses/shared/README_VI.md`](./defenses/shared/README_VI.md)
- [`defenses/shared/README_HYBRID_VI.md`](./defenses/shared/README_HYBRID_VI.md)
- [`defenses/shared/README_CASCADE_VI.md`](./defenses/shared/README_CASCADE_VI.md)

Các test replay:

```bash
python -m unittest discover -s defenses/shared -p "test_*.py" -v
```

Các script evaluation không tự tải corpus Kaggle và không chạy lại CEM. Chúng đọc điểm/output đã lưu, sau đó tính lại ranking và metric.

## Quy tắc diễn giải và phát triển

- Không gọi payload hiện tại là bằng chứng attacker đã điều khiển LLM; nhiều phép đo chỉ kiểm tra ranking retrieval.
- Không dùng kết quả trên cùng 100 query để tuyên bố khả năng tổng quát hóa; cần held-out queries và attacker thích nghi cho đánh giá mạnh hơn.
- Khi thêm defense, phải báo riêng: tỷ lệ chặn target, false positive trên tài liệu sạch, recall/nDCG trước và sau, số query bị rỗng context, và chi phí runtime.
- Không để tác giả tài liệu tự thêm cờ `trusted`; registry/hash phải được quản lý từ nguồn được duyệt bên ngoài corpus truy xuất.
- Defense không thay thế kiểm soát quyền tool. Nếu agent có quyền đọc file, gửi mail hoặc thực thi code, các quyền đó vẫn phải được kiểm tra tại nơi thực thi.
- Giữ nguyên provenance, cấu hình, hash input và giới hạn của từng lượt chạy trong output/report.
- Trước khi kết luận một phương pháp tốt hơn, kiểm tra cả chất lượng trên corpus sạch lẫn khả năng giảm target hit.

## Trạng thái lúc ghi file này

- Nhánh local hiện tại: `main`.
- Working tree ban đầu sạch.
- `origin/main` trỏ tới commit thêm notebook Dual-Filter.
- `origin/hybrid_retriever` trỏ tới commit thêm local hybrid retrieval defenses.
- Không tự động chạy Kaggle/notebook trong quá trình tạo file hướng dẫn này.
