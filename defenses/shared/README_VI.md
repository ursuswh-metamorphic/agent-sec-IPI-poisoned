# Chạy lại thử nghiệm phòng thủ local

Thư mục rút gọn chỉ giữ code và kết quả đã chạy. Các file điểm vector, corpus và output Kaggle gốc không được đóng gói ở đây. Muốn chạy lại, truyền `--root` tới thư mục dữ liệu Kaggle đã tải về máy.

Gói này thử hai biện pháp: bộ lọc câu lệnh đáng ngờ và cổng chỉ nhận tài liệu đã duyệt. Nó dùng lại điểm tìm kiếm đã lưu để đo ảnh hưởng của việc lọc. Không cần GPU, không cài thêm thư viện và không chạy lại CEM.

## 1. File trong gói

| File | Công việc |
|---|---|
| `defenses.py` | Kiểm tra nội dung, kiểm tra ID/hash với registry, loại tài liệu không được phép |
| `evaluate_defenses.py` | Đọc output Kaggle, áp dụng bộ lọc vào điểm đã lưu, tính hit@k/recall/nDCG |
| `test_defenses.py` | Kiểm tra code, bao gồm một số điểm yếu đã biết của rule |
| `run_kaggle.py` | Lệnh chạy ngắn, tìm dữ liệu trong output và tạo kết quả mới |

## 2. Chạy trên Kaggle

Đưa bốn file Python trong gói vào `/kaggle/working/defense_replay/`. Đường dẫn bên dưới giả định notebook đã tạo các thư mục `/kaggle/working/usenix_1226_AE/` và `/kaggle/working/usenix_1226_AE_work_gte/` như lượt được kiểm tra.

```bash
%%bash
set -e
python -m unittest discover -s /kaggle/working/defense_replay -p test_defenses.py -v
python /kaggle/working/defense_replay/run_kaggle.py \
  --root /kaggle/working \
  --output /kaggle/working/defense_results \
  --approve-clean-benchmark
```

Nếu dùng lại output đã lưu qua Add Input, thay `--root` bằng thư mục cha chứa `usenix_1226_AE` và `usenix_1226_AE_work_gte` trong input đó. Kết quả vẫn ghi vào `/kaggle/working/defense_results` để không ghi vào input chỉ đọc.

Tham số `--approve-clean-benchmark` có nghĩa: cho phép thí nghiệm coi toàn bộ corpus SciFact ban đầu là tập đã được duyệt. Đây là giả định benchmark, không phải bước tự động duyệt dữ liệu cho hệ thống thật.

Script sẽ dừng nếu dữ liệu thiếu hoặc thư mục kết quả đã có `results.json`, tránh vô tình ghi đè lượt cũ. Đổi `--output` để chạy một lượt mới.

## 3. Đọc kết quả

Mở `defense_results/summary.json`:

- `policies.none`: kết quả khi có tài liệu CEM nhưng chưa có bộ lọc.
- `policies.rules`: kết quả sau bộ lọc câu lệnh đáng ngờ.
- `policies.trusted_only`: kết quả khi chỉ nhận tài liệu đã ghi danh và khớp hash.
- `target_hits_at_k`: trong 100 query, có bao nhiêu lần tài liệu CEM vào top-k.
- `blocked_attack_documents`: số tài liệu CEM bị chặn.
- `blocked_clean_documents`: số tài liệu SciFact ban đầu bị chặn.
- `clean_recall_at_k`, `clean_ndcg_at_k`: chất lượng tìm tài liệu liên quan sau lọc.
- `exported_attack_recall_mismatches`: phải là danh sách rỗng nếu kết quả tính lại khớp file metric gốc.

Kết quả thực tế trên phiên bản Kaggle 344612066: không lọc = 100/100 lọt top-5; rule = 100/100; trusted-only = 0/100. Chế độ trusted-only đạt kết quả này nhờ giả định tập sạch đã duyệt và tài liệu mới chưa được duyệt.

## 4. Ghép cổng lọc vào luồng dùng tài liệu

Hàm `filter_results` nhận điểm và nội dung tài liệu, trả điểm đã lọc cùng lý do chặn. Có thể gọi sau retrieval và trước khi chọn top-k gửi vào LLM. Cần lấy nhiều ứng viên hơn k để có tài liệu thay thế khi bị lọc.

```python
from defenses import filter_results

# candidate_scores: {query_id: {document_id: score}}
# documents: {document_id: {"title": ..., "text": ...}}
safe_scores, blocked = filter_results(
    candidate_scores,
    documents,
    policy="rules",
)

# LLM chỉ nhận nội dung từ các ID đã được phép qua cổng lọc.
top_ids = sorted(safe_scores[query_id],
                 key=safe_scores[query_id].get, reverse=True)[:5]
```

Để dùng trusted-only, truyền `policy="trusted_only"` và `registry` là dictionary `documents` đọc từ registry đã được quản lý bên ngoài dữ liệu truy xuất.

Code không tự chạy lệnh xuất hiện trong tài liệu. Cổng lọc này cũng không thay thế kiểm soát quyền tool: nếu ứng dụng có công cụ đọc file/gửi email/thực thi code thì vẫn phải kiểm soát tại nơi thực thi.

Không triển khai cổng trusted bằng cách để tác giả tài liệu tự thêm `trusted: true`; không để agent cập nhật registry; không lấy chính kho đã bị đầu độc để tạo danh sách duyệt. Việc xác minh nội dung khớp hash phải dựa vào bản được duyệt trước đó.

## 5. Phạm vi kết luận

Đây là phép replay trên điểm của lượt Kaggle hiện có, mỗi query ghép thêm một tài liệu CEM mục tiêu. Chưa có LLM, chưa có agent và chưa tối ưu lại CEM sau phòng thủ. Tài liệu CEM ở lượt này chứa câu đánh dấu vô hại, nên chỉ đo sức chống chịu của bước truy xuất và kiểm soát đầu vào.
