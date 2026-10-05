# Thử dense trước, BM25 sau và RRF giới hạn ứng viên

Đây là code chạy local trên điểm dense đã lưu. Repo rút gọn không chứa snapshot dữ liệu Kaggle; kết quả mẫu đã chạy được lưu trong từng thư mục phương pháp.

## 1. Kịch bản gốc

Lượt Kaggle đã dùng GTE-modernbert-base và CEM để tạo một tài liệu nhắm vào mỗi query. Code chấm tài liệu mục tiêu cùng corpus sạch và xem nó có lọt top-5 không. Trong 100 query của lượt này, dense cho 100/100 target vào top-5. Đây là baseline của lượt thực nghiệm này, không phải tỷ lệ tổng quát của bài báo.

Script mới dùng lại điểm vector đã lưu và tính mới BM25/ranking. Không chạy lại CEM, không chạy model dense, không có LLM sinh câu trả lời, không thay đổi notebook online.

## 2. Hai biến thể mới

`dense_100_then_bm25`: lấy 100 tài liệu có điểm dense cao nhất, rồi chỉ chấm BM25 cho 100 tài liệu đó và chọn top-5. IDF và độ dài trung bình vẫn lấy từ index toàn corpus, không dựng BM25 chỉ trên 100 tài liệu. Khi điểm BM25 bằng nhau, dùng dense để phân xử; nếu mọi điểm BM25 bằng 0, thứ tự quay về dense.

`budgeted_rrf_100`: cũng lấy 100 ứng viên dense, nhưng kết hợp thứ hạng với trọng số 25% dense và 75% BM25, hằng số 60. Chỉ ứng viên có điểm BM25 dương được nhận phiếu lexical. Các ứng viên còn lại vẫn có điểm dense. Khi không có ứng viên BM25 dương, quay về thứ tự dense. Đây là một cấu hình đề xuất trong project, không tuyên bố là thuật toán mới trong nghiên cứu.

```text
score(d) = 0.25 / (60 + rank_dense(d))
         + 0.75 / (60 + rank_bm25_positive(d))
```

Mục tiêu là giảm chi phí BM25 và bớt phụ thuộc điểm dense đã bị CEM tối ưu, nhưng vẫn duy trì nhánh dense khi từ khóa không hữu ích. Fallback không bảo đảm an toàn: tài liệu CEM có thể quay lại. Luôn trả đủ 5 ứng viên nếu có đủ ứng viên không đồng nghĩa có đủ bằng chứng để LLM trả lời đúng.

Top-100, trọng số 25/75 và hằng số 60 được cố định trước khi chạy, không quét tham số để chọn kết quả. Các phương pháp cũ cũng được tính lại để so sánh công bằng trong cùng môi trường.

## 3. Chạy trên Kaggle

Giải nén code vào `/kaggle/working/defense_replay/`. Đảm bảo root có hai thư mục output `usenix_1226_AE` và `usenix_1226_AE_work_gte`.

```bash
%%bash
set -e
python -m pip install -r /kaggle/working/defense_replay/requirements_cascade.txt
python -m unittest discover -s /kaggle/working/defense_replay -v
python /kaggle/working/defense_replay/evaluate_cascade.py \
  --root /kaggle/working \
  --output /kaggle/working/cascade_results
```

Nếu dùng Add Input để lấy output đã lưu, đổi root thành thư mục cha chứa hai thư mục trên. Output mới phải đặt trong `/kaggle/working/`. Script không ghi đè thư mục đã có results.json. Không cần GPU.

## 4. Cách đo chi phí

Mỗi query/method có một lượt warm-up và năm lượt đo, thứ tự method được xáo trộn bằng seed 42. Lấy trung vị năm lượt cho từng query, rồi tính mean/p50/p95 trên 100 trung vị đó. Các mẫu thô nằm trong `results.json`.

Khoảng đo bao gồm chấm BM25 nếu method có dùng, chọn ứng viên, gộp/xếp hạng và lấy top-5 trên RAM. Không bao gồm đọc file, query encoding, dense search, truyền mạng hoặc xây index. Đây là chi phí CPU thêm vào sau khi đã có dense scores, không phải end-to-end latency hoặc p95 phục vụ nhiều người dùng đồng thời.

Mỗi tình huống attacked chỉ thêm một target như phép đo gốc. Vì target khác nhau giữa các query, script dựng một BM25 index mới cho mỗi tình huống để giữ đúng IDF. Thời gian dựng index được ghi riêng, không tính là chi phí online mỗi request. Khi triển khai phải quản lý index ngoài đường xử lý request; `rank-bm25` hiện là thư viện benchmark in-memory, không phải dịch vụ chỉ mục cập nhật tự động.

## 5. Chất lượng và khả năng trả lời

`queries_with_relevant_at_5` đếm số query tìm được ít nhất một tài liệu có nhãn liên quan SciFact. Đây là proxy khả năng có bằng chứng trả lời, không phải độ chính xác câu trả lời của LLM. Script còn đo recall/nDCG, query mất/được tài liệu liên quan so với dense và query trả dưới 5 kết quả.

Một phương pháp giảm CEM nhưng làm nhiều câu hỏi không còn bằng chứng liên quan thì không thể xem là thắng chỉ từ attack hit. Ngược lại, chỉ trả đủ 5 cũng không chứng minh câu hỏi trả lời được.

## 6. File bằng chứng

- `summary.json`: kết quả, runtime và timing tổng hợp.
- `results.json`: từng query, top-5, số đo thời gian thô, cấu hình và hash file đầu vào/code.
- `per_query.csv`: 1.200 dòng dữ liệu dự kiến = 100 query × 2 tình huống × 6 method, thuận tiện đối chiếu.
- `cascade_defense.py`: logic hai biến thể mới, không sử dụng nhãn attacker khi chấm điểm.
- `evaluate_cascade.py`: kịch bản kiểm tra và benchmark.
- `test_cascade.py`: kiểm tra cửa dense, lexical fallback, thống kê BM25 toàn corpus và tham số.

## 7. Giới hạn

Chưa có tấn công thích nghi với phòng thủ, chưa có query held-out mới và chưa có đo câu trả lời LLM. Payload CEM vẫn là câu đánh dấu vô hại. Không được coi số liệu này là khả năng phòng thủ ngoài thực tế hoặc tỷ lệ chung của bài báo.

Lượt cục bộ này dùng Python 3.12.14, numpy 2.3.5 và rank-bm25 0.2.2 vì Python 3.11 cũ không khởi động được trong phiên làm việc. Kết quả các phương pháp cũ phải được đối chiếu lại trước khi so sánh với lượt trước.
