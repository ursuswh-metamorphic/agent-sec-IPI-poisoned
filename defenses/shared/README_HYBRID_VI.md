# Thử phòng thủ bằng BM25 và tìm kiếm kết hợp

Đây là code chạy local trên điểm dense đã lưu. Repo rút gọn không chứa snapshot dữ liệu Kaggle; kết quả mẫu đã chạy được lưu trong từng thư mục phương pháp.

Bản này không dùng danh sách tài liệu duyệt hoặc hash để cho phép/chặn tài liệu. Tài liệu mới được đánh giá bằng độ phù hợp với câu hỏi.

## Cách hoạt động

Vector retrieval là hệ thống tìm kiếm ngữ nghĩa đã có trong lượt Kaggle. BM25 là nhánh tìm kiếm dựa trên các từ trong câu hỏi và tài liệu. Cả hai cùng đưa ra danh sách xếp hạng, rồi code thử hai cách kết hợp:

- `rrf_60`: cộng điểm theo thứ hạng của mỗi nhánh. Tài liệu xếp tốt ở cả hai nhánh thường có lợi thế. Không cộng trực tiếp điểm vector với BM25 vì chúng khác thang đo.
- `consensus_bm25_100`: tài liệu phải thuộc top-100 BM25 rồi mới được xếp hạng bằng điểm vector. Đây là cửa lọc theo từng câu hỏi, không phải danh sách duyệt cố định.

Ngoài ra, `dense` là mốc so sánh và `bm25_only` là phép đối chứng chỉ dùng từ khóa. Không có biện pháp nào ở đây chứng minh một tài liệu không chứa prompt injection.

## Chạy trên Kaggle

Giải nén gói vào `/kaggle/working/defense_replay/`, sau đó chạy:

```bash
%%bash
set -e
python -m pip install -r /kaggle/working/defense_replay/requirements_hybrid.txt
python -m unittest discover -s /kaggle/working/defense_replay -v
python /kaggle/working/defense_replay/evaluate_hybrid.py \
  --root /kaggle/working \
  --output /kaggle/working/hybrid_defense_results
```

`--root` phải chứa hai thư mục output cũ `usenix_1226_AE/` và `usenix_1226_AE_work_gte/`. Nếu dùng Add Input để lấy output đã lưu, thay root bằng thư mục cha tương ứng trong `/kaggle/input/`. Output mới vẫn ghi vào `/kaggle/working/`.

Không cần GPU hoặc chạy lại CEM. Nếu output đã tồn tại `results.json`, script yêu cầu đổi thư mục output để giữ kết quả cũ.

## File và output

| File | Vai trò |
|---|---|
| `hybrid_defense.py` | Chuẩn hóa từ, gọi BM25, trộn thứ hạng RRF hoặc lọc theo top-100 BM25 |
| `evaluate_hybrid.py` | Đọc output Kaggle, thử cả kho sạch và kho có tài liệu CEM, tính metric |
| `test_hybrid.py` | 7 test cho hành vi kết hợp, tài liệu mới và truy vấn không khớp |
| `evaluate_defenses.py`, `defenses.py` | Các hàm đọc dữ liệu và tính metric được dùng lại; thử nghiệm hybrid không gọi cơ chế kiểm tra registry |
| `requirements_hybrid.txt` | Phiên bản numpy và rank-bm25 dùng để chạy |
| `summary.json` | Kết quả tổng hợp |
| `results.json` | Kết quả từng query, top-5 và thứ hạng tài liệu mục tiêu |

Trong output có hai nhóm `clean` và `attacked`. Mỗi nhóm có bốn cách xếp hạng. `target_hits_at_k` càng thấp càng ít tài liệu CEM lọt top-5; `clean_recall_at_k` và `clean_ndcg_at_k` càng cao càng tốt cho tìm tài liệu liên quan. Khi không có tấn công, số hit tài liệu CEM bằng 0 vì chưa đưa chúng vào kho.

Trường hash ở `results.json` chỉ ghi nhận phiên bản file đầu vào để tái lập. Nó không tham gia quyết định xếp hạng hoặc chặn tài liệu.

## Cấu hình đã cố định trước khi chạy

BM25Okapi: `k1=1.5`, `b=0.75`, `epsilon=0.25`. Chuẩn hóa NFKC, chuyển chữ thường và tách từ bằng biểu thức Unicode `\w+`; chưa stemming hoặc loại stopword. Cùng cách chuẩn hóa áp dụng cho câu hỏi và tài liệu.

RRF: hai nhánh có trọng số bằng nhau, `rank_constant=60`, lấy tối đa 1.000 tài liệu mỗi nhánh. Công thức mỗi nhánh là `1 / (60 + rank)` với rank bắt đầu từ 1. Nhánh BM25 chỉ trả tài liệu có điểm dương.

Consensus: cửa BM25 lấy 100 tài liệu đầu. Không tự nới cửa nếu ít kết quả; có thể trả dưới 5 tài liệu khi câu hỏi không khớp từ khóa.

Đối với mỗi query bị tấn công, index BM25 được dựng trên toàn bộ 5.183 tài liệu sạch cộng đúng một tài liệu CEM mục tiêu, khớp kịch bản đánh giá trước. Dense dùng điểm top-1.000 sạch đã lưu cộng điểm tài liệu đó. RRF có thể lấy thêm tài liệu từ BM25 nằm ngoài top-1.000 dense.

## Giới hạn

Đây là replay điểm vector và tính mới BM25 trên output thật, chạy cục bộ; chưa sửa notebook công khai. CEM trong lượt cũ tối ưu vector, chưa tối ưu lại trước hybrid. Không được suy ra các con số là khả năng chống đối thủ thích nghi. Payload vẫn là câu đánh dấu retrieval vô hại, chưa có LLM/agent làm theo hướng dẫn trong tài liệu.

BM25 có thể bỏ lỡ câu hỏi dùng từ đồng nghĩa hoặc cách diễn đạt khác. Tài liệu độc chứa đủ từ khóa cũng có thể xếp cao. Cần xem cả bảng sạch và bảng bị tấn công để quyết định cấu hình; không chỉ chọn cột ít tài liệu CEM nhất.

Nguồn triển khai: [rank-bm25](https://github.com/dorianbrown/rank_bm25). Công thức RRF và ý nghĩa hằng số thứ hạng: [tài liệu Elasticsearch](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion).
