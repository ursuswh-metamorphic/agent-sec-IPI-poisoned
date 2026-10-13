# Báo cáo các lần cải tiến defense cho RAG

## Mục đích

File này ghi lại kết quả theo từng lần thay đổi defense trong notebook [`ipi-def-v2.ipynb`](./ipi-def-v2.ipynb). Mục tiêu là đánh giá đồng thời:

- khả năng giảm tài liệu poisoned lọt vào context;
- khả năng giảm tác động lên câu trả lời của LLM;
- false positive trên tài liệu sạch;
- chất lượng retrieval còn lại sau defense;
- tỷ lệ query bị mất toàn bộ context.

Không được đánh giá một defense chỉ bằng tỷ lệ chặn poison. Một defense chỉ hữu ích khi giữ được chất lượng tài liệu sạch và không làm hệ thống abstain quá thường xuyên.

## Cách hệ thống hoạt động và ý nghĩa của từng cơ chế

### 1. Baseline retrieval và tấn công CEM

Với mỗi query, dense retriever mã hóa query và các tài liệu thành vector rồi xếp
hạng theo cosine similarity. Tài liệu CEM được tạo bằng cách tối ưu một chuỗi
token ngắn ở đầu tài liệu để làm tăng điểm similarity với query. Vì vậy, tài liệu
độc có thể lọt vào top-k dù nội dung thật sự không liên quan.

Baseline dùng để đo hai điều: poison có lọt vào context hay không, và câu trả lời
của RAG có bị ép theo payload hay không. Đây là mốc so sánh trước khi thêm defense;
nó không có bước kiểm tra an toàn sau retrieval.

### 2. Dual-Filter Chunk Consistency

Sau khi retrieval chọn tài liệu ứng viên, mỗi tài liệu được chia thành các chunk
liền nhau tối đa 25 token. Defense tính hai tín hiệu:

1. **Semantic range:** mã hóa query và từng chunk bằng MiniLM, sau đó tính
   `max(cosine) - min(cosine)`. Tài liệu có một đoạn rất giống query nhưng các
   đoạn còn lại lệch mạnh sẽ có range lớn. Đây là dấu hiệu có nội dung chèn lệch
   chủ đề hoặc phần prefix bất thường.
2. **Perplexity:** tính perplexity GPT-2 riêng trên từng chunk rồi lấy trung bình.
   Perplexity cao nghĩa là chuỗi token khó đoán đối với mô hình ngôn ngữ; các
   chuỗi token ngẫu nhiên dùng trong CEM thường tạo tín hiệu này.

Ở cấu hình `evaluate_both_filters=True`, notebook lưu cả hai tín hiệu cho mọi
ứng viên. Một tài liệu bị đánh dấu khi semantic range vượt `threshold_1` hoặc
perplexity trung bình vượt `threshold_2`. Defense hoạt động ở vị trí
**post-retrieval, pre-RAG**: nó không sửa corpus và không thay đổi retriever,
chỉ quyết định tài liệu nào được đưa tiếp vào context.

Ý nghĩa: hai tín hiệu bổ sung cho nhau. Semantic range tìm sự không nhất quán
giữa các đoạn; perplexity tìm chuỗi ngôn ngữ bất thường. Đổi lại, tài liệu sạch
ngắn, chuyên ngành hoặc có văn phong lạ cũng có thể bị đánh dấu nhầm.

### 3. Candidate top-10 và final top-3

Thay vì chỉ lấy top-3 rồi lọc, notebook lấy **10 ứng viên đầu tiên**, chạy
defense trên cả 10, sau đó chọn 3 tài liệu an toàn để đưa vào RAG. Cách này tạo
ra một vùng dự phòng: nếu một tài liệu trong top-3 bị loại, các tài liệu ở vị trí
4–10 có thể thay thế.

Ý nghĩa: tăng candidate pool không tự làm detector chính xác hơn, nhưng làm hệ
thống ít phụ thuộc vào ba tài liệu đầu tiên và giảm nguy cơ mất toàn bộ context.
Chi phí là phải tính filter cho nhiều tài liệu hơn.

### 4. Refill tài liệu an toàn

Sau khi loại các ứng viên đáng ngờ, hệ thống duyệt phần còn lại theo thứ hạng và
điền đủ `final_k=3` bằng các tài liệu không bị đánh dấu. Nếu không đủ tài liệu an
toàn, context có thể ngắn hơn 3 tài liệu hoặc rỗng.

Ý nghĩa: refill giải quyết lỗi vận hành của detector quá chặt. Ở lượt 2, lọc
trực tiếp top-3 làm clean empty context `29%` và poisoned empty context `38%`.
Khi kết hợp top-10 với refill, các tỷ lệ này giảm xuống `6%`, rồi còn `0%` ở
cấu hình Balanced. Refill không làm poison an toàn; chỉ dùng các tài liệu đã qua
filter để thay thế tài liệu bị loại.

### 5. Prompt spotlighting

Trước khi gửi context vào RAG, notebook thêm chỉ dẫn định dạng để nói rõ phần
context là **dữ liệu tham khảo**, không phải instruction phải tuân theo. Cơ chế này
nhắm vào indirect prompt injection ở tầng RAG, nơi nội dung tài liệu có thể cố
gắng điều khiển model.

Ý nghĩa: spotlighting tạo ranh giới vai trò giữa system/user instruction và văn
bản retrieved. Nó là lớp giảm tác động lên câu trả lời, không phải detector poison;
vì vậy cần đánh giá riêng bằng ablation để biết đóng góp của nó.

### 6. Threshold sweep và chọn cấu hình Balanced

Notebook lưu tín hiệu semantic và perplexity trước, sau đó tính lại verdict với
nhiều cặp ngưỡng. Cách này tránh phải chạy lại CEM cho từng cặp threshold và cho
thấy trade-off giữa bảo mật và retrieval utility:

- `0.4/200`: chặn mạnh, poison exposure `0%`, nhưng clean FP `42,4%`.
- `0.6/300`: clean FP `10,5%`, poison detection `98%`, exposure `2%`.
- `0.8/500`: giữ tài liệu sạch tốt hơn, nhưng để lọt `31%` poison.

Chọn `0.6/300` có nghĩa là ưu tiên điểm cân bằng tạm thời: chấp nhận một tỷ lệ
poison rất nhỏ để giảm đáng kể việc loại nhầm tài liệu sạch. Đây chưa phải
calibration độc lập vì các ngưỡng được chọn sau khi xem cùng batch 100 query;
cần xác nhận trên clean calibration split và held-out query.

## Thiết lập chung

- Dataset: NFCorpus.
- Lượt mở rộng: 100 query đầu tiên.
- Corpus: 3.633 tài liệu sạch.
- Mỗi query có tối đa một tài liệu CEM mục tiêu.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- RAG model: `google/flan-t5-small`.
- Top-k cuối: 3.
- Dual-Filter: chunk 25 token NLTK, semantic range và mean GPT-2 perplexity.
- Cấu hình hiện tại: semantic range `0.6`, mean perplexity `300` (điểm vận hành
  Balanced, còn cần kiểm tra trên tập held-out).

## Lịch sử kết quả

| Lượt | Cấu hình | Query | Candidate/final | Poison exposure trước | Poison exposure sau | Target Yes ASR trước → sau | Clean empty context | Poisoned empty context | Clean FP rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Dual-Filter, không refill | 10 | 3/3 | 70% | 0% | 20% → 0% | 10% | 40% | 43,3% |
| 2 | Dual-Filter, không refill | 100 | 3/3 | 56% | 0% | 9% → 1% | 29% | 38% | 54,0% |
| 3 | Top-10 → filter → top-3, refill + spotlighting, ngưỡng `0,4/200` | 100 | 10/3 | 56% ở top-3; 66% trong top-10 | 0% | 9% → 0% | **6%** | **6%** | **42,4%** |
| 4 | Top-10 → filter → top-3, refill + spotlighting, ngưỡng Balanced `0,6/300` | 100 | 10/3 | 56% ở top-3; 66% trong top-10 | **2%** | 9% → **0%** | **0%** | **0%** | **10,5%** |

### Nguồn kết quả

- Lượt 1: `usenix_results_20261012T040649_851126Z.zip` — 10 query.
- Lượt 2: `usenix_results_20261012T164034_445939Z.zip` — 100 query, không refill.
- Lượt 3: [`result/usenix_results_20261013T073849_806751Z.zip`](./result/usenix_results_20261013T073849_806751Z.zip) — 100 query, candidate top-10, refill và prompt spotlighting.
- Lượt 4: [`result/usenix_results_20261013T085302_453423Z.zip`](./result/usenix_results_20261013T085302_453423Z.zip) — 100 query, cùng pipeline lượt 3 nhưng đổi ngưỡng sang `0,6/300`.

## Đánh giá lượt mới nhất

Lượt 3 đã cải thiện tính hữu dụng so với lượt 2:

- Clean empty context giảm từ `29%` xuống `6%`.
- Poisoned empty context giảm từ `38%` xuống `6%`.
- Target Yes ASR giảm từ `9%` xuống `0%`.
- Tài liệu poisoned được phát hiện `100/100`; các tài liệu poisoned đã lọt vào candidate pool đều bị chặn.

Trade-off vẫn còn lớn:

- `424/1000` cặp query–tài liệu sạch trong candidate top-10 bị chặn, tương đương false-positive rate `42,4%`.
- Chất lượng retrieval sạch sau defense giảm:

| Metric | Trước defense | Sau defense |
|---|---:|---:|
| Recall@3 | 0,0946 | 0,0260 |
| nDCG@3 | 0,3953 | 0,1294 |
| MRR@3 | 0,5050 | 0,2233 |

Kết luận của lượt 3: cơ chế candidate top-10 và refill đã giải quyết phần lớn vấn đề context rỗng, nhưng Dual-Filter vẫn loại nhầm quá nhiều tài liệu sạch. Chưa nên coi đây là cấu hình triển khai cuối.

## Đánh giá lượt 4: cấu hình Balanced

Lượt 4 dùng đúng notebook sau khi cập nhật commit `c7946c0`, với
`threshold_1=0.6`, `threshold_2=300`, `candidate_k=10`, `final_k=3`, refill và
prompt spotlighting được giữ nguyên.

| Metric | Trước defense | Sau defense |
|---|---:|---:|
| Clean Recall@3 | 0,0946 | **0,0724** |
| Clean nDCG@3 | 0,3953 | **0,3316** |
| Clean MRR@3 | 0,5050 | **0,4400** |

So với lượt 3, false positive giảm từ `42,4%` xuống `10,5%`, clean và poisoned
empty context đều giảm từ `6%` xuống `0%`. Detector phát hiện `98/100`
tài liệu poisoned; trong các tài liệu poisoned được truy xuất, phát hiện được
`64/66` (`96,97%`). Poison exposure sau defense còn `2%` và target ASR giảm từ
`9%` xuống `0%`. Thời gian filter là khoảng `103,3` giây cho 100 query.

Đây là điểm cân bằng tốt hơn lượt 3: utility retrieval giảm ít hơn trong khi
vẫn chặn phần lớn poison. Tuy nhiên, `thresholds_calibrated` vẫn là `false`.
Các con số `0,6/300` được chọn từ threshold sweep trên cùng batch 100 query,
chưa phải đánh giá độc lập.

### So sánh các ngưỡng trong lượt 4

| Ngưỡng semantic/perplexity | Clean FP | Poison detection | Poison exposure sau | Clean Recall@3 |
|---|---:|---:|---:|---:|
| `0,4/200` (aggressive) | 42,4% | 100% | 0% | 0,0260 |
| **`0,6/300` (Balanced)** | **10,5%** | **98%** | **2%** | **0,0724** |
| `0,8/500` (conservative) | 0,2% | 48% | 31% | 0,0936 |

Ngưỡng `0,8/500` giữ retrieval sạch tốt hơn nhưng để lọt quá nhiều poison,
không phù hợp làm cấu hình bảo vệ chính. Ngưỡng `0,6/300` là lựa chọn tạm
thời để tiếp tục thử nghiệm.

## Chuỗi cải tiến từ baseline 10 tài liệu

1. **Baseline 10 query:** chạy pipeline gốc trên 10 tài liệu/query để xác nhận
   CEM và đo được poison exposure `70%`, clean FP khoảng `43,3%`, clean empty
   `10%`, poisoned empty `40%`. Đây là smoke test nhỏ, chưa đủ để kết luận.
2. **Mở rộng lên 100 query:** tăng cỡ đánh giá để giảm dao động của kết quả.
   Lượt 2 cho thấy detector chặn được poison nhưng clean FP lên `54%` và
   empty context rất cao (`29%` clean, `38%` poisoned).
3. **Mở rộng candidate và refill:** lấy top-10 trước khi lọc, lọc tài liệu bị
   đánh dấu, sau đó refill tài liệu an toàn để đủ top-3; bật prompt spotlighting
   ở RAG. Lượt 3 làm empty context giảm còn `6%`, nhưng ngưỡng `0,4/200` quá
   chặt nên clean FP vẫn `42,4%`.
4. **Calibration sơ bộ bằng threshold sweep:** lưu đồng thời kết quả của cả hai
   filter rồi tính lại nhiều cặp ngưỡng mà không chạy lại CEM. Chọn `0,6/300`
   vì giảm clean FP xuống `10,5%` mà vẫn giữ poison detection `98%` và exposure
   `2%`. Đây là lượt 4 và là kết quả hiện tại.

## Cách đọc các metric

- `poison_exposure_before_final_k`: tỷ lệ query có tài liệu poisoned trong top-k cuối trước defense. Đây là số phù hợp để so với các lượt cũ.
- `poison_exposure_before`: tỷ lệ query có tài liệu poisoned trong candidate pool. Ở lượt 3, candidate pool là top-10 nên số này cao hơn top-3.
- `poison_exposure_after`: tỷ lệ query còn tài liệu poisoned trong context cuối sau defense.
- `target_asr_before/after`: tỷ lệ câu trả lời bị ép thành `Yes` trên cùng tập query đủ điều kiện. Đây là metric hành vi của model nhỏ trong notebook, không phải độ chính xác RAG tổng quát.
- `clean_false_positive_rate`: tỷ lệ tài liệu sạch bị detector đánh dấu độc trong các tài liệu được kiểm tra. Mẫu số phải được ghi rõ là top-3 hay top-10.
- `clean_empty_context_rate`: tỷ lệ query không còn tài liệu sạch sau defense.
- `Recall/nDCG/MRR`: chất lượng tài liệu liên quan sau defense; càng giảm nhiều thì defense càng làm hỏng utility.

## Khuyến nghị cho lượt tiếp theo

1. Chạy ablation riêng cho filter, refill, spotlighting và cấu hình kết hợp để biết thành phần nào tạo ra cải thiện.
2. Calibration threshold trên tập clean riêng; mục tiêu thực tế ban đầu là đưa clean false-positive rate xuống khoảng 5–10%.
3. Thử span-level removal: loại đoạn instruction đáng ngờ thay vì loại toàn bộ tài liệu.
4. Giữ candidate top-10/final top-3 để giảm context rỗng, nhưng đánh giá thêm top-20 nếu chi phí cho phép.
5. Thử adaptive CEM biết defense để kiểm tra khả năng bypass.
6. Không chọn threshold dựa trên kết quả test cuối rồi báo đó là hiệu quả tổng quát.

## Template thêm một lượt chạy

Mỗi lần chạy mới nên thêm một dòng vào bảng với các trường sau:

```text
Ngày chạy:
Commit/notebook version:
Số query:
Corpus:
Candidate K / Final K:
Threshold semantic / perplexity:
Refill:
Prompt spotlighting:
Poison exposure before final K:
Poison exposure after:
Target ASR before / after:
Clean FP rate và mẫu số:
Clean empty context rate:
Poisoned empty context rate:
Recall@K / nDCG@K / MRR@K trước và sau:
Thời gian filter:
File ZIP hoặc summary.json:
Nhận xét và giới hạn:
```

## Giới hạn hiện tại

Các kết quả trên cùng dùng 100 query đầu tiên và cùng kiểu CEM, chưa phải held-out evaluation. Payload poisoned hiện tại được tối ưu trước khi biết defense; chưa phải attacker thích nghi. Lượt mới bật refill và prompt spotlighting cùng lúc nên chưa thể tách riêng đóng góp của hai thay đổi này. Các kết quả cũng chưa chứng minh câu trả lời của LLM đúng trong hệ thống production.
