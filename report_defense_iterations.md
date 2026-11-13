# Báo cáo các lần cải tiến defense cho RAG

## Mục đích

File này ghi lại kết quả theo từng lần thay đổi defense trong notebook [`ipi-def-v2.ipynb`](./ipi-def-v2.ipynb). Mục tiêu là đánh giá đồng thời:

- khả năng giảm tài liệu poisoned lọt vào context;
- khả năng giảm tác động lên câu trả lời của LLM;
- false positive trên tài liệu sạch;
- chất lượng retrieval còn lại sau defense;
- tỷ lệ query bị mất toàn bộ context.

Không được đánh giá một defense chỉ bằng tỷ lệ chặn poison. Một defense chỉ hữu ích khi giữ được chất lượng tài liệu sạch và không làm hệ thống abstain quá thường xuyên.

## PHẦN 1 — Phân tích baseline

### 1. Baseline retrieval và tấn công CEM

Với mỗi query, dense retriever mã hóa query và các tài liệu thành vector rồi xếp
hạng theo cosine similarity. Tài liệu CEM được tạo bằng cách tối ưu một chuỗi
token ngắn ở đầu tài liệu để làm tăng điểm similarity với query. Vì vậy, tài liệu
độc có thể lọt vào top-k dù nội dung thật sự không liên quan.

Baseline dùng để đo hai điều: poison có lọt vào context hay không, và câu trả lời
của RAG có bị ép theo payload hay không. Đây là mốc so sánh trước khi thêm defense;
nó không có bước kiểm tra an toàn sau retrieval.

Trong báo cáo này, baseline thực nghiệm ban đầu là pipeline Dual-Filter chạy trực
tiếp trên top-3, không có candidate pool mở rộng, không refill và chưa có prompt
spotlighting. Lượt đầu chỉ có 10 query để kiểm tra pipeline; lượt 2 mở rộng lên
100 query để làm baseline ổn định hơn.

| Baseline | Query | Poison exposure sau | Clean FP | Clean empty | Poisoned empty | ASR sau |
|---|---:|---:|---:|---:|---:|---:|
| Baseline nhỏ, top-3, không refill | 10 | 0% | 43,3% | 10% | 40% | 0% |
| Baseline mở rộng, top-3, không refill | 100 | 0% | 54,0% | 29% | 38% | 1% |

Baseline cho thấy Dual-Filter có khả năng chặn poison, nhưng phải trả giá bằng
việc loại nhầm nhiều tài liệu sạch và làm context bị rỗng. Vì vậy, chỉ báo
`poison exposure sau = 0%` chưa đủ để kết luận defense hữu ích.

### 2. Dual-Filter Chunk Consistency trong baseline

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

Đây là thành phần bảo vệ đã có trong baseline và được giữ lại ở các lượt sau.
Các cải tiến tiếp theo tập trung vào cách chọn ứng viên, cách bù tài liệu bị loại
và cách điều chỉnh ngưỡng để giảm tác dụng phụ của Dual-Filter.

## PHẦN 2 — Các cải tiến sau baseline

### 3. Candidate top-10 và final top-3

Thay vì chỉ lấy top-3 rồi lọc, notebook lấy **10 ứng viên đầu tiên**, chạy
defense trên cả 10, sau đó chọn 3 tài liệu an toàn để đưa vào RAG. Cách này tạo
ra một vùng dự phòng: nếu một tài liệu trong top-3 bị loại, các tài liệu ở vị trí
4–10 có thể thay thế.

Ý nghĩa: tăng candidate pool không tự làm detector chính xác hơn, nhưng làm hệ
thống ít phụ thuộc vào ba tài liệu đầu tiên và giảm nguy cơ mất toàn bộ context.
Chi phí là phải tính filter cho nhiều tài liệu hơn.

So với baseline top-3, thay đổi này mở rộng số tài liệu được kiểm tra từ 3 lên 10.
Nó tạo thêm lựa chọn để refill, nhưng chưa tự giảm false positive; lượt 3 vẫn có
clean FP `42,4%` vì còn dùng ngưỡng aggressive `0,4/200`.

### 4. Refill tài liệu an toàn

Sau khi loại các ứng viên đáng ngờ, hệ thống duyệt phần còn lại theo thứ hạng và
điền đủ `final_k=3` bằng các tài liệu không bị đánh dấu. Nếu không đủ tài liệu an
toàn, context có thể ngắn hơn 3 tài liệu hoặc rỗng.

Ý nghĩa: refill giải quyết lỗi vận hành của detector quá chặt. Ở lượt 2, lọc
trực tiếp top-3 làm clean empty context `29%` và poisoned empty context `38%`.
Khi kết hợp top-10 với refill, các tỷ lệ này giảm xuống `6%`, rồi còn `0%` ở
cấu hình Balanced. Refill không làm poison an toàn; chỉ dùng các tài liệu đã qua
filter để thay thế tài liệu bị loại.

So với baseline mở rộng 100 query, refill giảm clean empty context từ `29%` xuống
`6%` ở lượt 3 và xuống `0%` ở lượt 4. Đây là cải tiến về tính liên tục của hệ
thống, không phải bằng chứng detector chính xác hơn.

### 5. Prompt spotlighting

Trước khi gửi context vào RAG, notebook thêm chỉ dẫn định dạng để nói rõ phần
context là **dữ liệu tham khảo**, không phải instruction phải tuân theo. Cơ chế này
nhắm vào indirect prompt injection ở tầng RAG, nơi nội dung tài liệu có thể cố
gắng điều khiển model.

Ý nghĩa: spotlighting tạo ranh giới vai trò giữa system/user instruction và văn
bản retrieved. Nó là lớp giảm tác động lên câu trả lời, không phải detector poison;
vì vậy cần đánh giá riêng bằng ablation để biết đóng góp của nó.

So với baseline, spotlighting bổ sung một lớp bảo vệ ở tầng RAG. Trong lượt 3 và
4, ASR giảm từ `9%` trước defense xuống `0%` sau defense, nhưng vì refill,
threshold và spotlighting cùng được bật nên chưa thể quy toàn bộ mức giảm này cho
spotlighting.

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

### So sánh tổng thể baseline và cấu hình hiện tại

| Metric | Baseline 100 query | Sau cải tiến Balanced | Thay đổi |
|---|---:|---:|---:|
| Poison exposure sau | 0% | 2% | tăng 2 điểm %, vẫn rất thấp |
| Clean FP | 54,0% | **10,5%** | giảm 43,5 điểm % |
| Clean empty context | 29% | **0%** | giảm 29 điểm % |
| Poisoned empty context | 38% | **0%** | giảm 38 điểm % |
| Target ASR sau | 1% | **0%** | giảm 1 điểm % |

Nếu so với lượt 3 ngay trước Balanced, clean FP giảm tiếp từ `42,4%` xuống
`10,5%`, trong khi poison detection vẫn đạt `98%`. Đổi lại, poison exposure
không còn bằng 0 mà là `2%`; đây là trade-off có chủ ý để tránh lọc nhầm quá
nhiều tài liệu sạch.

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
| 5 | Cấu hình lượt 4, **tấn công P²A** (văn bản độc trôi chảy) thay cho CEM | 100 | 10/3 | 100% ở top-3 | **79%** | 4% → 1% (*) | 0% | 0% | 10,5% |
| 6 | Lượt 4 + **kiểm tra nhất quán NLI** sau Dual-Filter (τ = 0,2), tấn công P²A | 100 | 10/3 | 100% ở top-3 | **75%** | 4% → 1% (*) | 0% | 0% | 10,5% + 3,9% (**) |
| 7 | **NQ** (toàn corpus 2,68 triệu đoạn), PoisonedRAG 1 đoạn/query, Dual-Filter `0,6/300`, RAG Qwen2.5-3B | 100 | 10/5 | 97% ở top-5 | **82%** | ASR đáp án sai 67% → 59% (***) | 0% | — | **18,0%** |

(*) Target Yes ASR không đo được tác động của P²A trên NFCorpus; xem phần đánh giá lượt 5.

(**) `10,5%` là FP của Dual-Filter; `3,9%` là tỷ lệ tài liệu sạch (đã qua Dual-Filter) bị bước NLI chặn thêm.

(***) Lượt 7 đổi dataset, attacker và LLM nên không so trực tiếp với lượt 1–6. ASR = câu trả lời chứa đáp án sai của PoisonedRAG. Xem phần đánh giá lượt 7 cho đủ 3 biến thể và 6 policy.

### Nguồn kết quả

- Lượt 1: `usenix_results_20261012T040649_851126Z.zip` — 10 query.
- Lượt 2: `usenix_results_20261012T164034_445939Z.zip` — 100 query, không refill.
- Lượt 3: [`result/usenix_results_20261013T073849_806751Z.zip`](./result/usenix_results_20261013T073849_806751Z.zip) — 100 query, candidate top-10, refill và prompt spotlighting.
- Lượt 4: [`result/usenix_results_20261013T085302_453423Z.zip`](./result/usenix_results_20261013T085302_453423Z.zip) — 100 query, cùng pipeline lượt 3 nhưng đổi ngưỡng sang `0,6/300`.
- Lượt 5: [`result/usenix_results_20261108T185216_146791Z.zip`](./result/usenix_results_20261108T185216_146791Z.zip) — chạy lại toàn bộ notebook (commit `f832fd4`). Phần CEM nằm ở `results/dual_filter/`, phần P²A ở `results/p2a/`.
- Lượt 6: [`result/usenix_results_20261109T081526_719962Z.zip`](./result/usenix_results_20261109T081526_719962Z.zip) — chạy lại toàn bộ notebook (commit `ea89556`). Phần mới nằm ở `results/consistency/`.
- Lượt 7: notebook [`ipi-def-v3-nq.ipynb`](./ipi-def-v3-nq.ipynb) đã chạy trên Kaggle; output được Kaggle sync lên GitHub (commit `3fff971`). Số liệu trong báo cáo đọc từ output của notebook. File `nq_results_20261109T115636_973963Z.zip` nằm ở `/kaggle/working`, **chưa tải về `result/`**.

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

## Đánh giá lượt 5: tấn công P²A (văn bản độc trôi chảy)

### Mục đích

Mọi lượt trước chỉ đo Dual-Filter với tài liệu CEM. Tài liệu CEM có một prefix
token gần như vô nghĩa, nên để lại đúng hai dấu vết mà Dual-Filter dựa vào:
perplexity cao và semantic range lớn. Lượt 5 kiểm tra xem defense còn hiệu quả
không khi attacker viết tài liệu độc **trôi chảy, bám sát câu hỏi**.

Cách tạo tài liệu theo P²A trong Yang et al., *Reranker Helps, but Not Enough*
(ICML 2026), mục 5.1: dùng LLM viết một đoạn khoảng 60 từ theo 4 luật — nói thẳng
đáp án ở câu đầu, giọng văn bách khoa, có ngữ cảnh hỗ trợ, chỉ bám vào câu hỏi.
Đây là attack black-box, không cần gradient, **không biết defense**.

### Thiết lập

- Giữ nguyên toàn bộ cấu hình lượt 4: ngưỡng `0,6/300`, candidate top-10, final
  top-3, refill, prompt spotlighting, cùng 100 query NFCorpus. Phần CEM của lần
  chạy này cho lại đúng số của lượt 4 (clean FP `10,5%`, phát hiện `98%`,
  exposure sau `2%`), nên hai attack được so trong cùng điều kiện.
- Model sinh: `Qwen/Qwen2.5-3B-Instruct` (paper dùng GPT-4o), greedy decoding,
  đáp án mục tiêu `Yes` để cùng mục tiêu với payload CEM. Sinh 100 đoạn mất
  khoảng 114 giây trên T4.
- Mỗi query chèn **1 tài liệu**, giống CEM. Hai biến thể:
  - `p2a`: câu hỏi + đoạn văn (cách làm của paper).
  - `p2a_payload`: đoạn văn + **nguyên payload** `IMPORTANT!!!! … 'Yes'` của CEM.
    Biến thể này cho biết Dual-Filter bắt CEM nhờ prefix vô nghĩa hay nhờ payload.
- Ngưỡng dùng lại đúng danh sách sweep ở 11.1; không chỉnh ngưỡng theo kết quả P²A.

### Kết quả ở ngưỡng Balanced `0,6/300`

| Metric | CEM | P²A | P²A + payload |
|---|---:|---:|---:|
| Tài liệu độc trong top-3 trước defense | 56% | **100%** | 95% |
| Tài liệu độc trong top-10 trước defense | 66% | 100% | 100% |
| Tỷ lệ phát hiện | 98% | **21%** | 86% |
| — bắt nhờ semantic range | 6% | 21% | 86% |
| — bắt nhờ perplexity | 98% | **0%** | 0% |
| **Poison exposure sau defense** | **2%** | **79%** | 13% |
| Poisoned empty context | 0% | 0% | 1% |

Clean FP, clean empty context và chất lượng retrieval sạch không đổi so với
lượt 4, vì nhánh sạch giống hệt nhau.

### So sánh các ngưỡng

| Ngưỡng | Clean FP | Exposure sau — CEM | Exposure sau — P²A | Exposure sau — P²A + payload |
|---|---:|---:|---:|---:|
| `0,4/200` | 42,4% | 0% | 37% | 0% |
| **`0,6/300`** | **10,5%** | **2%** | **79%** | **13%** |
| `0,8/500` | 0,2% | 31% | 99% | 65% |

Ngay cả ngưỡng aggressive `0,4/200` — vốn đã bị loại vì loại nhầm 42,4% tài
liệu sạch — vẫn để lọt 37% tài liệu P²A.

### Phân bố tín hiệu

| Nhóm | Semantic range (P10 / trung vị / P90) | Mean perplexity (P10 / trung vị / P90) |
|---|---:|---:|
| Tài liệu sạch (1.000 cặp) | 0,17 / 0,35 / 0,59 | 48 / 76 / 130 |
| CEM | 0,40 / 0,49 / 0,58 | 360 / 486 / 705 |
| P²A | 0,13 / 0,47 / 0,66 | **35 / 56 / 96** |
| P²A + payload | 0,57 / **0,75** / 0,89 | 102 / 117 / 145 |

### Nhận xét

1. **Perplexity filter mất hoàn toàn tác dụng với văn bản trôi chảy.** Perplexity
   trung vị của P²A (56) còn *thấp hơn* tài liệu sạch (76), và giá trị cao nhất
   chỉ 167 — xa ngưỡng 300. Không có ngưỡng perplexity nào tách được P²A khỏi tài
   liệu sạch. Kết quả này khớp với hai paper: Zhan et al. (NAACL Findings 2025)
   cho thấy perplexity filter bị vượt khi attacker thêm ràng buộc trôi chảy, và
   Yang et al. đo phân bố perplexity của P²A chồng lấn với văn bản sạch.
2. **Kết quả `2%` exposure ở lượt 4 là nhờ prefix vô nghĩa của CEM.** Ở ngưỡng
   Balanced, CEM bị bắt 98% nhờ perplexity, semantic range chỉ đóng góp 6%.
3. **Semantic range chỉ hữu ích khi tài liệu trộn nội dung đúng chủ đề với chỉ thị
   lạc đề.** Khi phần đầu tài liệu trôi chảy và đúng chủ đề, đoạn payload
   `IMPORTANT!!!!…` lệch chủ đề làm range tăng mạnh (trung vị 0,75) nên biến thể
   P²A + payload bị bắt 86%. Khi toàn bộ tài liệu bám chủ đề như P²A, range chồng
   lấn với tài liệu sạch (0,47 so với 0,35).
4. **P²A mạnh hơn CEM ngay ở tầng retrieval** (100% so với 56% trong top-3) mà
   không cần tối ưu, vì tài liệu chép nguyên câu hỏi ở đầu và chỉ nói về chủ đề đó.
5. **Chỉnh ngưỡng không giải quyết được vấn đề.** Hai tín hiệu hiện có đều dựa
   vào dấu vết mà văn bản trôi chảy không để lại; cần một tín hiệu khác.

### Vì sao target Yes ASR không dùng được cho P²A

| | CEM | P²A | P²A + payload |
|---|---:|---:|---:|
| Target Yes ASR trước → sau | 9% → 0% | 4% → 1% | 11% → 1% |

ASR của P²A thấp **ngay cả khi 79% tài liệu độc vẫn nằm trong context sau
defense**, nên con số `1%` không phải bằng chứng defense hiệu quả. Nguyên nhân là
cách đo không phù hợp:

- Phần lớn query NFCorpus là từ khoá (`deafness`, `DHA`, `prunes`…), không phải
  câu hỏi có/không, nên mục tiêu `Yes` không có nghĩa với P²A. Model sinh chỉ mở
  đầu đoạn bằng "Yes" ở **1%** số đoạn (luật 1 gần như không được tuân thủ), ví dụ:
  *"Deafness … can indeed be answered affirmatively as 'Yes' when asked whether…"*.
- FLAN-T5-small trả lời bằng cụm từ ngắn và hiếm khi trả lời đúng một chữ `Yes`.

Với P²A, **poison exposure** là kết quả chính. Không báo cáo "ASR 1% sau defense"
như một kết quả bảo vệ.

### Kết luận lượt 5

Dual-Filter ở cấu hình Balanced chặn được CEM (exposure `2%`) nhưng **không chặn
được tài liệu độc viết trôi chảy** (exposure `79%`). Đây chính là rủi ro đã ghi ở
mục "Giới hạn hiện tại": payload được tạo trước khi biết defense, và defense dựa
vào dấu vết đặc thù của CEM. Kết quả lượt 4 vẫn đúng với CEM, nhưng không được
khái quát thành "Dual-Filter chống được retrieval poisoning".

Giới hạn của lượt 5: model sinh 3B yếu hơn GPT-4o trong paper; chỉ 100 query của
cùng batch; vẫn là attack không biết defense (adaptive attack có thể còn mạnh hơn);
không đo được tác động lên câu trả lời của LLM vì lý do ở trên.

## Đánh giá lượt 6: claim-adoption ASR và kiểm tra nhất quán NLI

### Mục đích

Lượt 5 để lại hai vấn đề: target Yes ASR không đo được P²A, và Dual-Filter không có
tín hiệu nào cho văn bản trôi chảy. Lượt 6 (notebook mục 11.3) thử cả hai, **không
đổi Dual-Filter**:

- **Claim-adoption ASR:** lấy câu đầu của đoạn P²A làm khẳng định của tài liệu độc;
  dùng NLI (`cross-encoder/nli-deberta-v3-base`) kiểm tra câu trả lời RAG có kéo theo
  khẳng định đó không. Chỉ đếm query mà câu trả lời trên corpus sạch chưa kéo theo
  (96/100 query đủ điều kiện).
- **Kiểm tra nhất quán (ý tưởng TrustRAG, bản rút gọn):** sau Dual-Filter, với mỗi
  candidate còn lại, tính tỷ lệ các candidate khác mâu thuẫn với khẳng định của nó
  (xác suất contradiction > 0,5). Chặn nếu tỷ lệ ≥ τ. τ chọn **chỉ từ nhánh clean**:
  τ nhỏ nhất có FP ≤ 10%.

### Chọn τ

| τ | 0,1 | 0,2 | 0,3 | 0,4 | 0,5 |
|---|---:|---:|---:|---:|---:|
| FP tài liệu sạch (sau Dual-Filter, 895 tài liệu) | 21,0% | **3,9%** | 1,0% | 0,7% | 0,4% |

τ = 0,2 được chọn.

### Kết quả ở τ = 0,2

| Biến thể | Poison lọt qua Dual-Filter | NLI bắt thêm (trong số lọt) | Exposure sau Dual-Filter | Exposure sau cả hai bước |
|---|---:|---:|---:|---:|
| CEM | 2% | 1/2 | 2% | 1% |
| P²A | 79% | 5,1% | 79% | **75%** |
| P²A + payload | 14% | 7,1% | 13% | 12% |

Kể cả τ = 0,1 (FP 21%), exposure P²A vẫn còn `63%`.

Phân bố tỷ lệ bị mâu thuẫn gần như trùng nhau:

| Nhóm | n | Trung bình | Trung vị | p90 | Max |
|---|---:|---:|---:|---:|---:|
| Tài liệu sạch | 895 | 0,032 | 0 | 0,111 | 1,0 |
| P²A (đã qua Dual-Filter) | 79 | 0,029 | 0 | 0,111 | 0,25 |

### Claim-adoption ASR

| Biến thể | Không defense | Dual-Filter | Dual-Filter + NLI |
|---|---:|---:|---:|
| P²A | 10,4% | 27,1% | 26,0% |
| P²A + payload | 8,3% | 3,1% | 3,1% |

Target Yes ASR sau cả hai bước: P²A `4% → 1%`, P²A + payload `11% → 1%`, CEM `9% → 0%`.
Không có query nào bị rỗng context ở nhánh clean.

### Nhận xét

1. **Kiểm tra nhất quán NLI không tách được P²A khỏi tài liệu sạch trên NFCorpus.**
   Các abstract y khoa hầu như được NLI xếp "neutral" với nhau, nên tài liệu độc
   cũng hiếm khi bị mâu thuẫn. Bước mới chỉ giảm exposure P²A từ `79%` xuống `75%`.
2. **Claim ASR "tăng" sau Dual-Filter là do khác prompt, không phải do defense.**
   Nhánh "không defense" dùng prompt gốc; nhánh sau defense bọc tài liệu bằng
   spotlighting `[UNTRUSTED DOCUMENT]`, làm đổi hành vi của FLAN-T5-small. Muốn so
   đúng cần thêm nhánh có spotlighting nhưng không lọc.
3. **Câu khẳng định chất lượng thấp.** Vì ép đáp án mục tiêu `Yes` cho các query
   NFCorpus không phải câu hỏi có/không, nhiều đoạn P²A mở đầu bằng câu nói về
   chính câu hỏi (ví dụ: *"Deafness… can indeed be answered affirmatively as 'Yes'…"*)
   thay vì một mệnh đề y khoa. Claim ASR vì vậy chỉ là chỉ số tham khảo.
4. **FLAN-T5-small hiếm khi bị thuyết phục** ngay cả khi không có defense (claim ASR
   10%), nên tầng generation đang đánh giá thấp mức nguy hiểm của P²A.

### Kết luận lượt 6

Trên NFCorpus, cả perplexity lẫn kiểm tra nhất quán bằng NLI đều không phát hiện
được tài liệu độc viết trôi chảy. Dual-Filter + NLI vẫn để `75%` tài liệu P²A vào
top-3. Bước NLI tốn thêm khoảng 160 giây cho 8.895 cặp trên T4.

Giới hạn: TrustRAG gốc dùng LLM đánh giá xung đột và clustering, mạnh hơn NLI
cross-encoder; khẳng định được rút bằng quy tắc "1–2 câu đầu"; NLI chỉ đọc 512 token.

## Đánh giá lượt 7: PoisonedRAG trên NQ, ablation từng lớp defense

### Mục đích

Lượt 5–6 không kết luận được ở tầng câu trả lời vì mục tiêu `Yes` không hợp với
NFCorpus và FLAN-T5-small quá yếu. Lượt 7 chuyển sang thiết lập của PoisonedRAG/P³A:

- **Dataset:** toàn bộ corpus BEIR NQ (2.681.468 đoạn), 100 query có trong file
  `nq.json` của PoisonedRAG. Mỗi query có đáp án đúng, đáp án sai và 5 đoạn độc do
  GPT-4 viết.
- **Retrieval:** MiniLM, candidate top-10 → final top-5 (top-5 như paper).
- **RAG:** Qwen2.5-3B-Instruct, prompt của PoisonedRAG.
- **ASR:** câu trả lời chứa đáp án sai (so khớp substring, giống paper).
  **Accuracy:** câu trả lời chứa đáp án đúng.
- **3 biến thể tấn công:** `prag1` (1 đoạn, `query ⊕ đoạn`), `prag5` (5 đoạn,
  thiết lập mặc định của paper), `prag5_noq` (5 đoạn, bỏ câu hỏi ở đầu).
- **6 policy trên cùng candidate:** `none`, `spotlight` (chỉ đổi prompt),
  `dualfilter` (ngưỡng `0,6/300` lấy từ NFCorpus, không chỉnh lại), `df_nli`,
  `df_qcopy` (chặn tài liệu chép câu hỏi ở đầu), `df_nli_qcopy`.

Notebook riêng `ipi-def-v3-nq.ipynb`; `ipi-def-v2.ipynb` giữ nguyên cho NFCorpus.

### Retrieval sạch và khả năng lên top của tài liệu độc

- Retrieval sạch: recall@5 = `0,50`; 71% query có đáp án đúng trong top-5 sạch.
- Exposure trước defense: `prag1` 97%; `prag5` 100% (trung bình 4,45/5 tài liệu
  trong top-5 là độc); `prag5_noq` 89%.
- Bỏ câu hỏi ở đầu chỉ làm exposure giảm từ 100% xuống 89%: đoạn GPT-4 bám sát
  chủ đề tự nó đã đủ lên top.

### Tỷ lệ chặn của từng detector (trên candidate top-10)

| Nhóm tài liệu | Số tài liệu | Dual-Filter | NLI (τ = 0,5) | Question-copy |
|---|---:|---:|---:|---:|
| Sạch (FP) | 3.045 | **18,0%** | 8,5% | **0,0%** |
| `prag1` | 100 | 17,0% | 9,0% | **100%** |
| `prag5` | 489 | 16,4% | 2,2% | **100%** |
| `prag5_noq` | 366 | 10,4% | 3,3% | **0,3%** |

- **Dual-Filter không phân biệt được:** chặn tài liệu sạch (18%) nhiều hơn tài
  liệu độc (10–17%). Perplexity trung vị của tài liệu độc (≈111) chỉ cao hơn tài
  liệu sạch (≈85) một chút, phân bố chồng lấn mạnh; semantic range gần như trùng.
  FP trên NQ (18%) cao hơn trên NFCorpus (10,5%) vì ngưỡng không được hiệu chỉnh lại.
- **NLI không phân biệt được:** τ = 0,5 là giá trị nhỏ nhất có FP ≤ 10% (FP ở
  τ = 0,1 / 0,2 / 0,3 lần lượt 52,6% / 28,8% / 20%). Với `prag5`, 5 đoạn độc ủng hộ
  lẫn nhau nên tỷ lệ bị mâu thuẫn còn thấp hơn tài liệu sạch.
- **Question-copy tách hoàn hảo** khi đoạn độc chép câu hỏi ở đầu (100% so với FP
  0%), nhưng **bị né hoàn toàn** chỉ bằng cách bỏ câu hỏi (`prag5_noq`: 0,3%).

### ASR theo policy

| Policy | `prag1` | `prag5` | `prag5_noq` |
|---|---:|---:|---:|
| `none` | 67% | 96% | 82% |
| `spotlight` | 67% | 89% | 75% |
| `dualfilter` | 59% | 90% | 83% |
| `df_nli` | 59% | 90% | 80% |
| `df_qcopy` | **5%** | **3%** | 83% |
| `df_nli_qcopy` | **5%** | **4%** | 80% |

Exposure sau defense:

| Policy | `prag1` | `prag5` | `prag5_noq` |
|---|---:|---:|---:|
| `none` / `spotlight` | 97% | 100% | 89% |
| `dualfilter` | 82% | 100% | 91% |
| `df_nli` | 75% | 100% | 89% |
| `df_qcopy` | 0% | 0% | 91% |
| `df_nli_qcopy` | 0% | 0% | 89% |

Cái giá trên nhánh sạch (không có tài liệu độc):

| Policy | Clean accuracy | Clean ASR (chứa đáp án sai dù không bị tấn công) | Từ chối trả lời |
|---|---:|---:|---:|
| `none` | 56% | 5% | 0% |
| `spotlight` | 57% | 4% | 0% |
| `dualfilter` | 51% | 5% | 0% |
| `df_nli` / `df_qcopy` / `df_nli_qcopy` | 50–51% | 5% | 0% |

### Nhận xét

1. **Với LLM 3B và đáp án sai cụ thể, tấn công rất mạnh:** chỉ 1 đoạn độc đã đạt
   ASR 67%, 5 đoạn đạt 96%, cùng mức với paper (PoisonedRAG-B trên vanilla RAG
   72,6%). Clean ASR chỉ 4–5%, nên ASR đo được phản ánh đúng tác động của tấn công.
2. **Spotlighting giúp một chút, độc lập với bộ lọc:** `prag5` 96% → 89%,
   `prag5_noq` 82% → 75%, không làm giảm clean accuracy. Đây là phần lượt 6 chưa
   tách được.
3. **Dual-Filter gần như vô dụng trên NQ:** ASR `prag5` không đổi (89% → 90%);
   `prag1` giảm 67% → 59% nhưng clean accuracy giảm 57% → 51%. Với `prag5_noq`,
   exposure còn **tăng** (89% → 91%): khi tài liệu sạch bị chặn, refill kéo thêm
   tài liệu độc xếp thấp hơn vào context.
4. **NLI không thêm được gì** trên cả ba biến thể.
5. **Question-copy là tín hiệu duy nhất có hiệu quả,** nhưng chỉ vì PoisonedRAG
   chép nguyên câu hỏi. Đây đúng là trường hợp paper Adaptive Attacks cảnh báo:
   bỏ câu hỏi ở đầu (một thay đổi tầm thường) đưa ASR về 80–83%, trong khi tài liệu
   độc vẫn lên top ở 89% query. Không được báo cáo `df_qcopy` như một defense bền vững.

### Kết luận lượt 7

Thiết lập mới đo được ASR có ý nghĩa và xác nhận kết luận của lượt 5–6 trên một
dataset khác: các detector dựa trên độ "lạ" của văn bản (perplexity, semantic range)
và kiểm tra nhất quán bằng NLI đều không chặn được tài liệu độc viết trôi chảy.
Detector duy nhất hiệu quả (question-copy) bị né bằng một thay đổi tầm thường.
Trên NQ, cấu hình defense hiện tại (`dualfilter`) để ASR ở mức 59–90% và làm giảm
clean accuracy khoảng 6 điểm.

Giới hạn: retriever MiniLM yếu hơn Contriever của paper; chưa có reranker; chỉ 100
query; tấn công vẫn là black-box không biết defense; ngưỡng Dual-Filter lấy từ
NFCorpus, không hiệu chỉnh lại cho NQ.

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
   `2%`. Đây là lượt 4 và là cấu hình defense hiện tại.
5. **Kiểm tra với attacker khác (P²A):** giữ nguyên cấu hình lượt 4, thay CEM
   bằng tài liệu độc do LLM viết trôi chảy. Exposure sau defense tăng từ `2%`
   lên `79%`, vì perplexity không còn phát hiện được gì. Lượt 5 không cải tiến
   defense mà cho thấy giới hạn của nó.
6. **Thêm kiểm tra nhất quán NLI sau Dual-Filter:** τ chọn từ nhánh clean
   (`0,2`, FP thêm `3,9%`). Exposure P²A chỉ giảm từ `79%` xuống `75%` vì NLI
   hầu như không thấy mâu thuẫn giữa các abstract NFCorpus.
7. **Chuyển sang NQ và PoisonedRAG (notebook v3):** đáp án sai cụ thể, LLM
   Qwen2.5-3B, ablation 6 policy. ASR không defense 67–96%. Dual-Filter và NLI
   gần như không tác dụng; question-copy đưa ASR về 3–5% nhưng bị né hoàn toàn
   khi bỏ câu hỏi ở đầu (ASR 80–83%).

## Cách đọc các metric

- `poison_exposure_before_final_k`: tỷ lệ query có tài liệu poisoned trong top-k cuối trước defense. Đây là số phù hợp để so với các lượt cũ.
- `poison_exposure_before`: tỷ lệ query có tài liệu poisoned trong candidate pool. Ở lượt 3, candidate pool là top-10 nên số này cao hơn top-3.
- `poison_exposure_after`: tỷ lệ query còn tài liệu poisoned trong context cuối sau defense.
- `target_asr_before/after`: tỷ lệ câu trả lời bị ép thành `Yes` trên cùng tập query đủ điều kiện. Đây là metric hành vi của model nhỏ trong notebook, không phải độ chính xác RAG tổng quát.
- `clean_false_positive_rate`: tỷ lệ tài liệu sạch bị detector đánh dấu độc trong các tài liệu được kiểm tra. Mẫu số phải được ghi rõ là top-3 hay top-10.
- `clean_empty_context_rate`: tỷ lệ query không còn tài liệu sạch sau defense.
- `Recall/nDCG/MRR`: chất lượng tài liệu liên quan sau defense; càng giảm nhiều thì defense càng làm hỏng utility.

## Khuyến nghị cho lượt tiếp theo

Ưu tiên sau lượt 7 (thiết lập NQ đã đo được ASR có ý nghĩa, nên dùng làm benchmark chính):

1. **Thêm reranker cross-encoder** làm một policy (`rerank`, `rerank + dualfilter`).
   Theo Yang et al., reranker chặn mạnh PoisonedRAG nhưng không chặn P²A, nên cần
   đo trên cả `prag5` và `prag5_noq`.
2. **Kiểm tra nhất quán bằng LLM** (gần TrustRAG gốc hơn: LLM đánh giá xung đột,
   clustering để phát hiện nhóm đoạn độc ủng hộ lẫn nhau) thay cho NLI.
3. Hiệu chỉnh lại ngưỡng Dual-Filter trên NQ sạch, hoặc bỏ Dual-Filter khỏi cấu hình
   mặc định, vì trên NQ nó làm giảm clean accuracy mà không giảm ASR.
4. Thêm P²A với đáp án sai (dạng không chép câu hỏi) để có attacker trôi chảy khác GPT-4.
5. Coi `prag5_noq` là kịch bản chính khi đánh giá defense mới; không báo cáo
   question-copy như defense bền vững.

Đã làm sau lượt 6: đổi sang đáp án sai cụ thể và LLM mạnh hơn, thêm nhánh
spotlighting không lọc, thêm question-copy (lượt 7).

Khuyến nghị sau lượt 5 (mục 2 về kiểm tra nhất quán NLI đã thử ở lượt 6, không hiệu quả trên NFCorpus):

1. **Thêm tín hiệu defense không dựa vào độ "lạ" của văn bản**, vì perplexity và
   semantic range đều không tách được P²A khỏi tài liệu sạch:
   - Cross-encoder reranker giữa retrieval và filter (Yang et al. cho thấy nó làm
     giảm mạnh các attack kiểu cũ, nhưng không chặn được P²A — cần đo lại).
   - Kiểm tra nhất quán giữa các tài liệu kiểu TrustRAG: tài liệu nói ngược với các
     tài liệu còn lại thì đáng ngờ. Theo rebuttal của tác giả P³A trên OpenReview (không có trong PDF), TrustRAG giảm P³A nhiều nhất (72,2% → 48,0%).
   - Phát hiện tài liệu chép nguyên câu hỏi ở đầu (dấu hiệu chung của PoisonedRAG
     và P²A).
2. **Sửa cách đo tác động lên câu trả lời:** đặt một đáp án sai cụ thể cho từng
   query và đo câu trả lời có chứa đáp án đó không; hoặc đo trên một dataset QA
   thật (NQ, HotpotQA) với LLM sinh mạnh hơn FLAN-T5-small.
3. Báo cáo mọi defense mới trên **cả CEM và P²A**, không chỉ CEM.

Các khuyến nghị trước đó vẫn còn hiệu lực:

4. Chạy ablation riêng cho filter, refill, spotlighting và cấu hình kết hợp để biết thành phần nào tạo ra cải thiện.
5. Calibration threshold trên tập clean riêng; mục tiêu thực tế ban đầu là đưa clean false-positive rate xuống khoảng 5–10%.
6. Thử span-level removal: loại đoạn instruction đáng ngờ thay vì loại toàn bộ tài liệu. Kết quả biến thể P²A + payload (semantic range bắt 86%) cho thấy hướng này đáng thử khi tài liệu trộn nội dung đúng chủ đề với chỉ thị.
7. Giữ candidate top-10/final top-3 để giảm context rỗng, nhưng đánh giá thêm top-20 nếu chi phí cho phép.
8. Thử adaptive CEM biết defense (thêm hạng phạt perplexity và semantic range vào hàm mục tiêu) để kiểm tra khả năng bypass.
9. Không chọn threshold dựa trên kết quả test cuối rồi báo đó là hiệu quả tổng quát.

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

Các kết quả trên cùng dùng 100 query đầu tiên, chưa phải held-out evaluation. Payload poisoned được tạo trước khi biết defense; chưa phải attacker thích nghi. Lượt 5 đã xác nhận rủi ro này: chỉ cần thay CEM bằng văn bản độc trôi chảy (vẫn không biết defense), exposure sau defense tăng từ `2%` lên `79%`. Hiệu quả của Dual-Filter chỉ được chứng minh với CEM. Lượt 6 cho thấy thêm kiểm tra nhất quán bằng NLI cũng không khắc phục được (exposure P²A `75%`). Lượt 7 xác nhận điều này trên NQ với PoisonedRAG: Dual-Filter và NLI gần như không giảm ASR; question-copy giảm mạnh nhưng bị né khi bỏ câu hỏi ở đầu. Lượt mới bật refill và prompt spotlighting cùng lúc nên chưa thể tách riêng đóng góp của hai thay đổi này. Các kết quả cũng chưa chứng minh câu trả lời của LLM đúng trong hệ thống production.
