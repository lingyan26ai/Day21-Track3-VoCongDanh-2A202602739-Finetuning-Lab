# Lab 21 — Fine-tuning ticket CSKH tiếng Việt

**Họ tên:** Võ Công Danh

**MSSV:** 2A202602739

**Ngày lập báo cáo:** 07/10/2026

**Tier:** T4 · **Base model:** `unsloth/Qwen3.5-4B` · **GPU:** Colab T4 · **Precision:** fp16

> Kết quả đã đối chiếu với file gốc từ Colab. Cổng kiểm tra đạt 26 PASS, 1 WARN, 0 FAIL; test có 116 passed, 3 skipped. Cảnh báo là phán quyết model FAILED do điểm regression giảm, được phân tích trong báo cáo.

## 1. Setup

Thí nghiệm dùng model và corpus mặc định để tập trung kiểm tra mask, đối chứng và đánh giá, tránh thay đổi nhiều yếu tố cùng lúc. Corpus seed tổng hợp gồm ticket CSKH tiếng Việt có nhãn JSON bốn trường `intent`, `urgency`, `product`, `sentiment`, thuận tiện để chấm tự động. Kết quả trên corpus này chưa chứng minh khả năng xử lý ticket thực tế đa dạng hơn.

| Cấu hình | Giá trị |
|---|---|
| Dữ liệu train ban đầu | 250 mẫu |
| Train / validation | 225 / 25, seed 42 |
| Eval target / regression | 50 / 15; không giới hạn `EVAL_LIMIT` |
| `MASK_MODE` | `assistant-only` |
| `max_length` | 1024; p95 đo được 98, helper gợi ý 256 |
| Epochs / optimizer steps | 2 / 30 cho cả bốn run |
| Batch / gradient accumulation | 1 / 16; batch hiệu dụng 16 |
| LoRA chính | text-linear, r=16, alpha=32, LR=1e-4, base 16-bit |

NB1 đo p95=98, p99=100, max=101 token. Vì vậy 256 đã đủ cho corpus hiện tại. Lượt chạy vẫn giữ 1024 theo tier T4 và dùng thống nhất cho cả bốn run; đây không phải độ dài tối ưu suy ra từ p95. Nếu làm lại để tối ưu tài nguyên, nên áp dụng 256 cho toàn bộ các run, thay vì sửa riêng một đối chứng. Các mẫu đã đo không bị cắt bởi giới hạn 1024.

Template giữ được nội dung `<think>` trong mẫu kiểm tra NB1. Dataset train chỉ có đáp án JSON, không chứa trace suy luận thực; kiểm tra template này không thay thế thí nghiệm về reasoning trace.

## 2. Mask proof

| Kiểm tra trên mẫu NB1 | Kết quả |
|---|---:|
| Token được tính loss / tổng token | 39 / 94 |
| `supervised_fraction` | 0.4149 |
| `answer_is_supervised` | true |
| `question_is_masked` | true |

Phần được tính loss, trích từ `results/mask_proof.json`:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Câu trả lời nằm trong loss, còn ticket được che. Chế độ `everything` ở NB1 tính loss trên 94/94 token, gồm cả prompt; đó là minh họa lỗi và không được dùng train `correct`. Kiểm tra môi trường trên máy cá nhân có 116 test passed, 3 skipped; smoke không thay thế kiểm tra toàn bài trước nộp.

## 3. Ba baseline và mốc đóng băng

NB2 đo (a) và (b) trước train; (c) được đo ở NB5. Cả ba dùng cùng base model và tập eval. NB2 ghi `n_target=50`, `n_regression=15`, `eval_limit=null`, `smoke_mode=false`. SHA rút gọn của prompt tối ưu là `719e74d3b6232053`, khớp prompt trong checkout hiện tại. Lượt sinh bổ sung giữ nguyên prompt và cho trung bình điểm khớp NB2/NB5; nó không thay thế baseline gốc.

| Run | Target | Regression | Format | Latency (ms/mẫu) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 3186.9 |
| (b) base + optimized prompt | 0.7650 | 0.7911 | 1.0000 | 991.4 |
| (c) LoRA `correct` + naive prompt | 0.9700 | 0.6111 | 1.0000 | 1473.0 |

(b) tốt hơn (a) rõ rệt ở target và format. Không cần thay prompt để tạo baseline mạnh hơn trong lượt này. Fine-tune dùng lại prompt ngắn, không dùng prompt tối ưu dài của (b).

Target là độ chính xác trung bình **từng trường**, không phải tỷ lệ ticket đúng toàn bộ. (b) đúng 153/200 trường và đúng toàn bộ 13/50 ticket; FT đúng 194/200 trường và đúng toàn bộ 44/50 ticket. Regression là keyword recall theo bộ từ khóa của lab, không phải đánh giá đầy đủ mọi khía cạnh đúng/sai.

## 4. Giải phẫu cấu hình sai

| Run | Vị trí | r | Trainable params | LR | Train loss | Target | Train (s) | VRAM đỉnh (GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6261 | 0.9700 | 404.2 | 8.78 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | 0.5372 | 0.9700 | 305.1 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | 0.0000 | 442.0 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.9400 | 515.2 | 3.86 |

Cả bốn run có 30 step. `attn_only` đổi vị trí; rank và alpha được tính lại để khớp tham số và giữ alpha/r=2. Sai lệch tham số là 8,192/32,464,896 ≈ 0.0252%, dưới 5%. `wrong_lr` chỉ đổi LR; `qlora` đổi base thành 4-bit và được chấm trên base 4-bit tương ứng.

### 4.1. Vị trí và rank

`attn_only` hòa `correct` ở target 0.97 dù loss thấp hơn, 0.5372 so với 0.6261. Thứ tự loss là `attn_only`, `correct`, `qlora`, `wrong_lr`; thứ tự target là `attn_only = correct > qlora > wrong_lr`. Vì ngân sách tham số đã khớp, không thể giải thích kết quả chỉ bằng việc một run có nhiều tham số hơn. Trên tác vụ này, q,v với rank lớn đạt điểm tương đương text-linear với rank nhỏ. Chưa có quét rank cố định vị trí, nên không thể kết luận tăng rank luôn hữu ích hoặc vị trí không quan trọng với mọi tác vụ.

### 4.2. Learning rate

Giảm LR từ 1e-4 xuống 1e-5 ở cùng 30 step làm loss tăng từ 0.6261 lên 1.5702, target và format giảm về 0. Không có log loss từng step trong bộ artefact đã nhận, nên không mô tả đường loss là phẳng hoặc đoán step bắt đầu thay đổi. Kết quả phù hợp với giả thuyết LR nhỏ chưa giúp adapter học đủ trong ngân sách này. Nếu chỉ nhìn loss cao mà bỏ qua LR và số step, có thể kết luận vội rằng LoRA không phù hợp. Đây là kết quả của hai LR đã đo, không phải quy luật cho mọi LR nhỏ.

### 4.3. QLoRA

VRAM đỉnh giảm 4.92 GB, từ 8.78 xuống 3.86 GB, tương đương khoảng 56.0%. Đổi lại target giảm 3 điểm phần trăm, thời gian train tăng từ 404.2 lên 515.2 giây, latency tăng từ 1473.0 lên 1847.6 ms/mẫu. Format giữ 1.0. LoRA 16-bit vừa T4 và có target cao hơn trong lượt chạy này, phù hợp lựa chọn mặc định của lab cho cấu hình đã đo. Nếu bộ nhớ là ràng buộc chính, mức giảm VRAM của QLoRA vẫn có giá trị. Đây là đỉnh bộ nhớ do harness ghi, không phải toàn bộ bộ nhớ GPU được hệ thống sử dụng; chưa có nhiều lượt lặp để khái quát kết quả.

## 5. Phán quyết

**FAILED** · `target Δ=+0.205` · `regression Δ=-0.180` · `valid_trace_rate=0.0`.

Fine-tune cải thiện tác vụ mục tiêu từ 76.5% lên 97% và giữ format 100%. Tuy nhiên, cổng yêu cầu đồng thời giữ mức giảm regression trong 0.02. Kết quả giảm 0.18, từ 0.7911 xuống 0.6111, vượt ngưỡng cho phép. Các ví dụ cho thấy model đôi khi biến câu hỏi phổ thông thành JSON phân loại và bỏ qua câu trả lời cần đưa ra. Điều này phù hợp với hành vi chuyên biệt lan sang yêu cầu ngoài miền. Chưa đủ bằng chứng khẳng định kiến thức đã bị xóa: một số câu vẫn được trả lời đúng bên trong JSON, và keyword recall có hạn chế. Với điều kiện đánh giá này, adapter `correct` chưa đạt cổng triển khai. Nếu thử khắc phục bằng dữ liệu phổ thông hoặc nhiều dạng trả lời hơn, cần ghi một run mới, đánh giá lại và giữ nguyên kết quả lượt hiện tại.

`valid_trace_rate=0` không đủ chứng minh reasoning-trace collapse. Hàm sinh mặc định `enable_thinking=False`, corpus train không chứa trace, và chưa có đối chứng hai mask mode trên dữ liệu có trace. Chưa thực hiện điểm thưởng B3.

## 6. Định tính — có cả ca thắng và ca thua

Nguồn: `results/qualitative_comparison.json`. Index dưới đây bắt đầu từ **0** trong từng nhóm. Output đầy đủ được giữ trong file; bảng chỉ rút gọn phần cần đối chiếu.

| Nhóm | Số mẫu | FT thắng | FT thua | Hòa |
|---|---:|---:|---:|---:|
| Target | 50 | 33 | 0 | 17 |
| Regression | 15 | 1 | 5 | 9 |

Không có ca target FT thua baseline trong file này. Các ca thua được chọn thuộc regression; không gọi một ticket FT sai là ca thua khi baseline cũng sai.

| # | Nhóm / index | Yêu cầu rút gọn | Nhãn / nội dung cần có | (b) prompt | (c) FT | Nhận xét |
|---|---|---|---|---|---|---|
| 1 | target / 0 | Chuột không dây: “Cho tôi trả lại. Gấp. Shop hỗ trợ tốt.” | intent=`doi_tra` | intent=`hoan_tien`; 0.75 | intent=`doi_tra`; 1.00 | FT thắng: phân biệt đổi trả với hoàn tiền. |
| 2 | target / 6 | Balo laptop: “Đổi size. Hỏi cho biết thôi. Lần cuối mua ở đây.” | intent=`doi_tra`, urgency=`thap` | intent=`hoan_tien`, urgency=`cao`; 0.50 | intent=`doi_tra`, urgency=`thap`; 1.00 | FT thắng: sửa cả intent và urgency. |
| 3 | target / 3 | Bình giữ nhiệt: “Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều.” | urgency=`thap` | urgency=`trung_binh`; 0.75 | urgency=`trung_binh`; 0.75 | Hòa nhưng cùng sai urgency. |
| 4 | regression / 9 | Một năm có bao nhiêu tháng? | `12` | “Một năm bình thường có **12 tháng**.”; 1.00 | `{"intent":"hoi_thong_tin","urgency":"thap","product":null,"sentiment":"trung_tinh"}`; 0.00 | FT thua: phân loại thay vì trả lời số tháng. |
| 5 | regression / 13 | Thành phố Hồ Chí Minh trước đây có tên là gì? | `Sài Gòn` | Có “Sài Gòn”; 1.00 | `{"intent":"hoi_thong_tin","confidence":0.95,"urgency":"trung_binh","intent_label":"hoi_thong_tin","urgency_label":"trung_binh"}`; 0.00 | FT thua: thiếu tên cần trả lời. |
| 6 | regression / 3 | Viết một câu chúc mừng sinh nhật bằng tiếng Việt. | Một câu chúc; `sinh nhật` | “Chúc bạn một ngày sinh nhật thật vui vẻ, tràn đầy niềm vui và sức khỏe, cùng với những điều may mắn và hạnh phúc nhất trong năm mới!”; 1.00 | `{"intent":"chuc_mung_sinh_nhat","urgency":"trung_tinh","tone":"trung_tinh","product":null,"sentiment":"tich_cuc"}`; 0.00 | FT thua: nhận dạng ý định nhưng không viết lời chúc. |

Sáu ticket FT chưa đạt 1.0 đều sai urgency; bốn hòa baseline, hai vẫn cao điểm hơn baseline. Điểm FT theo trường là intent=1.00, urgency=0.88, product=1.00, sentiment=1.00. Cụm “Khi nào tiện” xuất hiện trong các ticket FT sai, gợi ý kiểm tra thêm dữ liệu urgency thấp ở một thí nghiệm mới.

Ở regression, ca thua rõ nhất là câu trả lời bị thay bằng JSON phân loại. Tuy vậy, điểm không đo toàn bộ nội dung: ở mẫu quang hợp, FT mô tả về thực vật nhưng chỉ đạt 0.5 vì thiếu từ “cây”, còn baseline đạt 1.0. Mẫu 2 mũ 10 là ca FT thắng từ 0 lên 1 vì có kết quả 1024, còn baseline dừng trước kết quả. Cần đọc output bên cạnh điểm số, không đồng nhất keyword recall với toàn bộ năng lực hay độ đúng thực tế.

## 7. Kết luận và điều học được

### Kết luận

Chưa nên triển khai adapter `correct` theo cổng đánh giá hiện tại. Trên tác vụ ticket, adapter có lợi ích rõ: dùng prompt ngắn vẫn đạt 97% độ chính xác từng trường, cao hơn 76.5% của base với prompt tối ưu, và cả hai tạo được JSON đúng yêu cầu. Tuy vậy, lợi ích đi cùng mức giảm 18 điểm phần trăm ở regression và latency cao hơn khoảng 48.6%. Các câu hỏi về số tháng, tên cũ của thành phố và lời chúc sinh nhật cho thấy model áp dụng hành vi phân loại sang yêu cầu ngoài miền, thay vì thực hiện nội dung được hỏi. Vì vậy cần đánh giá khả năng chung dù điểm target rất cao.

Trong các yếu tố đã thử, LR tạo ảnh hưởng lớn nhất lên target: giảm từ 1e-4 xuống 1e-5 làm run đạt 0 ở ngân sách 30 step. Đổi vị trí sau khi khớp tham số không thay đổi target, còn lượng tử hóa giảm 3 điểm phần trăm và tiết kiệm bộ nhớ. Kết luận này chỉ áp dụng cho cấu hình và corpus đã đo; chưa có quét rank cố định vị trí hoặc lặp nhiều seed. Mask đúng là điều kiện nền tảng đã kiểm chứng, nhưng chưa có run mask sai để định lượng ảnh hưởng. Nếu có thêm thời gian, nên tập trung vào yêu cầu ngoài miền và trường urgency còn sai, rồi đo lại với cùng mốc đã đóng băng. Giữ kết quả FAILED giúp thể hiện đúng đánh đổi của fine-tuning trong lượt này.

### Ba điều tôi học được

1. Tôi không thể chỉ nhìn điểm 97% rồi kết luận model tốt hơn. Khi hỏi những câu ngoài bài phân loại ticket, model có thể trả JSON thay vì trả lời câu hỏi. Tôi cần xem cả điểm regression và câu trả lời thực tế.
2. Loss thấp hơn chưa chắc cho kết quả tốt hơn. Run `attn_only` có loss thấp hơn `correct`, nhưng hai run cùng đạt target 0.97. Nếu chỉ xem loss, tôi sẽ bỏ qua việc chúng đang hòa nhau trên tập kiểm tra.
3. Tôi nên kiểm tra learning rate trước khi nghĩ đến tăng rank. Trong bài này, giảm LR 10 lần làm target về 0, còn đổi vị trí adapter với cùng ngân sách tham số không làm target thay đổi. QLoRA tiết kiệm nhiều VRAM, nhưng cũng có đánh đổi về điểm và thời gian.

**Nếu có thêm 2 giờ — đề xuất chưa thực hiện:** thử trộn một tỷ lệ nhỏ dữ liệu phổ thông có câu trả lời tự nhiên vào train và rà soát ví dụ urgency thấp. Giữ eval nguyên vẹn, lưu run riêng và đo lại bốn nhóm; chưa biết phương án này có đạt PASS hay không.

## Phụ lục — bằng chứng và kiểm tra trước nộp

- NB1: `results/mask_proof.json`, `template_check.json`, `token_stats.json` và split 225/25; đã đồng bộ file gốc từ Colab. Split Colab khớp split đã tạo trên máy cá nhân.
- NB2–NB5: `results/baselines_frozen.json`, `runs.csv`, `verdict.json`, `autopsy.json`, `qualitative.json`; số liệu báo cáo khớp các file gốc đã nhận.
- File bổ sung `qualitative_comparison.json` được giữ nguyên; `qualitative_summary.json` tính lại điểm, số ca thắng/thua/hòa và lưu index của sáu ví dụ.
- `adapters/correct/` đã có trọng số và cấu hình gốc trên máy cá nhân. Output Colab xác nhận cả bốn adapter đã được lưu trong phiên chạy. Repo GitHub nộp theo Option C, kèm code và toàn bộ kết quả; bản ZIP sao lưu theo Option A có adapter chính. Link repo nằm trong `submission/LINKS.md`.
- Sáu notebook trong ZIP khớp mã nguồn hiện tại sau khi chuẩn hóa xuống dòng; không sửa mã notebook khi nhập kết quả.
- Chưa làm NB6, dataset riêng, reasoning-trace contrast, quét rank hoặc publish adapter; chưa nhận các điểm thưởng này.
- Log kiểm tra: `results/verification.log`. Trên Windows, khôi phục LF để checksum dữ liệu khớp bản gốc; nội dung JSON và nhãn giữ nguyên. Test chạy với UTF-8 và thư mục tạm trong workspace để tránh lỗi môi trường.
- Thông tin cá nhân và `submission/REFLECTION.md` đã được điền. Báo cáo có sáu ví dụ định tính, gồm ba ca FT thua ở regression; phần phán quyết và kết luận đạt độ dài rubric yêu cầu.
