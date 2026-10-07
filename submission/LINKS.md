# Link nộp bài — Lab 21

**Họ tên:** Võ Công Danh · **MSSV:** 2A202602739

- Repository: https://github.com/lingyan26ai/Day21-Track3-VoCongDanh-2A202602739-Finetuning-Lab
- Báo cáo: [REPORT.md](REPORT.md)
- Reflection: [REFLECTION.md](REFLECTION.md)
- Kết quả gốc: [results/](../results/)
- Mã notebook: [notebooks/](../notebooks/)
- Thư viện: [requirements.txt](../requirements.txt)
- Phiên bản môi trường CPU đã kiểm tra: [requirements-cpu-lock.txt](../requirements-cpu-lock.txt)

## Định dạng

Nộp theo **Option C — code-only** của rubric. Các file JSON và `runs.csv` được đưa vào repo để đối chiếu số liệu. Adapter chính đã train xong và được lưu trên máy cá nhân, cùng bản ZIP đầy đủ của thí nghiệm. Chưa publish adapter lên Hugging Face Hub và chưa nhận điểm thưởng B5.

`requirements-cpu-lock.txt` ghi đúng phiên bản môi trường máy cá nhân đã chạy NB1 và test. Bộ ZIP Colab không chứa bản freeze toàn bộ thư viện GPU: chỉ xác nhận được PEFT 0.21.1 từ metadata của adapter. `requirements.txt` pin phiên bản PEFT này và giữ các giới hạn tương thích của scaffold cho những thư viện GPU còn lại; không coi lock CPU là snapshot môi trường GPU.

## Kết quả

- Target: baseline tối ưu 0.765; fine-tune 0.970.
- Regression: baseline tối ưu 0.7911; fine-tune 0.6111.
- Phán quyết model: FAILED do regression giảm 0.180, vượt ngưỡng 0.020.
- Cổng kiểm tra bài: 26 PASS, 1 WARN, 0 FAIL; test 116 passed, 3 skipped.
- File bằng chứng kiểm tra: [verification.log](../results/verification.log).
