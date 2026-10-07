# Reflection — Lab 21

**Họ tên:** Võ Công Danh · **MSSV:** 2A202602739

**1. Điều gì làm bạn ngạc nhiên nhất?**

Tôi thấy bất ngờ nhất là fine-tune đạt target 97% nhưng vẫn bị đánh giá FAIL. Nhìn riêng điểm phân loại ticket thì kết quả rất tốt, nhưng hỏi một năm có bao nhiêu tháng, model lại trả JSON phân loại thay vì trả lời 12 tháng. Điều này giúp tôi hiểu vì sao phải kiểm tra cả những câu hỏi ngoài tác vụ đã train.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Phần chờ ba run đối chứng ở NB4 làm tôi thấy tốn thời gian nhất. Mỗi run đều phải train thật rồi mới có số liệu để so sánh. Ban đầu tôi nghĩ train xong bản chính là gần hoàn thành, nhưng sau đó còn phải chạy đối chứng, đánh giá và đọc từng ví dụ. Phần này nhiều việc hơn tôi nghĩ lúc đầu.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Trước đây tôi dễ nghĩ rằng loss giảm và điểm tác vụ tăng thì fine-tune đã thành công. Bài này cho thấy model có thể giỏi hơn ở một việc nhưng trả lời kém đi ở việc khác. Tôi cũng thấy `attn_only` có loss thấp hơn `correct` nhưng điểm target lại bằng nhau. Vì vậy, tôi không còn xem loss là căn cứ đủ để chọn model tốt nhất.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI để đọc yêu cầu, hướng dẫn chạy từng notebook, giải thích số liệu và hỗ trợ viết báo cáo. Chỗ chưa tốt là ban đầu AI chỉ liệt kê các bước nên tôi chưa biết thao tác thế nào, phải yêu cầu hướng dẫn cụ thể hơn. Tôi chạy bài trên máy cá nhân và Colab rồi gửi output để AI đối chiếu. Phần so sánh từng mẫu giúp tôi hiểu các ca model làm tốt và các ca trả lời sai yêu cầu.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Tôi sẽ hỏi rõ khách hàng muốn model làm tốt việc gì và lỗi nào họ không chấp nhận được. Sau đó tôi lấy một tập ví dụ thực tế để thử base model với prompt tốt trước. Nếu cách này đã đáp ứng yêu cầu thì chưa cần fine-tune. Nếu phải train, tôi sẽ giữ riêng tập kiểm tra và theo dõi cả tác vụ chính lẫn những khả năng cần giữ lại.
