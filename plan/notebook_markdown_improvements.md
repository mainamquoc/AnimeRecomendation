# Đề xuất bổ sung Markdown cho notebook EDA & Data Preparation

Áp dụng cho: `notebooks/01_eda_data_preparation.ipynb`

## Mục tiêu

Notebook hiện đã có cấu trúc tốt: `tl;dr` ở đầu, các quyết định làm sạch có thể audit, ba biểu đồ EDA và checklist tái lập. Cần bổ sung phần dẫn chuyện sau các biểu đồ và một phần kết luận cuối notebook để người đọc hiểu:

1. Notebook đang trả lời những câu hỏi nào.
2. Các biểu đồ nói lên điều gì.
3. Những phát hiện này ảnh hưởng thế nào đến Notebook 02 và mô hình SVD.
4. Những giới hạn nào cần được nêu rõ.

Không cần thêm Markdown sau mọi code cell; ưu tiên các cell diễn giải có giá trị cho người đọc.

## Các Markdown cell cần thêm

| Vị trí | Nội dung | Mục tiêu |
|---|---|---|
| Sau `## Context & Outputs` | `## Câu hỏi phân tích` | Nêu rõ notebook kiểm tra điều gì và phạm vi của notebook. |
| Sau Figure 1 | `#### Diễn giải` | Giải thích phân phối rating và selection bias. |
| Sau Figure 2 | `#### Diễn giải` | Giải thích popularity bias. |
| Sau Figure 3 | `#### Diễn giải` | Giải thích user activity long-tail và hệ quả khi lọc dữ liệu. |
| Sau phần `Compact EDA insights` | `## Hàm ý cho Notebook 02` (có thể gộp vào kết luận) | Chuyển phát hiện EDA thành yêu cầu cho bước mô hình hóa. |
| Sau cell checklist cuối cùng | `## Kết luận & bàn giao sang Notebook 02` | Tóm tắt chất lượng dữ liệu, EDA, giới hạn và đầu vào cho notebook sau. |

## Nội dung Markdown đề xuất

### 1. Sau `## Context & Outputs`

```markdown
## Câu hỏi phân tích

Notebook này trả lời bốn câu hỏi:

1. Dữ liệu gốc có những vấn đề chất lượng nào cần xử lý trước khi xây dựng recommender system?
2. Sau khi làm sạch, tập tương tác explicit rating còn bao nhiêu người dùng, anime và lượt đánh giá?
3. Ma trận user–anime có thưa không, và hành vi đánh giá của người dùng có đặc điểm gì?
4. Những đặc điểm nào cần được lưu ý khi lấy dữ liệu làm đầu vào cho Notebook 02 (sampling, filtering, train/test split và SVD)?

**Phạm vi:** Đây là notebook chuẩn bị dữ liệu và EDA. Notebook không đánh giá chất lượng dự đoán, không so sánh mô hình và không diễn giải quan hệ nhân quả.
```

Ngoài ra, bổ sung nguồn, phiên bản và ngày lấy dữ liệu vào phần `Context & Outputs` nếu thông tin này có sẵn. Không tự suy đoán nguồn hoặc ngày dữ liệu.

### 2. Sau Figure 1 — Explicit rating distribution

```markdown
#### Diễn giải

Phân phối điểm explicit rating nghiêng về phía cao: điểm trung bình là **7,81**, mode là **8**, và **82,55%** lượt đánh giá có điểm từ 7 trở lên. Đây là tín hiệu cho thấy dữ liệu phản ánh hành vi tự chọn đánh giá và có xu hướng tích cực; vì vậy, không nên hiểu điểm cao là bằng chứng rằng mọi anime đều có chất lượng tương đương.

Mô hình ở Notebook 02 vẫn dùng trực tiếp thang điểm 1–10, nhưng khi đánh giá cần lưu ý phân phối rating không cân bằng quanh mức trung tính.
```

### 3. Sau Figure 2 — Top 10 anime by rating count

```markdown
#### Diễn giải

Số lượt đánh giá tập trung ở một nhóm anime phổ biến. **Death Note** đứng đầu với **34.226** explicit ratings, tiếp theo là các anime quen thuộc như *Sword Art Online* và *Shingeki no Kyojin*.

Đây là dấu hiệu của popularity bias: mô hình có nhiều tín hiệu hơn cho anime nổi tiếng và có thể kém ổn định hơn với anime ít được đánh giá. Vì vậy, các bước lọc hoặc đánh giá ở Notebook 02 cần báo cáo rõ mức độ bao phủ người dùng và anime sau khi áp dụng điều kiện tối thiểu.
```

### 4. Sau Figure 3 — User activity long-tail

```markdown
#### Diễn giải

Hoạt động đánh giá của người dùng có phân phối đuôi dài: median là **45** ratings, P90 là **230**, và P99 là **640**. Phần lớn người dùng có số tương tác vừa phải, trong khi một nhóm nhỏ đánh giá rất nhiều anime.

Đặc điểm này phù hợp với recommender system nhưng cũng tạo ra chênh lệch lượng thông tin giữa người dùng. Notebook 02 nên báo cáo tác động của sampling và minimum-interaction filtering để tránh chỉ giữ lại nhóm người dùng hoạt động cao.
```

### 5. Cell cuối cùng, sau checklist chạy lại

```markdown
## Kết luận & bàn giao sang Notebook 02

### Kết luận chính

- Từ **7.813.737** tương tác ban đầu, notebook giữ lại **6.337.234** explicit ratings sau khi loại **1.476.496** bản ghi `rating = -1` và gộp **7** tương tác trùng.
- Tập dữ liệu sạch gồm **69.600** người dùng, **9.927** anime đã được đánh giá và **12.294** bản ghi metadata anime.
- Ma trận user–anime có mật độ **0,9172%**, vì vậy đây là bài toán ma trận thưa, phù hợp để tiếp tục với collaborative filtering/SVD.
- Explicit ratings có xu hướng cao (mean **7,81**, mode **8**); hoạt động người dùng và độ phổ biến anime đều có phân phối đuôi dài.
- Metadata được giữ tách biệt với tín hiệu rating để phục vụ hiển thị. Vẫn còn thiếu một số trường như `genre`, `type`, `episodes` và `anime_average_rating`; các giá trị này không được tự suy diễn.

### Hàm ý cho mô hình hóa

Notebook 02 nên đọc trực tiếp `data/ratings_clean.csv` và `data/anime_clean.csv`, sau đó thực hiện sampling, minimum-interaction filtering và train/test split theo cách có thể tái lập. Mọi bước lọc mới cần báo cáo lại số users, anime, interactions và matrix density sau xử lý.

### Giới hạn cần lưu ý

Dữ liệu chỉ chứa các anime đã có explicit rating, nên không đại diện cho toàn bộ hành vi xem anime. Số lượt rating phản ánh mức độ phổ biến/tương tác, không đồng nghĩa trực tiếp với chất lượng anime. Các kết luận trong notebook mô tả dữ liệu, không phải đánh giá hiệu năng của mô hình.
```

## Điều chỉnh tính nhất quán bắt buộc

Trước khi thêm phần kết luận, cần xử lý sự không nhất quán hiện có giữa code, biểu đồ và diễn giải:

- Rating gốc hợp lệ đều là số nguyên từ 1 đến 10.
- Tuy nhiên, sau khi gộp duplicate `(user_id, anime_id)` bằng `mean`, dữ liệu đầu ra có hai rating phân số: `6.5` và `8.5`.
- Vì vậy, không được mô tả `ratings_clean` là chỉ gồm số nguyên. Mô tả chính xác là rating cuối cùng nằm trong khoảng **[1, 10]**; giá trị phân số có thể phát sinh từ phép lấy trung bình khi gộp duplicate.
- Figure 1 hiện chỉ reindex các mức nguyên 1–10 và vì thế không biểu diễn hai rating phân số này. Cần cập nhật code biểu đồ để hiển thị đầy đủ mọi mức rating sau khi gộp, hoặc nêu rõ cách xử lý hai bản ghi này.

Gợi ý chỉnh Markdown trong `Cleaning Decisions & Data Preparation`:

```markdown
Raw explicit ratings are integer values from 1 to 10. After duplicate user–anime interactions are consolidated by their mean, final ratings may be fractional but remain within the valid [1, 10] range.
```

## Ghi chú phong cách

- Giữ code và tên biến bằng tiếng Anh; thống nhất phần thuyết minh hướng người đọc sang tiếng Việt hoặc tiếng Anh, tránh trộn lẫn không chủ đích.
- Giữ `tl;dr` ở đầu notebook; phần kết luận ở cuối không thay thế `tl;dr`, mà tổng hợp phát hiện đã được xác thực và nêu rõ bước tiếp theo.
- Kết luận chỉ dùng số liệu đã xuất hiện trong output được thực thi và không dùng ngôn ngữ nhân quả.
