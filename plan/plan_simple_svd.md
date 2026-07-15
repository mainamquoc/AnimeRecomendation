# Kế hoạch tối giản — Anime Recommendation System (Group 14)

## 1. Quyết định thực hiện

Nhóm dùng **Matrix Factorization (SVD) cho explicit ratings** làm mô hình chính. Quy trình được tách thành **hai notebook** Python: một notebook EDA/chuẩn bị dữ liệu và một notebook train/đánh giá model. Hai notebook dùng CSV làm input/output; không dùng Spark, không chuyển Parquet, không cần pipeline nhiều module.

Lý do: đề bài chỉ yêu cầu recommender tạo Top-N và báo cáo RMSE, MAE, Precision@K, Recall@K. SVD đáp ứng trực tiếp các yêu cầu này, dễ giải thích trong báo cáo, và phù hợp với rating 1–10. Dataset có 12.294 anime và 7.813.737 tương tác; đây là dữ liệu đủ lớn và thưa cho collaborative filtering, nhưng không bắt buộc phải dùng hạ tầng big-data.

> Lưu ý data: đây là bộ dữ liệu đã khá tốt để dùng trực tiếp. Bước bắt buộc duy nhất là bỏ `rating = -1` khỏi model/metric vì nó có nghĩa là *đã xem nhưng chưa chấm điểm*, không phải rating số hay dislike. Có thể kiểm tra và bỏ duplicate nếu có; không cần làm sạch phức tạp.

## 2. Hai notebook và luồng dữ liệu

Mỗi notebook phải chạy độc lập từ thư mục gốc project. Notebook 01 ghi CSV đã chuẩn bị, notebook 02 chỉ đọc các CSV đó; vì vậy không có hidden state hoặc bước thủ công giữa hai phần.

### Notebook 01 — `notebooks/01_eda_data_preparation.ipynb`

| Phần | Việc làm | Output |
|---|---|---|
| 1. Setup | Import `pandas`, `numpy`, `matplotlib`, `seaborn`; đặt `SEED=42`. | Tham số rõ ràng. |
| 2. Load & check | Đọc `datasets/anime.csv`, `datasets/rating.csv`; in shape, null, duplicate và khoảng rating. | Bảng kiểm tra dữ liệu ngắn. |
| 3. Chuẩn bị | Giữ rating 1–10; bỏ `-1`; xử lý duplicate `(user_id, anime_id)` bằng mean nếu có; giữ metadata anime để hiển thị. | `data/ratings_clean.csv`, `data/anime_clean.csv`. |
| 4. EDA ngắn | Vẽ phân bố rating, Top-10 anime theo số rating, số rating/user (log scale). | 3 PNG tại `outputs/figures/`. |
| 5. Dataset summary | In số dòng, user, anime cuối cùng để notebook 02 và report dùng đúng cùng số liệu. | Bảng tóm tắt/dữ liệu cho slide Dataset. |

### Notebook 02 — `notebooks/02_svd_model_evaluation.ipynb`

| Phần | Việc làm | Output |
|---|---|---|
| 1. Setup | Import `pandas`, `numpy`, `surprise`, `matplotlib`; đặt `SEED=42`, `K=10`, `POSITIVE_RATING=8`. | Môi trường và tham số rõ ràng. |
| 2. Load prepared data | Đọc `data/ratings_clean.csv`, `data/anime_clean.csv`; xác nhận không còn `-1`. | Model input được kiểm tra. |
| 3. Sample (nếu cần) | Lấy sample tái lập được theo giới hạn đã chọn và lọc minimum interactions. | `ratings_model`. |
| 4. Split | Chia train/test 80/20 với seed 42 bằng `surprise.model_selection.train_test_split`. | `trainset`, `testset`. |
| 5. Train SVD | Dùng `surprise.SVD`; cấu hình khởi đầu `n_factors=50`, `n_epochs=20`, `lr_all=0.005`, `reg_all=0.02`, `random_state=42`. | Model SVD. |
| 6. Rating evaluation | Dự đoán trên test; tính RMSE và MAE. | Bảng metric. |
| 7. Top-N evaluation | Nhóm test predictions theo user, xếp theo `est` giảm dần; relevant khi `true_r >= 8`; tính Precision@10 và Recall@10. | Precision@10, Recall@10, số user được đánh giá. |
| 8. Recommend & export | Chọn 5 user minh họa; dự đoán các anime chưa rated, lấy Top-10; ghép metadata và lưu CSV. | `outputs/top_10_recommendations.csv`. |
| 9. Kết luận | Nêu metric, ý nghĩa, hạn chế cold-start và hướng genre fallback. | Nội dung cho Conclusion. |

## 3. Cách giữ notebook chạy được trên máy cá nhân

`rating.csv` khoảng 111 MB và sau khi bỏ `-1` còn hơn 6 triệu rating. Có thể đọc bằng pandas, nhưng SVD trên toàn bộ dữ liệu có thể mất lâu/RAM cao. Cách học tập, công bằng và đơn giản là:

1. Mặc định dùng **sample tái lập được**: sau cleaning, lấy tối đa `1_000_000` ratings với `random_state=42`.
2. Lọc user và anime có ít nhất `5` ratings trong sample (lặp tối đa 2 lần); điều này giảm cold-start nhân tạo và giúp SVD ổn định.
3. In rõ số dòng trước/sau sample trong notebook, report và slide Dataset. Không gọi sample này là toàn bộ dataset.
4. Nếu máy đủ mạnh, đổi `MAX_RATINGS = None` để chạy toàn bộ; phần code và metric không đổi.

Không cần cross-validation hoặc grid search lớn. Nếu còn thời gian, thử đúng **hai** cấu hình SVD (`n_factors=50` và `100`), chọn model có RMSE validation/test thấp hơn; nếu không, giữ cấu hình mặc định và giải thích đây là baseline SVD có seed cố định.

## 4. Quy tắc đánh giá cần ghi rõ

- **RMSE, MAE:** tính từ rating thật 1–10 ở test set và rating dự đoán của SVD.
- **Relevant item:** anime trong test có rating `>= 8`.
- **Precision@10/Recall@10:** tính trên danh sách item test được SVD dự đoán cho từng user (cách đánh giá ranking đơn giản, tái lập được trong một notebook); xếp theo predicted rating giảm dần rồi áp dụng công thức chuẩn.
- **Top-N xuất cho người dùng:** với 5 user minh họa, dự đoán toàn bộ anime chưa có trong train rồi lấy 10 điểm cao nhất. Chỉ làm cho 5 user nên không tốn tài nguyên lớn.
- Khi tạo Top-N, luôn loại anime user đã chấm trong train để tránh đề xuất lại.
- Báo cáo thêm số user thực sự đủ điều kiện tính ranking; không coi các user không có relevant item là Recall bằng 0.

## 5. File và thư viện

```text
datasets/
  anime.csv
  rating.csv
notebooks/
  01_eda_data_preparation.ipynb    # EDA + tạo CSV sạch
  02_svd_model_evaluation.ipynb    # SVD + metrics + Top-N
data/
  ratings_clean.csv                 # sinh bởi notebook 01
  anime_clean.csv                   # sinh bởi notebook 01
outputs/
  top_10_recommendations.csv       # bắt buộc
  figures/                         # các PNG nếu xuất biểu đồ
README.md
requirements.txt
```

Thay `requirements.txt` bằng các thư viện tối thiểu:

```text
pandas
numpy
scikit-surprise
matplotlib
seaborn
jupyterlab
```

README chỉ cần có: cách cài `pip install -r requirements.txt`, cách mở/chạy notebook, đường dẫn input/output, quy tắc rating `-1`, seed, và ví dụ một user được gợi ý.

## 6. Checklist hoàn thành phần code

- [ ] Chạy notebook 01 rồi notebook 02 từ trên xuống, không cần Spark/Java/Hadoop/Parquet.
- [ ] Notebook 01 tạo được `data/ratings_clean.csv`, `data/anime_clean.csv` và ba biểu đồ EDA.
- [ ] Notebook 02 có RMSE, MAE, Precision@10, Recall@10 và Top-10 cho user.
- [ ] Có `outputs/top_10_recommendations.csv` chứa `user_id`, rank, `anime_id`, name, genre, predicted_rating.
- [ ] `README.md` nêu cách cài thư viện, thứ tự chạy hai notebook, input/output, seed và quy tắc loại `rating = -1`.
