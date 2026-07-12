# Kế hoạch thực hiện — Anime Recommendation System (Group 14)

## 1. Mục tiêu và phạm vi

Xây dựng hệ gợi ý anime tạo danh sách **Top-N anime cho từng user**, sử dụng dữ liệu Anime Recommendations Database. Bài nộp phải có mã chạy được, README, PPT và báo cáo LaTeX/PDF; đánh giá tối thiểu bằng **RMSE, MAE, Precision@K, Recall@K**.

**Khuyến nghị mô hình chính:** collaborative filtering bằng **Spark MLlib ALS (explicit feedback)**.

Lý do: sau khi loại rating `-1`, dữ liệu vẫn có 6,337,241 ratings của 69,600 user và 9,927 anime, với độ thưa khoảng **99.08%** (density 0.917%). Đây là bài toán user–item sparse, quy mô đủ lớn để Spark ALS vừa phù hợp môn *Machine Learning with Large Datasets*, vừa trực tiếp dự đoán rating để tính RMSE/MAE và sinh Top-N. `genre` nên dùng để tạo **content-based fallback/baseline** cho anime mới hoặc user ít dữ liệu, không nên là mô hình chính nếu chỉ chọn một mô hình.

## 2. Những gì đã đọc và kiểm tra

### Yêu cầu đề bài

- Problem: gợi ý anime theo ratings/preferences trên nền tảng streaming.
- Hướng làm cho phép: collaborative filtering và/hoặc content-based theo genre.
- Bắt buộc có: Top-N per user, RMSE/MAE, Precision@K/Recall@K.
- Member 1 (Trương Nhật Trường): load/clean/preprocess, EDA, feature engineering/data preparation và tài liệu tham khảo APA.
- Member 2 (Mai Nam Quốc): chọn/xây model, tune, đánh giá + biểu đồ, PPT, báo cáo LaTeX và kết luận.

### Database hiện có

| File | Kích thước / grain | Nhận xét |
|---|---:|---|
| `database/anime.csv` | 12,294 anime, 7 cột | Metadata: name, genre, type, episodes, rating trung bình, members. |
| `database/rating.csv` | 7,813,737 dòng user–anime | 73,515 user; mỗi dòng biểu diễn rating/hành vi của một user với một anime. |

Kết quả profile quan trọng:

- `rating = -1`: 1,476,496 dòng (18.90%), nghĩa là đã xem nhưng không đánh giá; không được coi là rating số trong bài toán explicit rating.
- Sau khi lọc `-1`: 6,337,241 ratings hợp lệ, 69,600 user, 9,927 anime, range 1–10, mean 7.8085.
- Có 7 cặp `(user_id, anime_id)` trùng (một user), trong đó phần lớn có rating mâu thuẫn; có 1 dòng trùng hoàn toàn.
- Có 3 `anime_id` ở ratings không xuất hiện trong `anime.csv`: 20261, 30913, 30924.
- `anime.csv` không trùng ID; nhưng thiếu `genre` 62 dòng, `type` 25 dòng, `rating` metadata 230 dòng; 340 giá trị `episodes` không ép được sang số.
- `rating.csv` rất long-tail: median 45 explicit ratings/user và 57 ratings/anime; vì vậy cần xử lý cold-start và đánh giá theo ranking, không chỉ RMSE.

Kết luận: source CSV chưa hoàn toàn “clean”; tuy nhiên các lỗi có quy mô nhỏ (trừ sentinel `-1`, vốn là quy ước dữ liệu) và có quy tắc xử lý rõ ràng.

## 3. Thiết kế mô hình đề xuất

### 3.1 Mô hình chính — Spark ALS explicit

- Input: `user_id`, `anime_id`, `rating` sau khi chỉ giữ rating 1–10.
- Thuật toán: `pyspark.ml.recommendation.ALS` với `implicitPrefs=False`, `coldStartStrategy='drop'`, `nonnegative=True` (thử nghiệm khi tuning).
- Output: dự báo rating và `recommendForAllUsers(K)` / `recommendForUserSubset(K)`.
- Tuning: validation split cố định, thử `rank` = 20/50/100, `regParam` = 0.01/0.05/0.1, `maxIter` = 10/15/20; chọn cấu hình có validation RMSE thấp nhất, sau đó báo cáo test metrics.

### 3.2 Baseline và fallback (nên làm)

1. **Popularity baseline:** anime có số lượt `members`/số rating cao và average rating tốt. Dùng để chứng minh ALS tốt hơn cách gợi ý phổ biến.
2. **Content-based theo genre:** chuẩn hóa genre, multi-hot hoặc TF-IDF; cosine similarity. Khi user có ít explicit rating hoặc ALS không có factor (cold-start), lấy các anime cùng genre với anime user thích (`rating >= 8`), sau đó lọc anime đã xem.
3. **Hybrid serving rule (không bắt buộc phải đánh giá như model chính):** ưu tiên Top-N từ ALS; bổ sung content fallback chỉ khi user/anime không xuất hiện trong training. Điều này tận dụng metadata nhưng giữ metric RMSE/MAE nhất quán cho ALS.

Không khuyến nghị dùng deep learning/Neural CF ở phiên bản đầu: dữ liệu không có timestamp/text/user attributes, chi phí tuning lớn, khó chứng minh cải thiện rõ so với ALS trong thời gian bài tập.

## 4. Quy tắc clean & feature preparation (Member 1)

1. Đọc CSV với schema rõ ràng; kiểm tra số dòng trước/sau từng bước và lưu audit log.
2. Xóa exact duplicates; với cặp `(user_id, anime_id)` có nhiều rating khác nhau, dùng **mean rating theo cặp** (hoặc giữ rating cuối chỉ khi dataset có timestamp; hiện tại không có timestamp nên không được tự suy diễn thứ tự). Ghi lại quyết định trong README/report.
3. Loại `rating == -1` khỏi tập explicit ALS và các metric RMSE/MAE. Có thể lưu riêng làm tín hiệu “watched” cho phân tích sau, không biến thành rating 0.
4. Loại 3 rating có `anime_id` orphan khi cần join metadata/content; ALS có thể chỉ dùng interactions còn anime tồn tại để danh mục gợi ý nhất quán.
5. Chuẩn hóa metadata: genre null → `Unknown`; type null → `Unknown`; `episodes` → numeric với `Unknown`/`N/A` thành null; không dùng `anime.rating` làm label vì đó là aggregate có nguy cơ leakage.
6. Tạo `clean_ratings` và `clean_anime` ở định dạng Parquet (ưu tiên cho Spark); lưu bảng quality summary và data dictionary.
7. Tạo features content: genre tokens/multi-hot; type one-hot; `log1p(members)` và episodes chỉ dùng cho baseline/fallback, không dùng để đánh giá explicit ALS.

## 5. Chia dữ liệu và đánh giá (Member 2, có Member 1 review)

### Explicit rating metrics

- Dùng random split có seed cố định (gợi ý 80/10/10 train/validation/test), sau khi clean.
- Bảo đảm user/item ở validation/test cũng xuất hiện trong train; các interaction cold-start được tách ra để báo cáo coverage, tránh Spark `coldStartStrategy='drop'` làm metric bị hiểu sai.
- Tính RMSE và MAE trên test sau tuning, đồng thời nêu số prediction được đánh giá và coverage.

### Top-N ranking metrics

- Với mỗi user có đủ dữ liệu, giữ lại một phần interaction dương cho test (ví dụ leave-one-out hoặc 80/20 per-user; chọn một protocol và giữ cố định).
- Định nghĩa relevant rõ ràng: `rating >= 8` là positive (phân tích sensitivity `>=7` nếu còn thời gian).
- Sinh Top-K (K = 5, 10; tối thiểu báo cáo @10), lọc anime user đã thấy trong train.
- Tính Precision@K, Recall@K, coverage; nếu dùng candidate sampling để giảm tính toán, mô tả cách sample và seed trong report.
- So sánh ALS với popularity baseline; không so sánh RMSE với content-only vì content-only không dự báo rating theo cùng mục tiêu.

## 6. EDA cần có (Member 1)

- Phân bố explicit ratings và tỷ lệ `-1`.
- Số ratings/user và ratings/anime (log scale) để cho thấy sparsity/long tail.
- Top genres/types theo số anime và interaction; rating trung bình theo type/genre (có minimum-count filter).
- Top anime theo members và theo số explicit ratings.
- Bảng data-quality trước/sau cleaning: rows, users, items, duplicates, missingness, orphans.

## 7. Cấu trúc code/artifact đề xuất

```text
Anime Recommend/
├── database/                        # raw CSV, about_datasets.md
├── data/processed/                  # generated Parquet; không commit nếu quá lớn
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb   # Member 1
│   └── 02_als_model_evaluation.ipynb# Member 2
├── src/
│   ├── data_prep.py
│   ├── train_als.py
│   ├── evaluate.py
│   └── recommend.py
├── outputs/figures/
├── README.md
├── requirements.txt
├── report_template.tex
└── plan_anime_recommender.md
```

`README.md` phải nêu đường dẫn dữ liệu, cách cài PySpark, lệnh chạy end-to-end, seed, hardware/runtime dự kiến, file output và ví dụ Top-N cho vài user.

## 8. Phân công và mốc hoàn thành

| Giai đoạn | Chủ trách nhiệm | Deliverable / tiêu chí xong |
|---|---|---|
| Data audit + cleaning rules | Member 1 | Clean script/notebook, Parquet, audit table, data dictionary. |
| EDA + feature preparation | Member 1 | 4–6 biểu đồ, diễn giải, genre features/fallback candidates. |
| APA references | Member 1 | Dataset, Spark ALS, recommender evaluation references theo APA 7. |
| Baseline + ALS | Member 2 | Script/notebook training có seed và parameter grid. |
| Tuning + evaluation | Member 2 | RMSE, MAE, P@5/P@10, R@5/R@10, coverage + charts. |
| Top-N + cold-start fallback | Member 2 | Hàm/CLI nhận user ID và trả danh sách anime đã lọc watched. |
| Integration & testing | Cả hai | Chạy lại từ raw data, kiểm tra outputs và README. |
| PPT + LaTeX/PDF | Member 2, Member 1 review data/reference | Đủ các slide/section bắt buộc và contribution percentages. |

Gợi ý contribution để dễ minh bạch: 50/50 nếu khối lượng thực hiện đúng như phân công; chỉ điều chỉnh theo phần việc thực tế và dùng cùng tỷ lệ trong report/PPT.

## 9. Tài liệu tham khảo ban đầu cần đưa APA

- CooperUnion. (n.d.). *Anime recommendations database* [Data set]. Kaggle. https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. *Proceedings of the 2008 Eighth IEEE International Conference on Data Mining*, 263–272. https://doi.org/10.1109/ICDM.2008.22  
  *Lưu ý:* tham khảo phương pháp ranking/implicit; model chính của nhóm là ALS explicit.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer, 42*(8), 30–37. https://doi.org/10.1109/MC.2009.263
- Meng, X., et al. (2016). MLLib: Machine learning in Apache Spark. *Journal of Machine Learning Research, 17*(34), 1–7. http://jmlr.org/papers/v17/15-237.html

## 10. Quyết định cần chốt trước khi code model

- Chốt policy duplicate: mean theo `(user_id, anime_id)` (khuyến nghị), vì không có timestamp.
- Chốt threshold relevance `rating >= 8` và K = 5, 10.
- Chốt random seed chung (ví dụ 42) và protocol split; không đổi sau khi bắt đầu tuning.
- Chốt có làm hybrid fallback hay chỉ ALS + popularity baseline. Khuyến nghị làm fallback nhẹ nếu còn thời gian; ALS vẫn là mô hình được trình bày/đánh giá chính.
