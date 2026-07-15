# Kế hoạch xây dựng `01_eda_data_preparation.ipynb`

## 1. Mục tiêu và phạm vi

Notebook này là đầu vào tái lập được cho notebook mô hình SVD, không phải notebook huấn luyện. Nó cần:

- Đọc và kiểm tra hai nguồn thô: `datasets/anime.csv` và `datasets/rating.csv`.
- Ghi nhận chất lượng dữ liệu trước và sau làm sạch, cùng các quyết định làm sạch có lý do nghiệp vụ.
- Tạo dữ liệu đầu vào chuẩn cho collaborative filtering:
  - `data/ratings_clean.csv`: chỉ chứa rating tường minh từ 1 đến 10.
  - `data/anime_clean.csv`: metadata anime đã chuẩn hóa tối thiểu để ghép tên/genre khi xuất gợi ý.
- Tạo ba biểu đồ EDA phục vụ report/slide tại `outputs/figures/`.
- In dataset summary để notebook 02, report và slide dùng cùng một bộ số liệu.

Không thực hiện sampling, lọc minimum interactions, train/test split, chuẩn hóa rating, hay huấn luyện model trong notebook này. Các bước đó thuộc `02_svd_model_evaluation.ipynb` để tránh làm thay đổi bản dữ liệu sạch gốc.

## 2. Bối cảnh dữ liệu và nguyên tắc quyết định

| File | Vai trò | Schema kỳ vọng |
|---|---|---|
| `datasets/rating.csv` | Tương tác user-anime | `user_id`, `anime_id`, `rating` |
| `datasets/anime.csv` | Metadata anime | `anime_id`, `name`, `genre`, `type`, `episodes`, `rating`, `members` |

Quy tắc làm sạch cần nêu rõ trong markdown của notebook:

1. Giá trị `rating = -1` trong **rating.csv** nghĩa là user đã xem nhưng không chấm điểm. Đây không phải dislike hay numeric rating; loại khỏi dữ liệu SVD và mọi metric RMSE/MAE/ranking.
2. Rating tường minh hợp lệ là số nguyên trong khoảng 1-10. Giá trị thiếu, không parse được hoặc ngoài khoảng này phải được đếm và loại, đồng thời ghi số dòng ảnh hưởng.
3. Duplicate theo khóa `(user_id, anime_id)` (nếu có) được gộp bằng trung bình `rating`, sau đó xác nhận rating vẫn nằm trong [1, 10]. Lý do: giữ một quan sát duy nhất cho mỗi cặp user-item và không ưu tiên tùy tiện một bản ghi.
4. Phân phối số lượt chấm theo user/anime sẽ long-tail. Đây là cấu trúc tự nhiên của recommender, không phải outlier để xóa; dùng biểu đồ log scale và để notebook 02 xử lý minimum interactions nếu cần cho hiệu năng model.
5. `anime.csv` chỉ chuẩn hóa metadata phục vụ hiển thị. Không bịa genre/name bị thiếu; giữ `NaN` hoặc nhãn hiển thị `Unknown` khi xuất kết quả. Không dùng cột average `anime.rating` làm nhãn huấn luyện.
6. Không áp dụng z-score, IQR clipping hoặc scaling cho user ratings: thang 1-10 là hữu hạn và ý nghĩa ordinal/explicit rating phải được giữ nguyên cho SVD.

## 3. Cấu trúc notebook và cell-by-cell plan

Notebook dùng tiêu đề lớn `# Anime Recommendation System - EDA & Data Preparation` và mỗi code cell có markdown mô tả ngắn ngay trước nó.

### 0. `## Context & Outputs`

- Nêu mục tiêu: chuẩn bị explicit ratings cho collaborative filtering và giữ metadata để giải thích gợi ý.
- Liệt kê input/output paths, quy ước `-1`, và giới hạn: phân tích trên raw data có thể tốn RAM vì `rating.csv` lớn.
- Ghi rõ seed `SEED = 42` (dù bước làm sạch hiện tại mang tính deterministic) để thống nhất toàn dự án.

### 1. `## Setup`

**Cell 1 - Imports and configuration**

- Import `Path` từ `pathlib`, `pandas as pd`, `numpy as np`, `matplotlib.pyplot as plt`, `seaborn as sns`.
- Thiết lập `SEED = 42`, `RATING_MIN = 1`, `RATING_MAX = 10`.
- Thiết lập paths bằng `Path` từ project root:
  - `RAW_ANIME_PATH = Path("datasets/anime.csv")`
  - `RAW_RATINGS_PATH = Path("datasets/rating.csv")`
  - `DATA_DIR = Path("data")`
  - `FIGURES_DIR = Path("outputs/figures")`
- Tạo `data/` và `outputs/figures/` bằng `mkdir(parents=True, exist_ok=True)`.
- Cấu hình style biểu đồ nhất quán (`seaborn` whitegrid, `figure.dpi`, `tight_layout`) và format pandas cho output ngắn gọn.

**Cell 2 - Helper functions**

Tạo các hàm nhỏ, không có hidden state:

- `quality_snapshot(df, dataset_name, key_columns)`: trả về bảng gồm rows, columns, duplicate rows, duplicate keys, missing values theo cột và dtypes.
- `save_figure(filename)`: áp dụng `tight_layout`, lưu PNG chất lượng phù hợp vào `FIGURES_DIR`, rồi `plt.show()`.
- `display_change_log(change_log)`: hiển thị DataFrame audit gồm `step`, `rows_before`, `rows_after`, `rows_removed_or_merged`, `reason`.

### 2. `## Load & Initial Data Overview`

**Cell 3 - Read raw CSV files**

- Đọc `anime.csv` và `rating.csv` với dtypes rõ ràng khi phù hợp để giảm bộ nhớ:
  - IDs: integer nullable khi cần kiểm tra missing.
  - rating user: numeric, chưa ép nhỏ trước khi validation.
  - text metadata: `string`.
- In dung lượng file và shape của từng DataFrame.
- Fail fast với thông báo dễ hiểu nếu file/required columns không tồn tại.

**Cell 4 - Schema and preview**

- Hiển thị `head()`, danh sách cột, `info()`, `dtypes`, `describe(include="all")` với output giới hạn.
- Khẳng định vai trò biến:
  - ID: `user_id`, `anime_id`.
  - Numeric: user rating, average anime rating, `members`, và `episodes` sau khi kiểm tra parse.
  - Categorical/text: `name`, `genre`, `type`.

**Cell 5 - Initial quality report**

- Chạy `quality_snapshot` cho cả hai bảng.
- In các kiểm tra domain trước cleaning:
  - null IDs và null ratings;
  - phân bố/unique values của raw user rating, đặc biệt số dòng `-1`;
  - số `(user_id, anime_id)` duplicate;
  - số `anime_id` ở ratings không match metadata và số metadata anime không có ratings;
  - missing và chuỗi trống trong `name`, `genre`, `type`, cùng giá trị bất thường của `episodes`.
- Lưu snapshot trước cleaning trong biến `quality_before`; không cần export thêm file trừ khi nhóm muốn audit riêng.

### 3. `## Cleaning Decisions & Data Preparation`

**Cell 6 - Normalize and validate ratings**

- Dùng `pd.to_numeric(..., errors="coerce")` cho `user_id`, `anime_id`, `rating`.
- Loại record thiếu ID hoặc rating, ghi audit riêng.
- Đếm `rating == -1` và loại toàn bộ record này với lý do nghiệp vụ đã nêu.
- Kiểm tra và loại rating ngoài [1, 10] hoặc non-integer (nếu có); dùng assertion cuối cell để bảo đảm còn lại chỉ là 1-10.
- Ép `user_id`, `anime_id` về integer; rating giữ numeric (có thể float sau bước gộp duplicate).

**Cell 7 - Resolve interaction duplicates**

- Đo số duplicate theo `(user_id, anime_id)` sau khi đã loại invalid rows.
- Nếu có duplicate, dùng `groupby(["user_id", "anime_id"], as_index=False)["rating"].mean()`.
- Thêm audit row thể hiện số record được gộp; nếu không có duplicate, vẫn ghi rõ `0` để kết quả tái lập và dễ review.
- Kiểm tra lại uniqueness của khóa và domain rating.

**Cell 8 - Clean metadata for display**

- Xóa row metadata không có `anime_id`; loại duplicate `anime_id` theo nguyên tắc xác định (giữ bản ghi có `name`/metadata đầy đủ hơn, sau đó first stable row), và ghi lý do.
- Chuẩn hóa text có ý nghĩa hiển thị bằng `.str.strip()` cho `name`, `genre`, `type`; chuyển chuỗi rỗng thành `pd.NA`.
- Convert `episodes`, average `rating`, `members` sang numeric bằng `errors="coerce"`; chỉ **báo cáo** giá trị thiếu/không hợp lệ, không impute các số liệu metadata vì không phục vụ trực tiếp model.
- Lưu `anime_clean` với các cột gốc đã xác thực; giữ missing genre/name thay vì tạo thông tin giả.

**Cell 9 - Referential integrity and final validation**

- Xác nhận mọi `anime_id` trong `ratings_clean` có metadata. Nếu có ID không match, báo số lượng và quyết định rõ ràng:
  - mặc định giữ rating cho SVD (metadata không bắt buộc cho train),
  - nhưng ghi nhận những ID này sẽ không có tên/genre khi hiển thị recommendation.
- Tạo `quality_after` và bảng before-vs-after: số rows, unique users, unique anime, density (`ratings / (users * anime)`), missing/duplicates chính.
- Chạy assertions:
  - không còn `-1`;
  - rating thuộc [1, 10];
  - không còn duplicate `(user_id, anime_id)`;
  - IDs không missing;
  - `anime_clean["anime_id"]` unique.
- Hiển thị `change_log` như báo cáo quyết định làm sạch.

**Cell 10 - Export prepared data**

- Xuất theo thứ tự cột ổn định:
  - `ratings_clean`: `user_id`, `anime_id`, `rating`.
  - `anime_clean`: `anime_id`, `name`, `genre`, `type`, `episodes`, `rating`, `members` (có thể đặt tên `anime_average_rating` trong bản làm sạch nếu cần tránh nhầm với user rating; khi đó phải thống nhất với notebook 02).
- Dùng `index=False`, UTF-8, và không ghi index pandas.
- In đường dẫn, số dòng, số cột, và dung lượng file sau export.

### 4. `## Exploratory Data Analysis`

EDA chỉ dùng `ratings_clean` và `anime_clean`; không chạy pairplot/heatmap tổng quát vì dữ liệu tương tác rất lớn và các ID không có ý nghĩa tương quan số học.

**Cell 11 - Dataset summary table**

- Tạo một bảng dùng trực tiếp trong report/slide, gồm:
  - interactions raw;
  - interactions `-1` removed;
  - invalid/missing rows removed;
  - duplicates merged;
  - clean interactions;
  - unique users;
  - unique anime rated;
  - anime metadata count;
  - matrix density (%);
  - rating mean, median, min, max.
- In một câu diễn giải tự động nhưng không suy đoán: dữ liệu thưa nếu density thấp; đưa con số thực tế vào text.

**Cell 12 - Figure 1: Explicit rating distribution**

- Bar chart/countplot cho rating 1-10 của `ratings_clean`.
- Tiêu đề và axes bằng tiếng Anh hoặc Việt Nam nhưng thống nhất toàn notebook; ghi nguồn `rating.csv` và chú thích đã loại `-1`.
- Lưu `outputs/figures/eda_rating_distribution.png`.

**Cell 13 - Figure 2: Top 10 anime by number of explicit ratings**

- Đếm interactions theo `anime_id`, lấy top 10, rồi left join với `anime_clean` để có `name`.
- Dùng barplot ngang, sắp xếp tăng dần để dễ đọc; fallback label `Unknown title (ID: ...)` cho metadata missing.
- Lưu `outputs/figures/eda_top_10_anime_by_rating_count.png`.

**Cell 14 - Figure 3: User activity long-tail**

- Đếm số explicit ratings theo `user_id`.
- Vẽ histogram với trục y log scale (hoặc histogram x log-binned nếu phù hợp), tiêu đề nêu rõ đây là activity distribution.
- Hiển thị median, p90 và p99 để mô tả long-tail; không coi các user nhiều rating là outlier để loại.
- Lưu `outputs/figures/eda_user_rating_count_distribution.png`.

**Cell 15 - Compact EDA insights for handoff**

- In 3-5 bullet được sinh từ số liệu thực thi, không viết sẵn số giả định:
  - tỷ lệ interactions `-1` bị loại;
  - rating mode/mean;
  - anime nhiều rating nhất;
  - median và tail activity user;
  - matrix density và hệ quả: collaborative-filtering problem sparse.
- Thêm “Handoff to Notebook 02”: Notebook 02 đọc hai CSV sạch, có thể sample tối đa 1,000,000 interactions với seed 42 và lọc minimum interactions để phù hợp tài nguyên; phải báo số dòng sau sample riêng.

### 5. `## Reproducibility & Completion Check`

**Cell 16 - Final run checklist**

- Hiển thị checklist tự động với trạng thái pass/fail cho paths output, domain rating, duplicate key, unique IDs metadata, và ba PNG.
- In câu lệnh chạy xác nhận top-to-bottom:

```bash
jupyter nbconvert --execute --to notebook --inplace notebooks/01_eda_data_preparation.ipynb
```

- Khi hoàn tất chạy, điền phần `## tl;dr` ở đầu notebook bằng các số liệu đã quan sát thực tế; không ghi kết luận định lượng trước khi có output.

## 4. Tiêu chí nghiệm thu

- [ ] Notebook chạy độc lập từ project root, theo thứ tự cell, không cần output thủ công từ notebook khác.
- [ ] Hai input paths, schema, missing values, duplicates, range rating và mapping `anime_id` đều được kiểm tra.
- [ ] `-1` được loại có audit rõ ràng; tất cả rating xuất ra nằm trong [1, 10].
- [ ] Không còn duplicate `(user_id, anime_id)` trong `data/ratings_clean.csv`.
- [ ] `data/anime_clean.csv` có `anime_id` duy nhất và không tạo metadata giả cho missing values.
- [ ] Có bảng before-vs-after, change log, dataset summary và ba PNG EDA đúng tên/path.
- [ ] Ba biểu đồ có title, axis labels, thứ tự đọc được và không chứa raw dump quá lớn.
- [ ] Phần handoff nêu rõ toàn bộ decision boundary giữa notebook 01 và notebook 02.

## 5. Mapping với tài liệu tham khảo

- Hướng dẫn nhóm yêu cầu EDA, cleaning/preprocessing và feature/data preparation cho người phụ trách dữ liệu; notebook này hoàn thành đúng phần đó và tạo artifact để thành viên modeling dùng tiếp.
- `plan_simple_svd.md` quy định pipeline hai notebook, output CSV, ba biểu đồ EDA, và quy tắc loại `rating = -1`; kế hoạch này giữ nguyên các quyết định đó.
- Handbook EDA khuyến nghị luồng Import/Overview → Quality/Cleaning → Validation → EDA. Kế hoạch áp dụng luồng này nhưng điều chỉnh theo logic recommender: không dùng IQR/Z-score để xóa activity long-tail hợp lệ, và không impute ratings/metadata khi điều đó có thể làm sai tín hiệu mô hình.
