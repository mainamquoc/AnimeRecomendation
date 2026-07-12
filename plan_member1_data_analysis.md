# Plan thực hiện — Member 1: Data & Analysis Lead

**Dự án:** Anime Recommendation System — Group 14  
**Phụ trách:** Trương Nhật Trường (Member 1)  
**Mục tiêu bàn giao:** biến hai file raw trong `database/` thành bộ dữ liệu đã kiểm soát chất lượng, có thể dùng trực tiếp để train/evaluate Spark ALS (explicit feedback), cùng metadata/features cho popularity baseline và content-based fallback.

---

## 1. Phạm vi và nguyên tắc dữ liệu

| Hạng mục         | Member 1 thực hiện                                               | Đầu ra cho Member 2                                       |
| ---------------- | ---------------------------------------------------------------- | --------------------------------------------------------- |
| Data loading     | Đọc CSV với schema rõ ràng, kiểm tra kích thước và kiểu dữ liệu  | Bảng raw profile/audit                                    |
| Cleaning         | Loại sentinel không phù hợp, xử lý trùng lặp/orphan/null         | `clean_ratings`, `clean_anime`                            |
| EDA              | Thống kê chất lượng, sparsity, long tail, genre/type             | 4–6 biểu đồ + insight                                     |
| Data preparation | Chuẩn bị interaction cho ALS, danh mục candidate, features genre | Parquet + feature tables + data dictionary                |
| References       | Tổng hợp tài liệu APA 7 cho dataset/phương pháp                  | `references_member1.md` hoặc phần References trong report |

### Nguyên tắc bắt buộc

1. **Model chính là ALS explicit feedback.** Label chỉ là `rating` của người dùng trong khoảng **1–10**.
2. `rating = -1` nghĩa là _đã xem nhưng chưa cho điểm_, **không phải rating 0**. Không đưa dòng này vào train, RMSE, MAE, Precision@K hay Recall@K của ALS explicit.
3. Không dùng cột `anime.rating` (điểm trung bình cộng đồng) làm feature để dự đoán/đánh giá rating user: đây là aggregate target-like information và có nguy cơ leakage. Cột này chỉ được mô tả trong EDA nếu cần.
4. Không có timestamp, vì vậy không tự suy ra “rating mới nhất”. Nếu một cặp `(user_id, anime_id)` có nhiều explicit rating khác nhau, dùng **mean rating theo cặp**.
5. Mọi bước transform phải lưu số dòng trước/sau, lý do loại bỏ và seed (nếu có) để tái lập kết quả.

---

## 2. Kiểm kê dữ liệu hiện có

| File nguồn            | Grain                                 | Vai trò                                                                               |
| --------------------- | ------------------------------------- | ------------------------------------------------------------------------------------- |
| `database/rating.csv` | Một interaction `(user_id, anime_id)` | Nguồn chính cho ALS và ranking evaluation                                             |
| `database/anime.csv`  | Một anime theo `anime_id`             | Tên anime, genre, type, episodes, members; dùng catalogue, EDA và cold-start fallback |

Các điểm cần xử lý đã biết từ profile dự án:

- `rating.csv`: 7,813,737 interactions; có khoảng 18.9% dòng `rating = -1`.
- Sau khi bỏ `-1`, còn khoảng 6.34M explicit ratings trên khoảng 69.6K users và 9.9K anime; dữ liệu rất thưa và long-tail.
- Có duplicate `(user_id, anime_id)` và có 3 `anime_id` trong ratings không tồn tại ở catalogue.
- Metadata có giá trị thiếu ở `genre`, `type`, `rating`; `episodes` có giá trị text như `Unknown` nên không ép kiểu trực tiếp được.

---

## 3. Cấu trúc output đề xuất

```text
data/
├── processed/
│   ├── clean_ratings.parquet              # interaction explicit dùng cho ALS
│   ├── clean_anime.parquet                # catalogue đã chuẩn hóa
│   ├── watched_unrated.parquet            # rating=-1, tách riêng; không train ALS
│   ├── genre_features.parquet             # anime_id, genre tokens / multi-hot representation
│   ├── data_quality_summary.csv
│   ├── data_dictionary.md
│   └── cleaning_decisions.md
│   # Tạo ở giai đoạn handoff, chỉ sau khi Member 2 đã split:
│   ├── popularity_baseline.parquet        # Member 2 tạo chỉ từ train
│   ├── split_quality_summary.csv           # Member 1 chạy validator trên split
│   └── cold_start_excluded.parquet         # row eval không score được + reason
notebooks/
├── 01_data_preparation.ipynb  # hiểu raw data → xác định data contract → clean → kiểm tra → export
└── 02_eda_clean_data.ipynb    # chỉ đọc dữ liệu clean, vẽ EDA và rút insight
outputs/
└── figures/
    ├── rating_distribution.png
    ├── interactions_per_user.png
    ├── interactions_per_anime.png
    ├── genres_and_types.png
    └── data_quality_before_after.png
```

Ưu tiên **Parquet** cho bảng processed vì Spark đọc hiệu quả hơn CSV. Không commit file Parquet quá lớn; thêm đường dẫn này vào `.gitignore` và cung cấp lệnh tái tạo từ raw CSV.

---

## 4. Pipeline thực hiện chi tiết

### Bước 1 — Thiết lập schema và load dữ liệu

- Đọc `rating.csv` bằng schema: `user_id: int`, `anime_id: int`, `rating: float`.
- Đọc `anime.csv` bằng schema: `anime_id: int`, `name: string`, `genre: string`, `type: string`, `episodes: string`, `rating: float`, `members: int`.
- Đặt kiểm tra fail-fast: khóa ID không null; `user_id > 0`, `anime_id > 0`; raw rating chỉ thuộc `{-1, 1..10}`.
- Lưu audit ban đầu: row count, distinct users/items, schema, missing values, min/max rating và số record theo từng giá trị rating.

**Acceptance:** notebook chạy được từ raw CSV; `data_quality_summary.csv` có một hàng `raw`.

### Bước 2 — Phân tách watched-unrated và explicit ratings

- Tạo `watched_unrated` từ `rating == -1`, lưu riêng với cột `user_id`, `anime_id`.
- Tạo `explicit_ratings_raw` từ `rating BETWEEN 1 AND 10`.
- Loại hoặc gắn cờ mọi rating ngoài hai nhóm trên (nếu xuất hiện), và ghi vào audit.

**Quy tắc sử dụng:**

- `watched_unrated`: chỉ dùng nếu sau này cần lọc anime đã xem khi serving; **không** dùng làm negative label.
- `explicit_ratings_raw`: đầu vào duy nhất cho các metric rating/ranking của ALS.

### Bước 3 — Xử lý duplicate interaction

- Đếm exact duplicate rows và số cặp `(user_id, anime_id)` có nhiều hơn một explicit rating.
- Xóa duplicate hoàn toàn giống nhau.
- Với rating mâu thuẫn trong cùng một cặp, group theo `(user_id, anime_id)` và lấy `avg(rating)`; output phải bảo đảm mỗi cặp chỉ còn một dòng.
- Lưu số cặp bị gộp và mô tả quyết định: không có timestamp nên mean là lựa chọn trung lập, tái lập được.

### Bước 4 — Đồng bộ interaction với catalogue

- Kiểm tra anti-join từ explicit ratings sang `anime.csv` theo `anime_id`.
- Loại 3 orphan item khỏi `clean_ratings` để catalogue train và catalogue recommendation nhất quán.
- Lưu số dòng/cặp bị loại trong audit. Không tự tạo metadata giả cho orphan item.

**Kết quả ALS-ready:** `clean_ratings` chỉ gồm `user_id`, `anime_id`, `rating`; rating 1–10; unique `(user_id, anime_id)`; mọi `anime_id` tồn tại trong `clean_anime`.

> `clean_ratings` là **model-input chuẩn**, không phải chỉ là bảng “đã sạch”. Mỗi cột phải không null và có kiểu Spark tương thích: `user_id`/`anime_id` là `IntegerType` hoặc `LongType`, `rating` là `FloatType` hoặc `DoubleType`. ALS không cần one-hot genre, standardization rating hay normalisation ID; genre/type chỉ là dữ liệu cho fallback.

### Bước 5 — Làm sạch catalogue anime

- Giữ `anime_id` duy nhất và xác nhận `name` không rỗng cho các item còn trong interaction.
- Chuẩn hóa text: trim whitespace; thay null/rỗng bằng `Unknown` cho `genre` và `type`.
- Chuyển `episodes` sang numeric (`episodes_num`); các giá trị không parse được (`Unknown`, `N/A`, …) thành `null`, đồng thời tạo `episodes_missing` (0/1).
- Chuẩn hóa `members` sang numeric không âm; tạo `log_members = log1p(members)` khi làm fallback/baseline.
- Giữ `community_rating` (đổi tên từ `anime.rating`) chỉ để mô tả catalogue/EDA; gắn cảnh báo **không sử dụng trong training hoặc ranking evaluation**.
- Giữ/không giữ các anime không có explicit rating tùy catalogue serving, nhưng tách rõ `has_explicit_interaction` để không nhầm chúng là item trainable.

### Bước 6 — Chuẩn bị features cho fallback content-based

- Tách `genre` theo dấu phẩy, trim, chuẩn hóa tên token và loại token rỗng.
- Tạo một trong hai biểu diễn có thể tái lập:
  - `genre_tokens`: `anime_id`, `genres: array<string>` — ưu tiên cho Spark `CountVectorizer`; hoặc
  - multi-hot genre matrix nếu thực hiện bằng pandas/scikit-learn.
- Chuẩn bị `type` dạng categorical để one-hot ở pipeline fallback khi cần.
- Với cold-start user: lấy các anime user đã chấm `>= 8`, tìm anime có cosine similarity genre cao, rồi lọc tất cả anime user đã tương tác (explicit **và**, nếu phục vụ thực tế, watched-unrated).
- Với cold-start item hoặc user không có positive history: fallback về popularity list. Không trộn điểm content và ALS trong metric chính trừ khi nhóm định nghĩa riêng một hybrid protocol.

### 4.1. Execution contract để Codex triển khai không cần tự suy đoán

#### Môi trường và cấu hình

- Implementation mặc định dùng **PySpark DataFrame API** cho load, cleaning, audit và export. Chỉ chuyển dữ liệu aggregate nhỏ sang pandas/seaborn ở notebook EDA; không gọi `toPandas()` trên bảng interaction đầy đủ.
- Phiên bản Python, Java và PySpark phải được ghi trong README/requirements của dự án. Nếu repository chưa chốt phiên bản, Codex phải kiểm tra môi trường hiện có trước, chọn một bộ phiên bản tương thích và ghi lại lựa chọn; không âm thầm đổi framework sang pandas.
- Mọi path và tham số không hard-code rải rác trong notebook. Tạo một cell/config duy nhất (hoặc `src/config.py`) với tối thiểu:

```python
RAW_RATINGS_PATH = "database/rating.csv"
RAW_ANIME_PATH = "database/anime.csv"
PROCESSED_DIR = "data/processed"
FIGURES_DIR = "outputs/figures"
SEED = 42
POSITIVE_THRESHOLD = 8.0
TOP_K_VALUES = [5, 10]
UNKNOWN_TOKEN = "Unknown"
```

- Notebook phải chạy từ **repository root**. Trước khi xử lý, kiểm tra hai file raw tồn tại; tạo output directory nếu thiếu; lỗi phải nêu rõ path nào bị thiếu.
- Spark session phải có app name rõ ràng, timezone cố định (khuyến nghị `UTC`) và log level phù hợp. Việc ghi Parquet dùng `mode("overwrite")` chỉ với đúng các path con đã định nghĩa trong `PROCESSED_DIR`; không xóa cả thư mục cha.

#### Module/hàm nên tách để notebook chỉ orchestration

Nếu dự án cho phép tạo source code, ưu tiên cấu trúc sau để logic có thể unit test và tái sử dụng:

```text
src/
├── config.py
├── schemas.py          # StructType của hai CSV và schema output
├── data_io.py          # load_raw_data(), write_parquet_safe()
├── cleaning.py         # clean_ratings(), clean_anime(), build_genre_features()
├── audit.py            # profile_stage(), validate_*(), append_audit_row()
└── split_validation.py # validate_split(), classify_cold_start()
tests/
├── test_cleaning.py
├── test_data_contracts.py
└── test_split_validation.py
```

Chữ ký/hành vi tối thiểu:

```python
load_raw_data(spark, ratings_path, anime_path) -> tuple[DataFrame, DataFrame]
clean_ratings(raw_ratings, clean_anime_ids) -> tuple[DataFrame, DataFrame, DataFrame]
# returns: clean_ratings, watched_unrated, rejected_or_orphan_rows

clean_anime(raw_anime, explicit_item_ids) -> DataFrame
build_genre_features(clean_anime) -> DataFrame
profile_stage(stage_name, ratings_df, anime_df=None) -> DataFrame
validate_clean_ratings(ratings_df, anime_df) -> dict[str, bool | int | float]
validate_split(train_df, validation_df, test_df) -> tuple[DataFrame, DataFrame]
# returns: split_quality_summary, cold_start_excluded
```

Codex có thể đổi tên hàm theo convention hiện hữu của repository, nhưng phải giữ cùng input/output contract và không nhúng toàn bộ business logic vào các cell notebook.

#### Thứ tự dependency bắt buộc

```text
raw CSV
  → schema validation + raw audit
  → clean catalogue keys/text
  → split watched-unrated / explicit / invalid-rating
  → deduplicate explicit pairs
  → anti-join orphan items
  → clean_ratings + clean_anime
  → genre_features + quality validation
  → atomic export + read-back verification
  → EDA notebook (read-only consumers)
  → Member 2 split
  → split validation + train-only popularity baseline
```

Lưu ý: cần có tập `clean_anime`/danh sách key hợp lệ trước khi loại orphan, nhưng cờ `has_explicit_interaction` chỉ được tính sau khi `clean_ratings` hoàn tất. Tránh circular dependency bằng cách làm sạch catalogue theo hai pha: chuẩn hóa key/text trước, bổ sung cờ thống kê interaction sau.

### 4.2. Quy tắc transform ở mức code

#### Rating/interactions

1. Parse CSV theo schema khai báo. Record malformed hoặc không parse được không được tự biến thành null rồi biến mất; phải được đếm trong nhóm `invalid_schema`/`invalid_id`/`invalid_rating`.
2. Phân loại record bằng điều kiện loại trừ lẫn nhau:
   - `watched_unrated`: `rating == -1`;
   - `explicit_valid`: `1 <= rating <= 10`;
   - `rejected_invalid_rating`: mọi trường hợp còn lại, gồm null/NaN nếu có.
3. Với `watched_unrated`, áp dụng `dropDuplicates(["user_id", "anime_id"])` và anti-join catalogue giống explicit data. Ghi rõ số exact duplicate và orphan riêng; không để cùng một cặp lặp lại trong file serving filter.
4. Với explicit data, tính các audit metric **trước khi aggregate**: số row exact duplicate, số pair duplicate, số pair có `countDistinct(rating) > 1`. Sau đó group theo cặp và lấy `avg(rating)` cast về `double`. Không round mean; range vẫn phải nằm trong `[1, 10]`.
5. Orphan check dùng `left_anti` join trên danh sách `anime_id` distinct của catalogue. Lưu bảng rejected có `reason` nếu dung lượng nhỏ; nếu không lưu chi tiết thì tối thiểu lưu count và danh sách orphan `anime_id` trong `cleaning_decisions.md`.
6. Sort không phải một phần của data contract. Không viết test phụ thuộc thứ tự row trong Parquet; test bằng key/count/set hoặc order rõ ràng khi hiển thị.

#### Catalogue

1. `anime_id`: positive, non-null, unique. Nếu trùng ID:
   - exact duplicate: giữ một row;
   - metadata mâu thuẫn: không tự chọn ngẫu nhiên; ưu tiên row có nhiều trường non-null nhất, tie-break bằng quy tắc deterministic (ví dụ lexical), đồng thời ghi count và policy vào audit.
2. `name`: trim; empty string thành null. Anime có interaction explicit nhưng thiếu name là lỗi quality gate; anime không có interaction có thể dùng `Unknown` nếu nhóm muốn giữ trong serving catalogue và phải ghi quyết định.
3. `genre`: giữ một cột text chuẩn hóa cho hiển thị và một `genres` array cho feature. Token được trim, loại empty, chuẩn hóa khoảng trắng/case một cách deterministic, `array_distinct`, rồi sort để output ổn định. `Unknown` không được trở thành một genre signal khi tính cosine similarity; biểu diễn bằng array rỗng hoặc loại token này khi vectorize.
4. `type`: trim, empty/null thành `Unknown`; không gộp các type khác nhau nếu chưa có mapping được duyệt.
5. `episodes_num`: parse số nguyên dương; `Unknown`, `N/A`, rỗng, zero hoặc âm thành null và `episodes_missing = 1`. Nếu có số thập phân bất thường, reject/flag thay vì truncate âm thầm.
6. `members`: parse integer/long không âm. Invalid thành null và có cờ `members_missing`; chỉ tính `log_members` khi members hợp lệ. Không dùng `members` hoặc `log_members` trong metric ALS chính.
7. `community_rating`: numeric trong `[1, 10]` hoặc null; giá trị ngoài range thành null/flag. Cột này không được xuất vào `clean_ratings` hoặc feature vector của model chính.

### 4.3. Schema output chốt cứng

| Output                     | Schema/nullable đề xuất                                                                                                                                                                                                                                                                                                         | Partition/order                                  | Consumer                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------ |
| `clean_ratings.parquet`    | `user_id: long not null`, `anime_id: long not null`, `rating: double not null`                                                                                                                                                                                                                                                  | không yêu cầu partition; unique pair             | ALS của Member 2         |
| `watched_unrated.parquet`  | `user_id: long not null`, `anime_id: long not null`                                                                                                                                                                                                                                                                             | unique pair                                      | filter serving           |
| `clean_anime.parquet`      | `anime_id: long not null`, `name: string not null`, `genre: string`, `genres: array<string>`, `type: string not null`, `episodes_num: int`, `episodes_missing: int not null`, `members: long`, `members_missing: int not null`, `log_members: double`, `community_rating: double`, `has_explicit_interaction: boolean not null` | unique `anime_id`                                | catalogue/EDA/fallback   |
| `genre_features.parquet`   | `anime_id: long not null`, `genres: array<string> not null`                                                                                                                                                                                                                                                                     | unique `anime_id`; không chứa token rỗng/Unknown | content fallback         |
| `data_quality_summary.csv` | xem schema audit bên dưới                                                                                                                                                                                                                                                                                                       | sort theo `stage`, `metric` khi export           | report + reproducibility |

Không lưu Spark ML `VectorUDT` làm contract duy nhất của `genre_features`, vì vocabulary/index có thể khó đọc và phụ thuộc bước fit. Giữ `genres: array<string>` làm nguồn canonical; Member 2 fit `CountVectorizer` trên train/candidate phù hợp và lưu vocabulary/model nếu cần vector.

### 4.4. Audit contract và reconciliation

`data_quality_summary.csv` nên ở dạng long để dễ bổ sung metric:

| Cột            | Ý nghĩa                                                                                      |
| -------------- | -------------------------------------------------------------------------------------------- |
| `run_id`       | ID deterministic hoặc timestamp của lần chạy; cùng một run dùng cùng ID                      |
| `stage`        | `raw`, `parsed`, `explicit_before_dedup`, `explicit_after_dedup`, `clean`, `export_readback` |
| `entity`       | `ratings`, `watched_unrated`, `anime`, `genre_features`                                      |
| `metric`       | ví dụ `row_count`, `distinct_users`, `null_rating`, `duplicate_pairs`, `orphan_rows`         |
| `value`        | numeric value                                                                                |
| `unit`         | `rows`, `pairs`, `users`, `items`, `percent`                                                 |
| `rule_or_note` | điều kiện/policy tạo metric                                                                  |

Các phương trình reconciliation phải pass và được assert trong code:

```text
raw_rating_rows
= watched_unrated_raw_rows
 + explicit_valid_raw_rows
 + rejected_invalid_id_or_rating_rows

explicit_valid_raw_rows
= rows_removed_as_duplicate_or_aggregated
 + explicit_after_dedup_rows

explicit_after_dedup_rows
= orphan_explicit_rows
 + clean_ratings_rows
```

Vì aggregation nhiều row thành một pair, biến `rows_removed_as_duplicate_or_aggregated` phải được tính bằng chênh lệch row count, không cộng ngây thơ exact-duplicate count với conflicting-pair count. Tương tự, audit phải phân biệt `row_count` và `distinct_pair_count`.

Sau khi export, đọc lại từng Parquet và so sánh schema, row count, distinct key count với DataFrame trước export. Chỉ đánh dấu pipeline thành công sau khi read-back validation pass.

### 4.5. Test requirements và Definition of Done cho code

#### Unit tests bằng synthetic fixture nhỏ

Fixture phải chứa tối thiểu: một rating `-1`, rating biên 1 và 10, rating invalid/null, exact duplicate, conflicting duplicate, orphan anime, null genre/type, `Unknown` episodes, members âm, duplicate anime metadata và genre có whitespace/token lặp.

Các test bắt buộc:

- phân loại `-1` không lọt vào explicit và không biến thành 0;
- duplicate conflicting được aggregate đúng mean, mỗi pair còn một row;
- invalid IDs/ratings và orphan được đếm/loại đúng reason;
- catalogue normalization cho kết quả deterministic;
- genre tokens không rỗng, không lặp và không chứa `Unknown` như signal;
- validator fail khi rating ngoài range, null key, duplicate pair hoặc orphan còn tồn tại;
- split validator phân loại đúng `user_unseen`, `item_unseen`, `both`;
- read/write round-trip giữ schema và count.

#### Integration/smoke test trên raw data

- Chạy pipeline end-to-end trên raw CSV (hoặc sample giới hạn chỉ cho smoke test), tạo đủ output và không có quality gate fail.
- So sánh các con số đã biết ở Mục 2 bằng tolerance hợp lý; nếu lệch, báo rõ do phiên bản dataset hay do transform, không sửa assertion để ép pass.
- Notebook restart-and-run-all thành công từ kernel sạch. Không phụ thuộc biến được tạo thủ công ở cell chạy trước sai thứ tự.
- Ghi command tái tạo và test vào README, ví dụ `pytest -q` và lệnh chạy notebook/script thực tế của repository. Codex phải dùng đúng runner đã tồn tại trong repo; nếu chưa có thì tạo một entry point rõ ràng.

#### Definition of Done

Một bước chỉ được coi là xong khi đồng thời có: code transform, audit trước/sau, assertion quality, output đúng schema, test tương ứng và mô tả quyết định trong tài liệu. Không coi việc notebook “chạy không lỗi” là đủ nếu reconciliation hoặc read-back chưa pass.

### 4.6. Error handling, logging và reproducibility

- Fail pipeline với lỗi dễ hiểu khi thiếu file, schema header sai, ID/rating vi phạm contract vượt ngoài nhóm rejected đã định nghĩa, duplicate catalogue không thể resolve deterministic, hoặc output read-back không khớp.
- Warning nhưng vẫn chạy cho metadata optional bị thiếu (`genre`, `type`, `episodes`, `members`, `community_rating`) nếu policy chuẩn hóa ở trên xử lý được; count warning phải vào audit.
- Không log toàn bộ record/user data. Chỉ hiển thị sample nhỏ, aggregate và tối đa danh sách ID lỗi cần điều tra.
- Mọi sampling phải có seed và ghi sample fraction/limit. Mọi chart aggregate phải có câu lệnh tạo lại từ clean data.
- Ghi dataset fingerprint tối thiểu gồm file size, modified time và (nếu chi phí chấp nhận được) SHA-256 của raw files vào `cleaning_decisions.md` hoặc manifest để phát hiện raw data thay đổi.

### 4.7. Ranh giới trách nhiệm để tránh tạo artifact sai thời điểm

- Member 1 tạo bốn bảng canonical: `clean_ratings`, `clean_anime`, `watched_unrated`, `genre_features`, cùng audit/docs/figures.
- `popularity_baseline.parquet` **không được tạo từ full clean data**. Member 2 tạo nó sau split chỉ từ train; Member 1 chỉ cung cấp helper/contract và review leakage. Vì vậy file này không tính vào “bốn bảng” bàn giao ban đầu.
- `split_quality_summary.csv` và `cold_start_excluded` chỉ có dữ liệu thật sau khi Member 2 cung cấp train/validation/test. Member 1 chuẩn bị hàm/schema/test trước, rồi chạy validation trong bước handoff; không tạo file giả hoặc số liệu placeholder.
- Nếu Codex được giao riêng task Member 1 mà split chưa tồn tại, deliverable hợp lệ là helper + test + template schema, đồng thời ghi trạng thái `pending Member 2 split` trong handoff note.

---

## 5. Chuẩn bị dữ liệu để model train/evaluate đúng

### Hợp đồng dữ liệu bàn giao cho Member 2

| Table             | Cột bắt buộc                                                   | Điều kiện                                              |
| ----------------- | -------------------------------------------------------------- | ------------------------------------------------------ |
| `clean_ratings`   | `user_id`, `anime_id`, `rating`                                | rating 1–10, một dòng/cặp, item có trong catalogue     |
| `clean_anime`     | `anime_id`, `name`, `genre`, `type`, `episodes_num`, `members` | một dòng/anime; null đã chuẩn hóa theo data dictionary |
| `genre_features`  | `anime_id`, `genres` hoặc vector genre                         | chỉ phục vụ content fallback                           |
| `watched_unrated` | `user_id`, `anime_id`                                          | không dùng làm label/metric explicit                   |

### Điều kiện bắt buộc của tập dữ liệu trước khi train ALS

Đây là các điều kiện Member 1 phải kiểm tra và ghi vào handoff note. Nếu một điều kiện không đạt, không được gọi model là “đã train/evaluate đúng”.

| Nhóm                  | Dataset phải đáp ứng                                                                                                         | Cách xử lý nếu không đạt                                                                                                   |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Grain                 | Một dòng duy nhất cho một cặp `(user_id, anime_id)`                                                                          | Xóa exact duplicate, aggregate rating mâu thuẫn bằng mean                                                                  |
| Label                 | `rating` numeric, non-null, chỉ trong `[1, 10]`                                                                              | Bỏ `-1` và mọi giá trị invalid khỏi explicit input; không impute rating                                                    |
| IDs                   | `user_id`, `anime_id` numeric, positive, non-null                                                                            | Loại/gắn cờ record invalid; không mã hóa lại ID nếu không cần                                                              |
| Referential integrity | Mọi item trong interaction có tên/candidate trong catalogue                                                                  | Loại orphan interaction hoặc tách riêng khỏi model-ready table                                                             |
| Train support         | Mỗi user và item xuất hiện trong **train** ít nhất một lần; khuyến nghị ít nhất 2 interactions/user để giữ được test ranking | Đưa interaction không đủ support vào `cold_start_excluded` và báo cáo count/rate; không để Spark tự drop mà không ghi nhận |
| Candidate set         | Item có thể recommend phải thuộc item catalogue và đã có factor trong train                                                  | Sinh candidate từ `train_items`, không từ toàn bộ raw/validation/test                                                      |
| Leakage               | Không dùng `community_rating`, thống kê popularity hoặc genre vector fit từ validation/test để score/evaluate ALS            | Fit mọi thống kê/vectorizer chỉ trên train; metadata tĩnh chỉ dùng để fallback, tách khỏi ALS metric                       |

**Lưu ý về filter support:** không nên lọc “mạnh tay” toàn bộ user/item low-frequency ngay từ `clean_ratings`, vì sẽ làm dữ liệu lệch về active users/popular anime. Hãy giữ `clean_ratings` đầy đủ sau cleaning, rồi tạo báo cáo `cold_start_excluded` và repair split để đánh giá trên phần có thể score. Nếu nhóm muốn lọc (ví dụ user có dưới 2 explicit ratings), phải báo cáo rõ số record/user/item bị loại và chỉ áp dụng nhất quán cho protocol ranking.

### Bảng kiểm trước train và bảng bàn giao sau split

Member 1 chuẩn bị các hàm kiểm tra hoặc notebook cells để Member 2 chạy lại sau khi tạo split:

```text
clean_ratings
  ├── ratings_als_input                 # user_id, anime_id, rating; pass toàn bộ rule model-input
  ├── train / validation / test          # Member 2 tạo bằng seed cố định
  ├── eval_warm_validation / eval_warm_test
  │     # chỉ các row có cả user_id và anime_id tồn tại trong train
  └── cold_start_excluded
        # interaction test/validation không thể ALS score; reason=user_unseen/item_unseen/both
```

Các số liệu phải xuất hiện trong `split_quality_summary.csv`: số rows, distinct users/items, rating mean/std, duplicate count, số user/item chỉ có một interaction, số row warm/cold cho từng split, và coverage `warm_rows / total_eval_rows`. File này là bằng chứng để diễn giải vì sao số record tính RMSE/MAE có thể nhỏ hơn số record test ban đầu.

### Yêu cầu split để tránh metric bị sai

Member 1 chuẩn bị/kiểm tra helper split hoặc review code của Member 2 theo các quy tắc sau:

1. Chỉ split **sau cleaning**, trên `clean_ratings`.
2. Dùng seed cố định toàn nhóm (khuyến nghị `42`) và lưu seed trong config/README.
3. Với RMSE/MAE: random 80/10/10 train/validation/test là chấp nhận được, nhưng validation/test chỉ tính những cặp mà **cả user lẫn item đã xuất hiện trong train**. Các dòng cold-start phải được báo cáo riêng là coverage/excluded count, không bị Spark `coldStartStrategy='drop'` che mất.
4. Trước khi random split, ưu tiên một **repair split**: nếu một user/item chỉ rơi vào validation/test và không còn interaction nào trong train, chuyển tối thiểu một interaction của user/item đó về train. Nếu không repair, phải loại interaction này khỏi metric và ghi vào `cold_start_excluded`. Không được silently drop.
5. Với Top-N: thực hiện split theo user (ví dụ giữ lại 20% positive hoặc leave-one-out), nhưng chỉ hold out positive của user có đủ history để vẫn còn ít nhất một interaction train. Định nghĩa relevant cố định: `rating >= 8`; báo cáo tối thiểu P@10 và R@10, khuyến nghị thêm @5.
6. Recommendation candidate phải là `train_items` và phải loại anime user đã tương tác trong train. Khi demo/serving, nên loại cả `watched_unrated` để tránh gợi ý lại anime đã xem.
7. Popularity baseline phải được tính **chỉ từ train split**, không tính từ toàn bộ dataset/test; dùng count explicit ratings và mean rating có minimum-count/shrinkage để tránh ưu tiên item có quá ít rating.

---

## 6. EDA cần thực hiện và insight cần viết

Tất cả chart có tiêu đề, trục, đơn vị, caption ngắn; dùng một seed nếu có sampling. Với `rating.csv` lớn, có thể aggregate trên Spark trước rồi mới chuyển bảng nhỏ sang pandas/seaborn để vẽ.

| Biểu đồ/bảng                       | Cách tính                                                          | Ý nghĩa cần nêu                                               |
| ---------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------- |
| Distribution of ratings            | Count/phần trăm theo `-1`, 1…10                                    | Chứng minh vì sao `-1` phải tách khỏi explicit label          |
| Quality before/after cleaning      | Rows, users, items, duplicate pairs, orphan rows, missing metadata | Minh bạch transformation và khả năng tái lập                  |
| Interactions per user (log scale)  | Số explicit rating/user, histogram/percentiles                     | Chỉ ra user long-tail và cold-start                           |
| Interactions per anime (log scale) | Số explicit rating/anime, histogram/percentiles                    | Chỉ ra popularity bias/item long-tail                         |
| Genre/type distribution            | Anime count và/or explicit interaction count theo genre/type       | Mô tả catalogue, động cơ dùng genre fallback                  |
| Top anime                          | Top theo explicit rating count và top theo members                 | Phân biệt popularity trong interaction với metadata community |

Insight tối thiểu đưa vào report/PPT:

- Bài toán là user–item matrix sparse nên matrix factorization ALS phù hợp hơn mô hình dense/simple lookup.
- Sentinel `-1` là missing explicit feedback, không phải dislike.
- Long-tail làm RMSE đơn lẻ không đủ; cần Precision@K/Recall@K và coverage.
- Genre hữu ích cho fallback/cold-start, nhưng không thay thế fairness của test protocol ALS.

---

## 7. Quality gates trước khi bàn giao

- [ ] `clean_ratings.rating` nằm trong `[1, 10]`; không còn `-1`.
- [ ] Ba cột ALS (`user_id`, `anime_id`, `rating`) đều non-null, đúng numeric type và ID dương.
- [ ] `clean_ratings` không còn duplicate `(user_id, anime_id)`.
- [ ] Mọi `clean_ratings.anime_id` join được với `clean_anime.anime_id`.
- [ ] Schema, số dòng trước/sau, số duplicate/orphan/null đều có trong `data_quality_summary.csv`.
- [ ] `clean_anime` có 1 dòng/anime; genre/type đã có giá trị hợp lệ (`Unknown` khi thiếu); `episodes_num` đúng numeric/null.
- [ ] `genre_features` join 1:1 với catalogue và không có token rỗng.
- [ ] Không feature/model input nào dùng `community_rating` hay test/full-dataset popularity gây leakage.
- [ ] Sau split, `split_quality_summary.csv` có warm/cold coverage và lý do excluded; mọi row được tính RMSE/MAE đều có user/item trong train.
- [ ] Tập candidate của ranking chỉ chứa `train_items`; holdout Top-N không lấy hết positive history của một user khỏi train.
- [ ] Test một sample user: lấy top candidate, xác nhận filter loại interaction train (và watched-unrated khi serving).
- [ ] Notebook tái chạy thành công từ raw CSV và tạo cùng schema/output.

---

## 8. Deliverables của Member 1

1. `notebooks/01_data_preparation.ipynb`: notebook xử lý chính chạy tuần tự từ load → khám phá dữ liệu raw và ý nghĩa các cột → xác định data contract cho ALS/Top-N → audit → cleaning → kiểm tra quality gate → tạo feature → export Parquet. Không thực hiện EDA/report chart trong notebook này ngoài các bảng kiểm cần thiết để ra quyết định cleaning.
2. `notebooks/02_eda_clean_data.ipynb`: chỉ đọc các bảng Parquet đã clean; tạo 4–6 biểu đồ EDA, captions và insight để đưa vào report/PPT. Notebook này không thay đổi dữ liệu processed.
3. `data/processed/` (hoặc link/nơi lưu ngoài Git): bốn bảng canonical ở Mục 3 (`clean_ratings`, `clean_anime`, `watched_unrated`, `genre_features`).
4. `data_quality_summary.csv`, `data_dictionary.md`, `cleaning_decisions.md`; thêm `split_quality_summary.csv` và `cold_start_excluded.parquet` sau khi nhận split thật từ Member 2.
5. 4–6 ảnh EDA trong `outputs/figures/` để Member 2 đưa vào PPT/report.
6. `references_member1.md`: tài liệu APA 7 về dataset, Spark MLlib/ALS và recommender evaluation.
7. Handoff note cho Member 2: path bảng, schema, số record cuối, seed, policy duplicate, relevance threshold, và các hạn chế dữ liệu.

---

## 9. Kế hoạch thời gian gợi ý

| Buổi | Công việc                                            | Tiêu chí hoàn tất                                               |
| ---- | ---------------------------------------------------- | --------------------------------------------------------------- |
| 1    | Load, schema, raw audit, xác nhận rules              | Có quality profile raw và quyết định cleaning được ghi lại      |
| 2    | Tách `-1`, deduplicate, xử lý orphan, clean metadata | Xuất Parquet và tất cả quality gates dữ liệu pass               |
| 3    | Genre/type features, EDA, captions/insights          | Có 4–6 figure sẵn sàng đưa vào report/PPT                       |
| 4    | Handoff + review split/leakage với Member 2          | Member 2 train được ALS từ `clean_ratings`; README tái lập được |

---

## 10. References APA 7 khởi tạo

- CooperUnion. (n.d.). _Anime recommendations database_ [Data set]. Kaggle. https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. _Computer, 42_(8), 30–37. https://doi.org/10.1109/MC.2009.263
- Meng, X., Bradley, J., Yavuz, B., Sparks, E., Venkataraman, S., Liu, D., Freeman, J., Tsai, D. B., Amde, M., Owen, S., Xin, D., Xin, R., Franklin, M. J., Zadeh, R., Zaharia, M., & Talwalkar, A. (2016). MLLib: Machine learning in Apache Spark. _Journal of Machine Learning Research, 17_(34), 1–7. http://jmlr.org/papers/v17/15-237.html
- Herlocker, J. L., Konstan, J. A., Terveen, L. G., & Riedl, J. T. (2004). Evaluating collaborative filtering recommender systems. _ACM Transactions on Information Systems, 22_(1), 5–53. https://doi.org/10.1145/963770.963772

> Lưu ý: Member 1 chuẩn hóa APA theo thứ tự alphabet trong report/PPT cuối; kiểm tra lại URL và định dạng italic khi dàn trang LaTeX.

---

## 11. Checklist thực thi dành cho Codex

Khi được yêu cầu code theo plan này, Codex thực hiện theo thứ tự sau và không bỏ qua bước xác minh:

1. Đọc `README`, `AGENTS.md` (nếu có), file dependency và toàn bộ plan này; kiểm tra cấu trúc repository và trạng thái các file hiện hữu trước khi sửa.
2. Profile header/schema/sample của hai raw CSV và đối chiếu với Mục 2; không load toàn bộ bằng pandas.
3. Chốt entry point, config, schema và output contract; ghi rõ mọi giả định mới trong `cleaning_decisions.md`.
4. Viết synthetic tests trước hoặc song song với từng transform quan trọng, đặc biệt sentinel, duplicate, orphan và metadata parse.
5. Implement pipeline theo dependency ở Mục 4.1; notebook chỉ điều phối và trình bày kết quả audit.
6. Chạy unit tests, smoke/integration test, quality assertions và reconciliation.
7. Export Parquet/CSV/docs, đọc lại output để xác minh schema/count/key; không chỉ kiểm tra file tồn tại.
8. Restart-and-run-all hai notebook; xác nhận notebook EDA chỉ đọc processed data và không mutate output canonical.
9. So sánh số liệu cuối với baseline đã biết, giải thích mọi sai lệch; kiểm tra `git diff` để không ghi đè thay đổi không liên quan.
10. Handoff bằng danh sách file/path, command tái tạo, test đã chạy, row/schema cuối, seed, quyết định cleaning, known limitations và phần đang chờ Member 2.

### Tiêu chí Codex phải dừng và hỏi thay vì tự quyết

- Raw CSV/header khác đáng kể so với contract hoặc có phiên bản dataset khác.
- Có duplicate `anime_id` với metadata mâu thuẫn mà quy tắc deterministic đề xuất làm mất thông tin quan trọng.
- Repository đã có pipeline/split protocol trái với plan và việc thay đổi có thể ảnh hưởng phần việc của thành viên khác.
- Cần tải dependency/dataset mới, thay đổi phiên bản môi trường lớn, hoặc ghi/commit artifact dung lượng lớn.
- Một quality gate fail trên dữ liệu thật và không thể giải thích bằng nhóm rejected đã được plan cho phép.

Ngoài các trường hợp trên, Codex được phép tự chọn chi tiết implementation nhỏ phù hợp convention hiện hữu, nhưng phải ghi lại lựa chọn và chứng minh bằng test/audit.
