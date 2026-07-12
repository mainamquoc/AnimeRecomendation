# Kế hoạch thực hiện - Thành viên 1: Trưởng nhóm dữ liệu & phân tích

**Dự án:** Hệ thống đề xuất anime — Nhóm 14  
**Phụ trách:** Trương Nhật Trường (Thành viên 1)  
**Mục tiêu phân phối:** biến hai tệp thô trong `database/` thành tập dữ liệu được kiểm soát chất lượng có thể được sử dụng trực tiếp để đào tạo/đánh giá Spark ALS (phản hồi rõ ràng), với siêu dữ liệu/tính năng dành cho đường cơ sở về mức độ phổ biến và dự phòng dựa trên nội dung.

---

## 1. Phạm vi và nguyên tắc dữ liệu

| Danh mục | Thành viên 1 làm | Đầu ra cho Thành viên 2 |
|---|---|---|
| Đang tải dữ liệu | Đọc CSV với lược đồ rõ ràng, kiểm tra kích thước và loại dữ liệu | Hồ sơ thô/bảng kiểm toán |
| Dọn dẹp | Loại trọng điểm không phù hợp, xử lý trùng lặp/mồ côi/null | `clean_ratings`, `clean_anime` |
| EDA | Thống kê chất lượng, thưa thớt, đuôi dài, thể loại/thể loại | 4–6 biểu đồ + thông tin chi tiết |
| Chuẩn bị dữ liệu | Chuẩn bị tương tác cho ALS, danh mục ứng viên, thể loại tính năng | Sàn gỗ + bảng tính năng + từ điển dữ liệu |
| Tài liệu tham khảo | Tóm tắt tài liệu APA 7 về bộ dữ liệu/phương pháp | `references_member1.md` hoặc phần Tài liệu tham khảo trong báo cáo |

### Nguyên tắc bắt buộc

1. **Mô hình chính là phản hồi rõ ràng ALS.** Nhãn chỉ là `rating` của người dùng trong phạm vi **1–10**.
2. `rating = -1` có nghĩa là *đã xem nhưng chưa xếp hạng*, **chưa xếp hạng 0**. Không bao gồm dòng này trong tàu, RMSE, MAE, Precision@K hoặc Recall@K của ALS một cách rõ ràng.
3. Không sử dụng cột `anime.rating` (điểm cộng đồng trung bình) làm tính năng để dự đoán/đánh giá xếp hạng của người dùng: đây là thông tin tổng hợp giống mục tiêu và có nguy cơ rò rỉ. Cột này chỉ được mô tả trong EDA nếu cần.
4. Không có dấu thời gian nên bạn không thể suy ra "xếp hạng mới nhất". Nếu một cặp `(user_id, anime_id)` có nhiều xếp hạng rõ ràng khác nhau, hãy sử dụng **xếp hạng trung bình theo cặp**.
5. Mỗi bước biến đổi phải lưu lại số dòng trước/sau, lý do loại bỏ và hạt giống (nếu có) để tái tạo kết quả.

---

## 2. Kiểm kê dữ liệu hiện có

| Tệp nguồn | Ngũ cốc | Vai trò |
|---|---|---|
| `database/rating.csv` | Tương tác `(user_id, anime_id)` | Nguồn chính cho ALS và đánh giá xếp hạng |
| `database/anime.csv` | Một anime của `anime_id` | Tên anime, thể loại, thể loại, tập, thành viên; sử dụng danh mục, EDA và dự phòng khởi động nguội |

Các điểm cần giải quyết được biết từ hồ sơ dự án:

- `rating.csv`: 7.813.737 lượt tương tác; Có khoảng 18,9% dòng `rating = -1`.
- Sau khi xóa `-1`, có khoảng 6,34 triệu xếp hạng rõ ràng trên khoảng 69,6K người dùng và 9,9K anime; Dữ liệu rất thưa thớt và dài dòng.
- Có `(user_id, anime_id)` trùng lặp và có 3 `anime_id` được xếp hạng không tồn tại trong danh mục.
- Siêu dữ liệu thiếu giá trị trong `genre`, `type`, `rating`; `episodes` có giá trị văn bản như `Unknown` nên không thể truyền trực tiếp.

---

## 3. Cấu trúc đầu ra đề xuất```text
data/
├── processed/
│   ├── clean_ratings.parquet              # explicit interaction data for ALS
│   ├── clean_anime.parquet                # normalized catalogue
│   ├── watched_unrated.parquet            # rating=-1, stored separately; not used to train ALS
│   ├── genre_features.parquet             # anime_id, genre tokens / multi-hot representation
│   ├── data_quality_summary.csv
│   ├── data_dictionary.md
│   └── cleaning_decisions.md
│   # Created during handoff, only after Member 2 has split the data:
│   ├── popularity_baseline.parquet        # Member 2 creates it from train only
│   ├── split_quality_summary.csv           # Member 1 runs the validator on the split
│   └── cold_start_excluded.parquet         # evaluation row that cannot be scored + reason
notebooks/
├── 01_data_preparation.ipynb  # understand raw data → define data contract → clean → validate → export
└── 02_eda_clean_data.ipynb    # reads clean data only, creates EDA charts, and derives insights
outputs/
└── figures/
    ├── rating_distribution.png
    ├── interactions_per_user.png
    ├── interactions_per_anime.png
    ├── genres_and_types.png
    └── data_quality_before_after.png
```
Ưu tiên **Parquet** cho các bảng đã xử lý vì Spark đọc hiệu quả hơn CSV. Không cam kết các tệp Parquet quá lớn; thêm đường dẫn này vào `.gitignore` và cung cấp lệnh để tạo lại từ CSV thô.

---

## 4. Quy trình thực hiện chi tiết

### Bước 1 - Thiết lập lược đồ và tải dữ liệu

- Đọc `rating.csv` với lược đồ: `user_id: int`, `anime_id: int`, `rating: float`.
- Đọc `anime.csv` với lược đồ: `anime_id: int`, `name: string`, `genre: string`, `type: string`, `episodes: string`, `rating: float`, `members: int`.
- Đặt kiểm tra lỗi nhanh: ID khóa không rỗng; `user_id > 0`, `anime_id > 0`; xếp hạng thô chỉ thuộc về `{-1, 1..10}`.
- Lưu kiểm tra ban đầu: số hàng, người dùng/mục riêng biệt, lược đồ, giá trị còn thiếu, xếp hạng tối thiểu/tối đa và số lượng bản ghi cho mỗi giá trị xếp hạng.

**Chấp nhận:** sổ ghi chép có thể chạy từ CSV thô; `data_quality_summary.csv` có hàng `raw`.

### Bước 2 — Tách biệt xếp hạng rõ ràng và chưa xếp hạng đã xem

- Tạo `watched_unrated` từ `rating == -1`, lưu trữ riêng với các cột `user_id`, `anime_id`.
- Tạo `explicit_ratings_raw` từ `rating BETWEEN 1 AND 10`.
- Loại bỏ hoặc gắn cờ tất cả các xếp hạng ngoài hai nhóm trên (nếu có) và ghi nhận vào kiểm toán.

**Quy tắc sử dụng:**

- `watched_unrated`: chỉ sử dụng nếu bạn cần lọc anime đã xem sau khi phục vụ; **không** được sử dụng làm nhãn phủ định.
- `explicit_ratings_raw`: đầu vào duy nhất cho số liệu xếp hạng/xếp hạng ALS.

### Bước 3 — Xử lý các tương tác trùng lặp

- Đếm chính xác các hàng trùng lặp và số lượng cặp `(user_id, anime_id)` có nhiều hơn một xếp hạng rõ ràng.
- Xóa các bản sao giống hệt nhau.
- Với xếp hạng xung đột trong cùng một cặp, nhóm theo `(user_id, anime_id)` và nhận `avg(rating)`; đầu ra phải đảm bảo rằng mỗi cặp chỉ còn lại một dòng.
- Lưu trữ số lượng các cặp đã hợp nhất và mô tả quyết định: không có dấu thời gian nên giá trị trung bình là một lựa chọn trung lập, có thể lặp lại.

### Bước 4 - Đồng bộ hóa tương tác với danh mục

- Kiểm tra chống tham gia từ xếp hạng rõ ràng đến `anime.csv` bởi `anime_id`.
- Loại bỏ 3 mục mồ côi khỏi `clean_ratings` để danh mục đào tạo và danh mục khuyến nghị thống nhất.
- Lưu số dòng/cặp bị loại bỏ khi kiểm tra. Không tạo siêu dữ liệu giả mạo theo cách thủ công cho các mục mồ côi.

**Kết quả sẵn sàng cho ALS:** `clean_ratings` chỉ bao gồm `user_id`, `anime_id`, `rating`; xếp hạng 1–10; `(user_id, anime_id)` độc đáo; mọi `anime_id` đều tồn tại trong `clean_anime`.

> `clean_ratings` là **đầu vào mô hình tiêu chuẩn**, không chỉ là một bảng “sạch”. Mỗi cột phải không có giá trị rỗng và có loại Spark tương thích: `user_id`/`anime_id` là `IntegerType` hoặc `LongType`, `rating` là `FloatType` hoặc `DoubleType`. ALS không cần thể loại hấp dẫn, xếp hạng tiêu chuẩn hóa hoặc ID chuẩn hóa; thể loại/loại chỉ là dữ liệu để dự phòng.

### Bước 5 — Làm sạch danh mục anime

- Giữ `anime_id` duy nhất và xác minh `name` không trống đối với các mục vẫn đang tương tác.
- Chuẩn hóa văn bản: cắt bớt khoảng trắng; thay thế null/trống bằng `Unknown` cho `genre` và `type`.
- Chuyển đổi `episodes` thành số (`episodes_num`); Các giá trị không thể phân tích cú pháp (`Unknown`, `N/A`, …) trở thành `null`, đồng thời tạo `episodes_missing` (0/1).- Chuẩn hóa `members` thành số không âm; tạo `log_members = log1p(members)` khi thực hiện dự phòng/đường cơ sở.
- Giữ `community_rating` (được đổi tên từ `anime.rating`) chỉ dành cho mô tả danh mục/EDA; đính kèm cảnh báo **không sử dụng trong đào tạo hoặc đánh giá xếp hạng**.
- Giữ/không giữ anime không có xếp hạng rõ ràng tùy theo danh mục phục vụ, nhưng phân tách rõ ràng `has_explicit_interaction` để không nhầm là vật phẩm có thể huấn luyện.

### Bước 6 - Chuẩn bị các tính năng cho dự phòng dựa trên nội dung

- Phân tách `genre` bằng dấu phẩy, cắt bớt, chuẩn hóa tên token và loại token trống.
- Tạo một trong hai biểu diễn có thể tái tạo:
  - `genre_tokens`: `anime_id`, `genres: array<string>` — ưu tiên Spark `CountVectorizer`; hoặc
  - ma trận thể loại đa dạng hấp dẫn nếu được thực hiện bằng pandas/scikit-learn.
- Chuẩn bị `type` được phân loại cho đường ống dự phòng nóng một lần khi cần.
- Dành cho người dùng mới bắt đầu: lấy những người dùng anime đã ghi điểm `>= 8`, tìm những anime có độ tương đồng về thể loại cosine cao, sau đó lọc tất cả những người dùng anime đã tương tác (rõ ràng **và**, nếu phục vụ thực tế, đã xem chưa được xếp hạng).
- Với các mặt hàng bắt đầu nguội hoặc người dùng không có lịch sử tích cực: dự phòng vào danh sách phổ biến. Không kết hợp nội dung và điểm ALS trong chỉ số chính trừ khi nhóm xác định cụ thể giao thức kết hợp.

### 4.1. Hợp đồng thực thi để Codex triển khai mà không cần suy đoán

#### Môi trường và cấu hình

- Việc triển khai mặc định sử dụng **PySpark DataFrame API** để tải, dọn dẹp, kiểm tra và xuất. Chỉ chuyển dữ liệu tổng hợp nhỏ sang gấu trúc/sinh vật biển trong sổ ghi chép EDA; Không gọi `toPandas()` trên bảng tương tác đầy đủ.
- Phiên bản Python, Java và PySpark phải được nêu rõ trong README/yêu cầu của dự án. Nếu kho lưu trữ chưa được lập phiên bản, Codex phải kiểm tra môi trường hiện có trước, chọn một bộ phiên bản tương thích và ghi lại lựa chọn; Đừng âm thầm thay đổi khung thành gấu trúc.
- Không mã hóa đường dẫn và tham số trong toàn bộ sổ ghi chép. Tạo một ô/cấu hình (hoặc `src/config.py`) có ít nhất:```python
RAW_RATINGS_PATH = "database/rating.csv"
RAW_ANIME_PATH = "database/anime.csv"
PROCESSED_DIR = "data/processed"
FIGURES_DIR = "outputs/figures"
SEED = 42
POSITIVE_THRESHOLD = 8.0
TOP_K_VALUES = [5, 10]
UNKNOWN_TOKEN = "Unknown"
```
- Notebook phải chạy từ **root kho lưu trữ**. Trước khi xử lý, hãy kiểm tra xem có tồn tại hai tệp thô hay không; tạo thư mục đầu ra nếu thiếu; Lỗi phải nêu rõ đường dẫn nào bị thiếu.
- Phiên Spark phải có tên ứng dụng rõ ràng, múi giờ cố định (được khuyến nghị `UTC`) và cấp độ nhật ký phù hợp. Writing Parquet chỉ sử dụng `mode("overwrite")` với các đường dẫn phụ chính xác được xác định trong `PROCESSED_DIR`; Đừng xóa thư mục mẹ nữa.

#### Các mô-đun/chức năng phải được tách riêng để sổ ghi chép chỉ có thể được sắp xếp

Nếu dự án cho phép tạo mã nguồn, hãy ưu tiên cấu trúc sau để logic có thể được unit test và tái sử dụng:```text
src/
├── config.py
├── schemas.py          # StructTypes for the two CSV files and output schemas
├── data_io.py          # load_raw_data(), write_parquet_safe()
├── cleaning.py         # clean_ratings(), clean_anime(), build_genre_features()
├── audit.py            # profile_stage(), validate_*(), append_audit_row()
└── split_validation.py # validate_split(), classify_cold_start()
tests/
├── test_cleaning.py
├── test_data_contracts.py
└── test_split_validation.py
```
Chữ ký/hành vi tối thiểu:```python
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
Codex có thể đổi tên các hàm theo quy ước hiện có của kho lưu trữ, nhưng phải giữ nguyên hợp đồng đầu vào/đầu ra và không nhúng tất cả logic nghiệp vụ vào các ô sổ ghi chép.

#### Thứ tự phụ thuộc bắt buộc```text
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
Lưu ý: cần có bộ `clean_anime`/danh sách khóa hợp lệ trước khi mồ côi, nhưng cờ `has_explicit_interaction` chỉ được tính sau khi `clean_ratings` hoàn tất. Tránh sự phụ thuộc vòng tròn bằng cách làm sạch danh mục theo hai giai đoạn: chuẩn hóa khóa/văn bản trước, thêm cờ thống kê tương tác sau.

### 4.2. Chuyển đổi quy tắc ở cấp mã

#### Xếp hạng/tương tác

1. Phân tích CSV theo lược đồ đã khai báo. Một bản ghi không đúng định dạng hoặc không thể phân tích cú pháp không thể tự động trở thành rỗng và sau đó biến mất; phải được tính vào nhóm `invalid_schema`/`invalid_id`/`invalid_rating`.
2. Phân loại hồ sơ theo điều kiện loại trừ lẫn nhau:
   - `watched_unrated`: `rating == -1`;
   - `explicit_valid`: `1 <= rating <= 10`;
   - `rejected_invalid_rating`: tất cả các trường hợp còn lại, bao gồm cả null/NaN nếu có.
3. Đối với `watched_unrated`, hãy áp dụng `dropDuplicates(["user_id", "anime_id"])` và danh mục chống tham gia như dữ liệu rõ ràng. Chỉ định riêng các số trùng lặp và số mồ côi chính xác; Đừng để cùng một cặp lặp lại trong tệp bộ lọc phục vụ.
4. Với dữ liệu rõ ràng, hãy tính toán số liệu kiểm tra **trước khi tổng hợp**: số hàng trùng lặp chính xác, số cặp trùng lặp, số cặp có `countDistinct(rating) > 1`. Sau đó nhóm theo cặp và chuyển `avg(rating)` sang `double`. Không có nghĩa tròn; phạm vi vẫn phải nằm trong `[1, 10]`.
5. Kiểm tra mồ côi bằng cách sử dụng `left_anti` tham gia vào danh sách `anime_id` riêng biệt của danh mục. Lưu bảng bị từ chối bằng `reason` nếu dung lượng nhỏ; Nếu không lưu chi tiết thì ít nhất hãy lưu số đếm và danh sách mồ côi `anime_id` trong `cleaning_decisions.md`.
6. Sắp xếp không phải là một phần của hợp đồng dữ liệu. Không viết bài kiểm tra phụ thuộc vào thứ tự hàng trong Parquet; kiểm tra bằng phím/đếm/bộ hoặc xóa thứ tự khi được hiển thị.

#### Danh mục

1. `anime_id`: dương, không rỗng, duy nhất. Nếu ID khớp:
   - trùng lặp chính xác: giữ một hàng;
   - siêu dữ liệu không nhất quán: không có lựa chọn ngẫu nhiên; Ưu tiên hàng có nhiều trường không rỗng nhất, liên kết với các quy tắc xác định (ví dụ: từ vựng), số lượng bản ghi và chính sách trong kiểm tra.
2. `name`: cắt tỉa; chuỗi trống thành null. Anime có tương tác rõ ràng nhưng thiếu tên là lỗi cổng chất lượng; anime không có tương tác có thể sử dụng `Unknown` nếu nhóm muốn giữ lại trong danh mục phục vụ và phải ghi lại quyết định.
3. `genre`: chứa cột văn bản chuẩn hóa để hiển thị và mảng `genres` cho các tính năng. Mã thông báo được cắt bớt, trống, khoảng trắng/chữ thường được chuẩn hóa một cách xác định, `array_distinct`, sau đó được sắp xếp để có đầu ra ổn định. `Unknown` không thể là tín hiệu thể loại khi tính toán độ tương tự cosin; được biểu thị bằng một mảng trống hoặc loại mã thông báo này khi vector hóa.
4. `type`: cắt, trống/null thành `Unknown`; Không kết hợp các loại khác nhau khi chưa có bản đồ được phê duyệt.
5. `episodes_num`: phân tích số nguyên dương; `Unknown`, `N/A`, trống, 0 hoặc âm thành null và `episodes_missing = 1`. Nếu có số thập phân bất thường, hãy từ chối/gắn cờ thay vì cắt ngắn một cách im lặng.
6. `members`: phân tích số nguyên/dài không âm. Không hợp lệ thành null và có cờ `members_missing`; Chỉ tính `log_members` khi thành viên hợp lệ. Không sử dụng `members` hoặc `log_members` trong chỉ số ALS chính.7. `community_rating`: số trong `[1, 10]` hoặc null; Các giá trị ngoài phạm vi sẽ trở thành null/cờ. Cột này không được xuất sang `clean_ratings` hoặc vectơ đặc trưng của mô hình chính.

### 4.3. Lược đồ đầu ra cố định

| Đầu ra | Lược đồ được đề xuất/nullable | Phân vùng/thứ tự | Người tiêu dùng |
|---|---|---|---|
| `clean_ratings.parquet` | `user_id: long not null`, `anime_id: long not null`, `rating: double not null` | không cần phân vùng; cặp độc nhất | ALS của thành viên 2 |
| `watched_unrated.parquet` | `user_id: long not null`, `anime_id: long not null` | cặp độc nhất | phục vụ bộ lọc |
| `clean_anime.parquet` | `anime_id: long not null`, `name: string not null`, `genre: string`, `genres: array<string>`, `type: string not null`, `episodes_num: int`, `episodes_missing: int not null`, `members: long`, `members_missing: int not null`, `log_members: double`, `community_rating: double`, `has_explicit_interaction: boolean not null` | `anime_id` độc đáo | danh mục/EDA/dự phòng |
| `genre_features.parquet` | `anime_id: long not null`, `genres: array<string> not null` | `anime_id` độc đáo; không chứa mã thông báo trống/Không xác định | dự phòng nội dung |
| `data_quality_summary.csv` | Xem kiểm tra lược đồ bên dưới | sắp xếp theo `stage`, `metric` khi xuất | báo cáo + khả năng tái tạo |

Không lưu Spark ML `VectorUDT` làm hợp đồng duy nhất của `genre_features`, vì từ vựng/chỉ mục có thể khó đọc và phụ thuộc vào bước phù hợp. Giữ `genres: array<string>` làm nguồn chuẩn; Thành viên 2 phù hợp với `CountVectorizer` trên tàu/ứng viên phù hợp và lưu từ vựng/mô hình nếu cần vectơ.

### 4.4. Hợp đồng kiểm toán và đối chiếu

`data_quality_summary.csv` phải dài để dễ dàng bổ sung các số liệu:

| Cột | Ý nghĩa |
|---|---|
| `run_id` | ID xác định hoặc dấu thời gian của lần chạy; cùng một lần chạy sử dụng cùng một ID |
| `stage` | `raw`, `parsed`, `explicit_before_dedup`, `explicit_after_dedup`, `clean`, `export_readback` |
| `entity` | `ratings`, `watched_unrated`, `anime`, `genre_features` |
| `metric` | ví dụ `row_count`, `distinct_users`, `null_rating`, `duplicate_pairs`, `orphan_rows` |
| `value` | giá trị số |
| `unit` | `rows`, `pairs`, `users`, `items`, `percent` |
| `rule_or_note` | số liệu tạo điều kiện/chính sách |

Các phương trình đối chiếu phải vượt qua và được khẳng định trong mã:```text
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
Vì nhiều hàng được tổng hợp thành một cặp nên biến `rows_removed_as_duplicate_or_aggregated` phải được tính toán bằng cách sử dụng chênh lệch số lượng hàng, chứ không phải thêm số lượng trùng lặp chính xác vào số lượng cặp xung đột một cách ngây thơ. Tương tự, kiểm toán phải phân biệt giữa `row_count` và `distinct_pair_count`.

Sau khi xuất, hãy đọc lại từng Parquet và so sánh lược đồ, số hàng, số khóa riêng biệt với DataFrame trước khi xuất. Chỉ đánh dấu quy trình là thành công sau khi vượt qua quá trình xác thực đọc lại.

### 4.5. Yêu cầu kiểm tra và Định nghĩa Hoàn thành cho mã

#### Kiểm tra đơn vị bằng vật cố định tổng hợp nhỏ

Lịch thi đấu phải chứa ít nhất: xếp hạng `-1`, xếp hạng cận biên 1 và 10, xếp hạng không hợp lệ/null, trùng lặp chính xác, trùng lặp xung đột, anime mồ côi, thể loại/loại null, tập `Unknown`, thành viên phủ định, siêu dữ liệu anime trùng lặp và thể loại có khoảng trắng/mã thông báo trùng lặp.

Các bài kiểm tra bắt buộc:

- bộ phân loại `-1` không nhập rõ ràng và không chuyển thành 0;
- xung đột trùng lặp được tổng hợp với ý nghĩa chính xác, mỗi cặp có một hàng;
- ID/xếp hạng không hợp lệ và trẻ mồ côi được tính/loại bỏ vì lý do chính xác;
- chuẩn hóa danh mục cho các kết quả xác định;
- mã thông báo thể loại không trống, không trùng lặp và không chứa tín hiệu giống `Unknown`;
- Trình xác thực không thành công khi xếp hạng nằm ngoài phạm vi, khóa null, cặp trùng lặp hoặc mồ côi vẫn tồn tại;
- trình xác thực phân tách phân loại chính xác `user_unseen`, `item_unseen`, `both`;
- đọc/ghi chuyến đi khứ hồi giữ lược đồ và số lượng.

#### Tích hợp/thử nghiệm khói trên dữ liệu thô

- Chạy quy trình từ đầu đến cuối trên CSV thô (hoặc mẫu giới hạn chỉ dành cho thử nghiệm khói), tạo ra đủ đầu ra và không có lỗi cổng chất lượng.
- So sánh các số đã biết ở Phần 2 với độ sai số hợp lý; Nếu sai, hãy nêu rõ là do phiên bản tập dữ liệu hay do biến đổi. Đừng chỉnh sửa xác nhận để buộc vượt qua.
- Notebook khởi động lại và chạy tất cả thành công từ kernel sạch. Không phụ thuộc vào các biến được tạo thủ công ở ô trước đó không đúng thứ tự.
- Viết lệnh tạo lại và kiểm tra trong README, ví dụ `pytest -q` và lệnh chạy notebook/script thực tế của kho lưu trữ. Codex phải sử dụng chính xác trình chạy đã tồn tại trong repo; Nếu không, hãy tạo một điểm vào rõ ràng.

#### Định nghĩa Hoàn thành

Một bước chỉ được coi là hoàn thành khi nó đồng thời có: chuyển đổi mã, kiểm tra trước/sau, chất lượng xác nhận, lược đồ đầu ra chính xác, mô tả quyết định và thử nghiệm tương ứng trong tài liệu. Đừng coi việc sổ ghi chép "chạy không có lỗi" là đủ nếu việc đối chiếu hoặc đọc lại chưa được thực hiện.

### 4.6. Xử lý lỗi, ghi nhật ký và khả năng tái tạo

- Quy trình không thành công với các lỗi có thể hiểu được như thiếu tệp, tiêu đề lược đồ sai, ID/xếp hạng vi phạm hợp đồng vượt quá nhóm bị từ chối đã xác định, danh mục trùng lặp không thể được giải quyết một cách xác định hoặc đầu ra đọc lại không khớp.
- Cảnh báo nhưng vẫn chạy do thiếu siêu dữ liệu tùy chọn (`genre`, `type`, `episodes`, `members`, `community_rating`) nếu chính sách tiêu chuẩn hóa ở trên có thể xử lý được; cảnh báo đếm phải được đưa vào kiểm toán.- Không ghi lại tất cả hồ sơ/dữ liệu người dùng. Chỉ hiển thị các mẫu nhỏ, tổng hợp và tối đa danh sách ID lỗi cần điều tra.
- Mỗi lần lấy mẫu phải có hạt giống và ghi lại tỷ lệ/giới hạn mẫu. Mỗi tổng hợp biểu đồ phải có lệnh xây dựng lại từ dữ liệu sạch.
- Ghi lại dấu vân tay tập dữ liệu tối thiểu bao gồm kích thước tệp, thời gian sửa đổi và (nếu chi phí chấp nhận được) SHA-256 của tệp thô trong `cleaning_decisions.md` hoặc bảng kê khai để phát hiện các thay đổi của dữ liệu thô.

### 4.7. Ranh giới trách nhiệm để tránh tạo ra các hiện vật không đúng lúc

- Thành viên 1 tạo bốn bảng chuẩn: `clean_ratings`, `clean_anime`, `watched_unrated`, `genre_features` và Audit/docs/figures.
- `popularity_baseline.parquet` **không được tạo từ dữ liệu sạch hoàn toàn**. Thành viên 2 tạo ra nó sau khi chỉ tách khỏi tàu; Thành viên 1 chỉ cung cấp người trợ giúp/hợp đồng và xem xét rò rỉ. Vì vậy, tập tin này không được đưa vào “bốn bảng” bàn giao ban đầu.
- `split_quality_summary.csv` và `cold_start_excluded` chỉ có dữ liệu thực sau khi Thành viên 2 cung cấp đào tạo/xác nhận/kiểm tra. Thành viên 1 chuẩn bị chức năng/lược đồ/kiểm tra trước, sau đó chạy xác thực ở bước chuyển giao; Không tạo tệp giả mạo hoặc dữ liệu giữ chỗ.
- Nếu Codex phân công nhiệm vụ Thành viên 1 riêng biệt nhưng không tồn tại sự phân chia thì sản phẩm phân phối hợp lệ là lược đồ trợ giúp + kiểm tra + mẫu và ghi lại trạng thái `pending Member 2 split` trong ghi chú bàn giao.

---

## 5. Chuẩn bị dữ liệu để đào tạo/đánh giá mô hình đúng cách

###Hợp đồng dữ liệu đã bàn giao cho Member 2

| Bảng | Cột bắt buộc | Điều kiện |
|---|---|---|
| `clean_ratings` | `user_id`, `anime_id`, `rating` | xếp hạng 1–10, một dòng/cặp, mục trong danh mục |
| `clean_anime` | `anime_id`, `name`, `genre`, `type`, `episodes_num`, `members` | một dòng/anime; null được chuẩn hóa theo từ điển dữ liệu |
| `genre_features` | `anime_id`, `genres` hoặc thể loại vector | chỉ phục vụ dự phòng nội dung |
| `watched_unrated` | `user_id`, `anime_id` | Không sử dụng làm nhãn/số liệu rõ ràng |

### Điều kiện bắt buộc của tập dữ liệu trước khi huấn luyện ALS

Đây là những điều kiện mà Thành viên 1 phải kiểm tra và ghi vào biên bản bàn giao. Nếu một điều kiện không được đáp ứng, đừng gọi mô hình là "được đào tạo/đánh giá chính xác".

| Nhóm | Bộ dữ liệu phải đáp ứng | Cách xử lý nếu không đạt |
|---|---|---|
| Ngũ cốc | Một dòng duy nhất cho một cặp `(user_id, anime_id)` | Xóa trùng lặp chính xác, xung đột xếp hạng tổng hợp với giá trị trung bình |
| Nhãn | `rating` dạng số, không rỗng, chỉ có trong `[1, 10]` | Xóa `-1` và mọi giá trị không hợp lệ khỏi đầu vào rõ ràng; đánh giá không bị buộc tội |
| ID | `user_id`, `anime_id` số, dương, không rỗng | Bản ghi loại/cờ không hợp lệ; không mã hóa lại ID nếu không cần thiết |
| Tính toàn vẹn tham chiếu | Mỗi mục trong tương tác đều có tên/ứng cử viên trong danh mục | Nhập tương tác mồ côi hoặc tách nó khỏi bảng sẵn sàng cho mô hình |
| Hỗ trợ đào tạo | Mỗi người dùng và vật phẩm xuất hiện trong **tàu** ít nhất một lần; Đề xuất ít nhất 2 tương tác/người dùng để duy trì thứ hạng thử nghiệm | Bao gồm các tương tác được hỗ trợ không đầy đủ trong `cold_start_excluded` và số lượng/tỷ lệ báo cáo; Đừng để Spark rơi mà không ghi || Bộ ứng cử viên | Các vật phẩm được đề xuất phải thuộc danh mục vật phẩm và có hệ số train | Tạo ứng viên từ `train_items`, không phải từ toàn bộ thô/xác thực/kiểm tra |
| Rò rỉ | Không sử dụng `community_rating`, số liệu thống kê mức độ phổ biến hoặc vectơ thể loại phù hợp từ xác nhận/kiểm tra đến điểm/đánh giá ALS | Chỉ phù hợp với mọi số liệu thống kê/vectorizer trên tàu; siêu dữ liệu tĩnh chỉ được sử dụng để dự phòng, tách biệt với số liệu ALS |

**Lưu ý về lọc hỗ trợ:** Không lọc mạnh mẽ tất cả người dùng/vật phẩm tần suất thấp trực tiếp từ `clean_ratings`, vì việc này sẽ thiên vị dữ liệu về phía người dùng đang hoạt động/anime phổ biến. Giữ lại `clean_ratings` đầy đủ sau khi làm sạch, sau đó tạo báo cáo `cold_start_excluded` và sửa phần tách để đánh giá tập hợp con có thể cho điểm. Nếu nhóm muốn lọc (ví dụ: người dùng có ít hơn 2 xếp hạng rõ ràng), nhóm phải báo cáo rõ ràng số lượng bản ghi/người dùng/mục bị xóa và áp dụng bộ lọc một cách nhất quán cho giao thức xếp hạng.

### Checklist trước khi đào tạo và bàn giao sau khi chia tách

Thành viên 1 chuẩn bị các hàm kiểm tra hoặc ô sổ ghi chép để Thành viên 2 chạy lại sau khi tạo phần tách:```text
clean_ratings
  ├── ratings_als_input                 # user_id, anime_id, rating; passes all model-input rules
  ├── train / validation / test          # Member 2 creates these with a fixed seed
  ├── eval_warm_validation / eval_warm_test
  │     # only rows whose user_id and anime_id both exist in train
  └── cold_start_excluded
        # test/validation interactions ALS cannot score; reason=user_unseen/item_unseen/both
```
Số liệu phải xuất hiện trong `split_quality_summary.csv`: số lượng hàng, người dùng/mục riêng biệt, trung bình/tiêu chuẩn xếp hạng, số lần trùng lặp, số lượng người dùng/mục chỉ có một tương tác, số lượng hàng ấm/lạnh cho mỗi phần tách và mức độ bao phủ `warm_rows / total_eval_rows`. Tệp này là bằng chứng giải thích tại sao số lượng bản ghi để tính RMSE/MAE có thể nhỏ hơn số lượng bản ghi kiểm tra ban đầu.

### Yêu cầu tách để tránh số liệu sai

Thành viên 1 chuẩn bị/kiểm tra trình trợ giúp phân tách hoặc đánh giá mã của Thành viên 2 theo các quy tắc sau:

1. Chỉ chia **sau khi làm sạch**, trên `clean_ratings`.
2. Sử dụng hạt giống cố định cho toàn bộ nhóm (`42` được khuyến nghị) và lưu hạt giống trong config/README.
3. Với RMSE/MAE: chuyến tàu/xác thực/kiểm tra ngẫu nhiên 80/10/10 có thể chấp nhận được, nhưng xác nhận/kiểm tra chỉ tính các cặp trong đó **cả người dùng và vật phẩm đều xuất hiện trong chuyến tàu**. Các dòng khởi động nguội phải được báo cáo riêng dưới dạng phạm vi phủ sóng/số lượng bị loại trừ, không được Spark `coldStartStrategy='drop'` bao gồm.
4. Trước khi phân chia ngẫu nhiên, hãy ưu tiên **phân chia sửa chữa**: nếu một người dùng/vật phẩm chỉ rơi vào giai đoạn xác thực/kiểm tra và không còn tương tác nào trong nhóm, hãy chuyển ít nhất một tương tác của người dùng/vật phẩm đó sang nhóm. Nếu không được sửa chữa, tương tác này phải được xóa khỏi chỉ số và được ghi lại trong `cold_start_excluded`. Đừng bỏ rơi âm thầm.
5. Với Top-N: chia theo người dùng (ví dụ: giữ 20% tích cực hoặc loại bỏ một lần), nhưng chỉ giữ lại những tích cực của người dùng có đủ lịch sử để vẫn có ít nhất một chuỗi tương tác. Đã sửa định nghĩa liên quan: `rating >= 8`; báo cáo tối thiểu P@10 và R@10, khuyến nghị bổ sung @5.
6. Ứng viên đề xuất phải là `train_items` và phải loại trừ những người dùng anime đã tương tác trên tàu. Khi demo/phục vụ, bạn cũng nên xóa `watched_unrated` để tránh gợi ý những anime bạn đã xem.
7. Đường cơ sở về mức độ phổ biến phải được tính **chỉ từ phần tách đoàn tàu**, không phải từ toàn bộ tập dữ liệu/kiểm tra; Sử dụng số lượng xếp hạng rõ ràng và xếp hạng trung bình với số lượng/độ thu nhỏ tối thiểu để tránh ưu tiên các mục có quá ít xếp hạng.

---

## 6. Cần phải thực hiện EDA và viết những hiểu biết sâu sắc

Tất cả các biểu đồ đều có tiêu đề, trục, đơn vị và chú thích ngắn; Sử dụng hạt giống nếu có sẵn mẫu. Với `rating.csv` lớn, bạn có thể tổng hợp trên Spark trước rồi chuyển bảng nhỏ sang pandas/seaborn để vẽ.

| Biểu đồ/bảng | Cách tính | Ý nghĩa cần được nêu rõ |
|---|---|---|
| Phân phối xếp hạng | Đếm/phần trăm theo `-1`, 1…10 | Chứng minh tại sao `-1` phải được tách khỏi nhãn rõ ràng |
| Chất lượng trước/sau khi vệ sinh | Hàng, người dùng, mục, cặp trùng lặp, hàng mồ côi, thiếu siêu dữ liệu | Tính minh bạch và khả năng tái tạo của chuyển đổi |
| Tương tác trên mỗi người dùng (thang log) | Số lượng xếp hạng/người dùng rõ ràng, biểu đồ/phần trăm | Cho biết người dùng có đuôi dài và khởi động nguội |
| Tương tác trên mỗi anime (thang log) | Số lượng xếp hạng rõ ràng/anime, biểu đồ/phần trăm | Biểu thị xu hướng phổ biến/đuôi dài của mặt hàng |
| Phân phối thể loại/loại | Số lượng anime và/hoặc số lượng tương tác rõ ràng theo thể loại/loại | Mô tả danh mục, động cơ sử dụng thể loại dự phòng || Phim hoạt hình hàng đầu | Đứng đầu theo số lượng xếp hạng rõ ràng và đứng đầu theo thành viên | Phân biệt mức độ phổ biến trong tương tác với siêu dữ liệu cộng đồng |

Thông tin chi tiết tối thiểu có trong báo cáo/PPT:

- Vấn đề là ma trận hạng mục người dùng thưa thớt nên ma trận ALS phân tích nhân tử phù hợp hơn mô hình tra cứu dày đặc/đơn giản.
- Sentinel `-1` có nghĩa là thiếu phản hồi rõ ràng chứ không phải là không thích.
- Chỉ đuôi dài như RMSE thôi là chưa đủ; Cần Precision@K/Recall@K và phạm vi bảo hiểm.
- Thể loại này hữu ích cho việc dự phòng/khởi động nguội, nhưng không thay thế được tính công bằng của giao thức kiểm tra ALS.

---

##7. Cổng chất lượng trước khi bàn giao

- [ ] `clean_ratings.rating` nằm trong `[1, 10]`; không còn `-1` nữa.
- [ ] Ba cột ALS (`user_id`, `anime_id`, `rating`) đều không rỗng, đúng loại số và ID dương.
- [ ] `clean_ratings` không còn trùng lặp `(user_id, anime_id)` nữa.
- [ ] Tất cả `clean_ratings.anime_id` đều có thể được nối với `clean_anime.anime_id`.
- [ ] Lược đồ, số dòng trước/tiếp theo, số trùng lặp/mồ côi/null đều có trong `data_quality_summary.csv`.
- [ ] `clean_anime` có 1 dòng/anime; thể loại/loại đã có giá trị hợp lệ (`Unknown` khi thiếu); `episodes_num` số thực/null.
- [ ] `genre_features` tham gia 1:1 với danh mục và không có mã thông báo trống.
- [ ] Không có tính năng/mô hình đầu vào nào sử dụng `community_rating` hoặc mức độ phổ biến của thử nghiệm/tập dữ liệu đầy đủ gây ra rò rỉ.
- [ ] Sau khi chia tách, `split_quality_summary.csv` có vùng phủ sóng ấm/lạnh và không bao gồm lý do; Mỗi hàng được tính RMSE/MAE đều có người dùng/mục trong đoàn tàu.
- [ ] Bộ ứng cử viên của Bảng xếp hạng chỉ chứa `train_items`; Việc giữ lại Top-N không xóa tất cả lịch sử tích cực của người dùng khỏi tàu.
- [ ] Kiểm tra người dùng mẫu: chọn ứng viên hàng đầu, xác nhận bộ lọc loại đào tạo tương tác (và đã xem chưa được xếp hạng khi phục vụ).
- [ ] Notebook chạy lại thành công từ CSV thô và tạo cùng một lược đồ/đầu ra.

---

## 8. Sản phẩm của thành viên 1

1. `notebooks/01_data_preparation.ipynb`: sổ ghi chép xử lý chính chạy tuần tự từ tải → khám phá dữ liệu thô và ý nghĩa cột → xác định hợp đồng dữ liệu cho ALS/Top-N → kiểm tra → dọn dẹp → kiểm tra cổng chất lượng → tạo tính năng → xuất Parquet. Không thực hiện biểu đồ EDA/báo cáo trong sổ tay này ngoài danh sách kiểm tra cần thiết để đưa ra quyết định làm sạch.
2. `notebooks/02_eda_clean_data.ipynb`: chỉ đọc các bảng Parquet đã được làm sạch; Tạo 4–6 biểu đồ EDA, chú thích và thông tin chi tiết để đưa vào báo cáo/PPT. Sổ ghi chép này không thay đổi dữ liệu đã xử lý.
3. `data/processed/` (hoặc liên kết/lưu bên ngoài Git): bốn bảng chuẩn trong Phần 3 (`clean_ratings`, `clean_anime`, `watched_unrated`, `genre_features`).
4. `data_quality_summary.csv`, `data_dictionary.md`, `cleaning_decisions.md`; thêm `split_quality_summary.csv` và `cold_start_excluded.parquet` sau khi nhận được phần chia thực tế từ Thành viên 2.
5. 4–6 hình ảnh EDA trong `outputs/figures/` để Thành viên 2 đưa vào PPT/báo cáo.
6. `references_member1.md`: Tài liệu APA 7 về bộ dữ liệu, Spark MLlib/ALS và đánh giá đề xuất.
7. Ghi chú bàn giao cho Thành viên 2: đường dẫn bảng, lược đồ, số bản ghi cuối cùng, hạt giống, chính sách trùng lặp, ngưỡng liên quan và các hạn chế về dữ liệu.

---

## 9. Kế hoạch thời gian gợi ý

| Phiên | Công việc | Tiêu chí hoàn thành |
|---|---|---|| 1 | Tải, lược đồ, kiểm tra thô, xác nhận quy tắc | Có hồ sơ chất lượng thô và quyết định làm sạch được ghi lại |
| 2 | Tách `-1`, loại bỏ trùng lặp, xử lý trẻ mồ côi, làm sạch siêu dữ liệu | Xuất sàn gỗ và tất cả các cổng chất lượng đều vượt qua dữ liệu |
| 3 | Tính năng thể loại/loại, EDA, chú thích/thông tin chi tiết | Có 4–6 số liệu sẵn sàng được đưa vào báo cáo/PPT |
| 4 | Handoff + review chia/rò rỉ với Thành viên 2 | Thành viên 2 huấn luyện ALS từ `clean_ratings`; README có thể tái tạo |

---

## 10. Tài liệu tham khảo APA 7 ban đầu

- Liên hiệp Cooper. (nd). *Cơ sở dữ liệu đề xuất anime* [Bộ dữ liệu]. Kaggle. https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database
- Koren, Y., Bell, R., & Volinsky, C. (2009). Kỹ thuật nhân tố ma trận cho các hệ thống tư vấn. *Máy tính, 42*(8), 30–37. https://doi.org/10.1109/MC.2009.263
- Meng, X., Bradley, J., Yavuz, B., Sparks, E., Venkataraman, S., Liu, D., Freeman, J., Tsai, D. B., Amde, M., Owen, S., Xin, D., Xin, R., Franklin, M. J., Zadeh, R., Zaharia, M., & Talwalkar, A. (2016). MLLib: Học máy trong Apache Spark. *Tạp chí Nghiên cứu Học máy, 17*(34), 1–7. http://jmlr.org/papers/v17/15-237.html
- Herlocker, J. L., Konstan, J. A., Terveen, L. G., & Riedl, J. T. (2004). Đánh giá hệ thống tư vấn lọc cộng tác. *Giao dịch ACM trên Hệ thống Thông tin, 22*(1), 5–53. https://doi.org/10.1145/963770.963772

> Lưu ý: Thành viên 1 chuẩn hóa APA theo thứ tự bảng chữ cái trong báo cáo cuối cùng/PPT; Kiểm tra URL và định dạng in nghiêng khi bố trí trang LaTeX.

---

## 11. Danh sách kiểm tra triển khai cho Codex

Khi được yêu cầu viết mã theo kế hoạch này, Codex thực hiện theo thứ tự sau và không bỏ qua bước xác minh:

1. Đọc `README`, `AGENTS.md` (nếu có), tệp phụ thuộc và toàn bộ gói; Kiểm tra cấu trúc kho lưu trữ và trạng thái của các tệp hiện có trước khi chỉnh sửa.
2. Tiêu đề/lược đồ/mẫu hồ sơ của hai tệp CSV thô và so sánh với Phần 2; Đừng tải toàn bộ thứ bằng gấu trúc.
3. Hoàn thiện điểm vào, cấu hình, lược đồ và hợp đồng đầu ra; chỉ định bất kỳ giả định mới nào trong `cleaning_decisions.md`.
4. Viết các bài kiểm tra tổng hợp trước hoặc song song với mỗi biến đổi quan trọng, đặc biệt là phân tích trọng điểm, trùng lặp, mồ côi và siêu dữ liệu.
5. Triển khai quy trình theo phụ thuộc tại Mục 4.1; Sổ ghi chép chỉ phối hợp và trình bày kết quả kiểm toán.
6. Chạy thử nghiệm đơn vị, thử nghiệm khói/tích hợp, xác nhận và đối chiếu chất lượng.
7. Xuất Parquet/CSV/docs, đọc lại kết quả đầu ra để xác minh lược đồ/đếm/khóa; Đừng chỉ kiểm tra tập tin tồn tại.
8. Khởi động lại và chạy tất cả hai sổ ghi chép; Xác nhận rằng sổ ghi chép EDA chỉ đọc dữ liệu đã xử lý và không thay đổi dữ liệu chuẩn đầu ra.
9. So sánh dữ liệu cuối cùng với đường cơ sở đã biết, giải thích mọi sai lệch; kiểm tra `git diff` để không ghi đè những thay đổi không liên quan.
10. Chuyển giao với danh sách tệp/đường dẫn, lệnh được tạo lại, chạy thử nghiệm, hàng/lược đồ cuối cùng, hạt giống, quyết định làm sạch, các giới hạn đã biết và Thành viên 2 đang chờ xử lý.

### Tiêu chí Codex là dừng lại và hỏi thay vì tự mình quyết định

- CSV/tiêu đề thô khác biệt đáng kể so với hợp đồng hoặc có phiên bản tập dữ liệu khác.- Có một `anime_id` trùng lặp với siêu dữ liệu xung đột mà quy tắc xác định cho thấy sẽ mất thông tin quan trọng.
- Kho đã có giao thức pipe/split trái với kế hoạch và việc thay đổi có thể ảnh hưởng đến công việc của các thành viên khác.
- Cần tải xuống các phụ thuộc/bộ dữ liệu mới, thay đổi các phiên bản môi trường lớn hoặc viết/xác nhận các tạo phẩm lớn.
- Lỗi cổng chất lượng trên dữ liệu thực mà nhóm bị từ chối được kế hoạch cho phép không thể giải thích được.

Ngoài các trường hợp trên, Codex được phép lựa chọn các chi tiết triển khai nhỏ phù hợp với các quy ước hiện có nhưng phải ghi lại lựa chọn đó và chứng minh bằng thử nghiệm/kiểm toán.