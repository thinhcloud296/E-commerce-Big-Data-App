# Phân tích Dữ liệu Thương mại điện tử — Spark + HDFS + Streamlit

Dự án Big Data nhỏ, hoàn chỉnh từ đầu đến cuối, dùng để phân tích hành vi khách hàng thương mại điện tử. Bao gồm:

* **ETL với Apache Spark** (đọc CSV → làm sạch → ghi Parquet phân vùng theo ngày)
* **Bảng điều khiển phân tích (Analytics)**: KPI, doanh thu theo ngày, danh mục và phương thức thanh toán phổ biến
* **Phân cụm khách hàng (KMeans - RFM)** tự động chọn số cụm bằng Silhouette + bảng hồ sơ cụm
* **Lưu trữ dữ liệu trên HDFS** (hoặc chạy trên file local) và giao diện **Streamlit**

> Tài liệu này hướng dẫn **2 cách chạy dự án**: bằng **Docker (đề xuất)** hoặc **Conda (local)**. Bạn có thể chọn một trong hai cách.

---

## 0) Cấu trúc dự án

```
E:/DoAnBigData/
├─ main.py                 # Ứng dụng Streamlit (ETL + Analytics + KMeans)
├─ requirements.txt        # Danh sách thư viện pip
├─ environment.yml         # Cấu hình Conda (tùy chọn)
└─ ecommerce_data.csv      # File dữ liệu đầu vào mẫu
```

---

## A) Chạy bằng Docker (Khuyến nghị)

Yêu cầu có sẵn cụm Spark + Hadoop + container `jupyter_app` trong cùng mạng Docker.

### A1) Khởi động cụm

```bash
docker compose up -d spark-master spark-worker namenode datanode jupyter_app

docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

* **Jupyter**: [http://localhost:8888](http://localhost:8888)
* **Streamlit**: [http://localhost:8501](http://localhost:8501)

### A2) Cài đặt thư viện Python trong container `jupyter_app`

```bash
docker exec -it jupyter_app bash -lc "pip install -U pip && pip install -r /home/jovyan/work/requirements.txt"
# Nếu chưa có requirements.txt: pip install streamlit pandas matplotlib pyspark==3.3.0
```

### A3) Tải dữ liệu lên HDFS

```bash
docker exec -it jupyter_app bash -lc "\
  hdfs dfs -mkdir -p /input && \
  hdfs dfs -put -f /home/jovyan/work/ecommerce_data.csv /input/ && \
  hdfs dfs -ls /input"
```

### A4) Chạy ứng dụng Streamlit

```bash
docker exec -it jupyter_app bash -lc "cd /home/jovyan/work && python -m streamlit run main.py --server.address=0.0.0.0 --server.port=8501"
```

Sau đó mở [http://localhost:8501](http://localhost:8501)

### A5) Cấu hình sidebar (Docker mode)

* **Spark master**: `spark://spark-master:7077`
* **CSV HDFS path**: `hdfs://namenode:8020/input/ecommerce_data.csv`
* **Parquet output**: `hdfs://namenode:8020/warehouse/ecommerce_parquet`
* **Định dạng thời gian**: `M/d/yyyy H:mm` (ví dụ: `9/8/2020 9:38`)

### A6) Quy trình thao tác

1. Nhấn **Khởi tạo SparkSession** (Restart Spark nếu cần)
2. Chạy **ETL** (đọc CSV, chuẩn hóa, ghi Parquet theo ngày)
3. Chạy **Analytics** (hiển thị KPI, biểu đồ, top category/payment)
4. Chạy **Phân cụm khách hàng (KMeans)** — tự động chọn k tốt nhất, hiển thị:

   * Bảng điểm Silhouette theo k
   * Danh sách khách hàng & cụm
   * Bảng **Hồ sơ cụm (Cluster Profile)**: trung bình R/F/M, số khách hàng

---

## B) Chạy local bằng Conda (không cần Docker)

Dành cho demo đơn giản không dùng HDFS.

### B1) Tạo môi trường

```bash
conda create -n ecom-bigdata python=3.10 -y
conda activate ecom-bigdata
conda install -c conda-forge openjdk=8 -y
pip install -U pip
pip install -r requirements.txt
```

> Nếu bị lỗi Java, thêm biến môi trường trong PowerShell:

```powershell
$env:JAVA_HOME = $env:CONDA_PREFIX
$env:PATH = "$env:JAVA_HOME\Library\bin;$env:JAVA_HOME\bin;$env:PATH"
java -version
```

### B2) Chạy ứng dụng

```bash
cd E:/DoAnBigData
streamlit run main.py
```

### B3) Cấu hình sidebar (local)

* **Spark master**: `local[*]`
* **CSV path**: `file:///E:/DoAnBigData/ecommerce_data.csv`
* **Parquet output**: `file:///E:/DoAnBigData/ecommerce_parquet`
* **Định dạng thời gian**: `M/d/yyyy H:mm`

---

## C) Cài đặt Docker Desktop

Nếu bạn chưa có Docker, làm theo các bước sau để chạy được cụm Spark + HDFS + Streamlit.

### C1) Tải và cài Docker Desktop

1. Mở: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
2. Chọn đúng hệ điều hành:

   * **Windows 10/11** (64-bit) → Docker Desktop for Windows *(yêu cầu WSL2)*
   * **macOS (Intel/M1/M2)** → Docker Desktop for Mac
   * **Linux** → cài `docker` và `docker-compose` qua terminal theo distro
3. Cài xong, mở Docker Desktop và chờ biểu tượng cá voi 🐳 báo **Running**.
4. Kiểm tra trong terminal:

   ```bash
   docker --version
   docker compose version
   ```

### C2) Cấu hình WSL2 (Windows)

1. PowerShell (Run as Administrator):

   ```bash
   wsl --install
   ```

   Khởi động lại máy nếu được yêu cầu.
2. Đặt WSL2 mặc định:

   ```bash
   wsl --set-default-version 2
   ```
3. Docker Desktop → **Settings → General**: bật

   * *Use the WSL 2 based engine*
   * *Start Docker Desktop when you log in*
4. **Settings → Resources → WSL Integration**: bật tích cho distro (ví dụ **Ubuntu**).

### C3) Chạy thử container mẫu

```bash
docker run hello-world
```

Nếu in thông điệp *Hello from Docker!* → Docker đã hoạt động đúng.

### C4) Chạy cụm Hadoop + Spark + Jupyter của dự án

Tại thư mục dự án có `docker-compose.yml`:

```bash
docker compose up -d
```

Các dịch vụ sẽ khởi chạy: **namenode, datanode, spark-master, spark-worker, jupyter_app**.

---

## Tính năng chính

* Làm sạch dữ liệu, xử lý null, chuyển đổi thời gian, ghi Parquet phân vùng theo ngày.
* Thống kê & biểu đồ: doanh thu, đơn hàng, AOV, danh mục, phương thức thanh toán.
* Phân cụm khách hàng theo RFM, chọn số cụm tự động bằng Silhouette.
* Cache dữ liệu Parquet giúp chạy lại nhanh hơn.

---

## Giải thích kết quả KMeans

* **recency_days**: số ngày kể từ lần mua gần nhất → càng nhỏ càng mới.
* **frequency**: số lần mua hàng.
* **monetary**: tổng chi tiêu.
* **prediction**: số cụm (0..k-1)

Ví dụ phân loại:

| Đặc điểm                                   | Nhóm khách hàng          |
| ------------------------------------------ | ------------------------ |
| Recency thấp, Frequency cao, Monetary cao  | VIP / Trung thành        |
| Recency thấp, Frequency thấp               | Khách mới                |
| Recency cao, Frequency thấp, Monetary thấp | Không hoạt động / Rời bỏ |

---

## Khắc phục lỗi thường gặp

### 1) `UnknownHostException: namenode`

Chạy local nhưng dùng đường dẫn HDFS → hãy đổi thành `file:///...` hoặc chạy trong Docker.

### 2) `Cannot call methods on a stopped SparkContext`

Spark bị khởi động lại, cache cũ còn giữ. Nhấn **Restart SparkSession** rồi thử lại.

### 3) Cảnh báo màu vàng: lỗi parse thời gian

Nhập đúng định dạng thời gian: `M/d/yyyy H:mm`, `yyyy-MM-dd HH:mm:ss`, `dd/MM/yyyy`.

### 4) `streamlit: command not found`

Container bị reset → cài lại:

```bash
docker exec -it jupyter_app bash -lc "pip install -U pip && pip install -r /home/jovyan/work/requirements.txt"
```

---

## Yêu cầu môi trường

* **Docker mode**: Docker + Docker Compose, cụm Spark/Hadoop + `jupyter_app`
* **Local mode**: Python 3.10, PySpark 3.3.0, Java 8, Streamlit, pandas, matplotlib

### requirements.txt

```
streamlit
pyspark==3.3.0
pandas
matplotlib
```

### environment.yml

```yaml
name: ecom-bigdata
channels:
  - conda-forge
dependencies:
  - python=3.10
  - openjdk=8
  - pip
  - pip:
      - streamlit
      - pyspark==3.3.0
      - pandas
      - matplotlib
```

---

## Bản quyền

Sử dụng cho mục đích học tập và nghiên cứu.
