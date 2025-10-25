import os
import traceback
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)
from pyspark.sql.functions import (
    col, to_timestamp, date_format, coalesce, trim, regexp_replace,
    countDistinct, count, sum as _sum, avg, expr, when, max as _max
)



# =========================
# Config mặc định
# =========================
DEFAULT_SPARK_MASTER = os.getenv("SPARK_MASTER_URL", os.getenv("SPARK_MASTER", "spark://spark-master:7077"))
DEFAULT_INPUT = "hdfs://namenode:8020/input/ecommerce.csv"
DEFAULT_WAREHOUSE = os.getenv("WAREHOUSE_PATH", "hdfs://namenode:8020/warehouse")
DEFAULT_OUTPUT = os.getenv("OUTPUT_PATH", "hdfs://namenode:8020/output")
PARQUET_DIR = f"{DEFAULT_WAREHOUSE.rstrip('/')}/ecommerce_parquet"

# =========================
# Spark helpers (1 version only)
# =========================
# ---- CACHE: đọc Parquet 1 lần cho cả Analytics & ML ----
def add_churn_from_recency(df, customer_col="customer_id", ts_col="purchase_ts", cutoff_days=180):
    """
    Tạo cột churn dựa trên 'recency':
      - Tìm lần mua gần nhất của từng khách hàng.
      - Nếu số ngày từ lần mua gần nhất tới mốc thời gian max trong dataset > cutoff_days => churn=1, ngược lại 0.
    Trả về DataFrame đã có cột 'churn' (int).
    """
    mx = df.agg(_max(ts_col).alias("mx")).collect()[0]["mx"]
    if mx is None:
        # Không có timestamp hợp lệ -> không thể tính recency: set churn=0 nếu chưa có
        return df.withColumn("churn", when(col("churn").isNull(), 0).otherwise(col("churn")).cast("int"))

    last_by_cust = (
        df.groupBy(customer_col)
          .agg(_max(ts_col).alias("last_purchase"))
          .withColumn("days_since_last", expr(f"datediff(to_timestamp('{str(mx)}'), last_purchase)"))
          .withColumn("churn_calc", (col("days_since_last") > cutoff_days).cast("int"))
          .select(customer_col, "churn_calc")
    )

    df2 = (
        df.join(last_by_cust, on=customer_col, how="left")
          .withColumn("churn", coalesce(col("churn").cast("int"), col("churn_calc")).cast("int"))
          .drop("churn_calc")
    )
    return df2
@st.cache_resource(show_spinner=False)
def load_parquet_cached(app_id: str, parquet_path: str):
    """
    Cache DataFrame theo app_id hiện tại để tránh giữ DF gắn với SparkSession đã stop.
    """
    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    df = spark.read.parquet(parquet_path).cache()   # cache trên executors
    _ = df.count()                                   # materialize để lần sau rất nhanh
    return df


def _spark_is_alive(spark):
    try:
        return spark is not None and not spark.sparkContext._jsc.sc().isStopped()
    except Exception:
        return False

def _hard_reset_spark_jvm():
    """Xoá session mặc định/active và stop SparkContext còn treo trong JVM."""
    try:
        # clear default/active SparkSession (tầng JVM)
        SparkSession.clearActiveSession()
        SparkSession.clearDefaultSession()
    except Exception:
        pass
    try:
        # stop SparkContext đang active nếu còn
        from pyspark import SparkContext
        sc = SparkContext._active_spark_context
        if sc is not None:
            sc.stop()
    except Exception:
        pass

def start_spark(app_name: str = "EcomBigDataApp", master: str = DEFAULT_SPARK_MASTER, log_level: str = "WARN"):
    # Lần 1: thử tạo bình thường
    try:
        spark = (SparkSession.builder
                 .appName(app_name)
                 .master(master)
                 .config("spark.sql.shuffle.partitions", "64")
                 .config("spark.ui.showConsoleProgress", "true")
                 .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
                 .getOrCreate())
        if not _spark_is_alive(spark):
            raise RuntimeError("SparkContext stopped right after getOrCreate()")
        spark.sparkContext.setLogLevel(log_level)
        return spark
    except Exception as e1:
        print("[WARN] First attempt failed → reset JVM sessions then retry. Reason:", repr(e1))
        # DỌN SẠCH rồi thử lại với master thật
        _hard_reset_spark_jvm()
        try:
            spark = (SparkSession.builder
                     .appName(app_name)
                     .master(master)
                     .config("spark.sql.shuffle.partitions", "64")
                     .config("spark.ui.showConsoleProgress", "true")
                     .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
                     .getOrCreate())
            if not _spark_is_alive(spark):
                raise RuntimeError("SparkContext stopped again after getOrCreate()")
            spark.sparkContext.setLogLevel(log_level)
            return spark
        except Exception as e2:
            print("[WARN] Second attempt with master failed → fallback local[*]. Reason:", repr(e2))
            _hard_reset_spark_jvm()
            spark = (SparkSession.builder
                     .appName(app_name + "-local")
                     .master("local[*]")
                     .config("spark.sql.shuffle.partitions", "8")
                     .config("spark.ui.showConsoleProgress", "true")
                     .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
                     .getOrCreate())
            spark.sparkContext.setLogLevel(log_level)
            return spark
def ensure_spark(master: str, app_name: str = "EcomBigDataApp", log_level: str = "WARN"):
    spark = st.session_state.get("spark")
    if not _spark_is_alive(spark):
        _hard_reset_spark_jvm()
        spark = start_spark(app_name=app_name, master=master, log_level=log_level)
        st.session_state["spark"] = spark
    return spark


# =========================
# Schema dataset theo mô tả
# =========================
SCHEMA = StructType([
    StructField("Customer ID", StringType(), True),
    StructField("Customer Name", StringType(), True),
    StructField("Customer Age", IntegerType(), True),
    StructField("Gender", StringType(), True),
    StructField("Purchase Date", StringType(), True),          # parse sau
    StructField("Product Category", StringType(), True),
    StructField("Product Price", DoubleType(), True),
    StructField("Quantity", IntegerType(), True),
    StructField("Total Purchase Amount", DoubleType(), True),
    StructField("Payment Method", StringType(), True),
    StructField("Returns", IntegerType(), True),               # 0/1
    StructField("Churn", IntegerType(), True)                  # 0/1
])

# =========================
# ETL
# =========================
def run_etl(spark, src_path: str, dst_parquet: str, ts_format: str):
    st.info(f"Đọc CSV từ: {src_path}")

    # Đọc theo schema định nghĩa sẵn (SCHEMA / SCHEMA_WITH_CUSTOMER)
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")  
        .csv(src_path)
    )

    # --- Đổi tên cột về schema ngắn gọn/thống nhất ---
    rename_pairs = {
        "Customer ID": "customer_id",
        "Customer": "customer_id",                  # hỗ trợ header "Customer"
        "Customer Name": "customer_name",
        "Customer Age": "customer_age",
        "Gender": "gender",
        "Purchase Date": "purchase_date_raw",
        "Product Category": "product_category",
        "Product Price": "product_price",
        "Quantity": "quantity",
        "Total Purchase Amount": "total_amount",
        "Payment Method": "payment_method",
        "Returns": "returns",
        "Churn": "churn",
    }
    for old, new in rename_pairs.items():
        if old in df.columns and new not in df.columns:
            df = df.withColumnRenamed(old, new)

    # --- Làm sạch số & ép kiểu an toàn ---
    for num_col in ["product_price", "total_amount"]:
        if num_col in df.columns:
            df = df.withColumn(
                num_col,
                regexp_replace(col(num_col).cast("string"), r"[^0-9\.\-]", "").cast("double")
            )
    if "quantity" in df.columns:
        df = df.withColumn("quantity",
            regexp_replace(col("quantity").cast("string"), r"[^0-9\-]", "").cast("int")
        )
    if "returns" in df.columns:
        df = df.withColumn("returns", col("returns").cast("int"))
    if "churn" in df.columns:
        df = df.withColumn("churn", col("churn").cast("int"))

    # --- Parse thời gian đa định dạng ---
    raw = "purchase_date_raw"
    if raw in df.columns:
        df = df.withColumn(raw, trim(col(raw)))
        df = df.withColumn(raw, regexp_replace(col(raw), "T", " "))
        df = df.withColumn(raw, regexp_replace(col(raw), "Z$", ""))
        df = df.withColumn(raw, regexp_replace(col(raw), r"[\u00A0\u2007\u202F]", " "))
        df = df.withColumn(raw, regexp_replace(col(raw), r"\s+", " "))

        patterns = []
        if ts_format:
            patterns.append(ts_format)

        # Ưu tiên dữ liệu thực tế của bạn: "9/8/2020 9:38" (không giây, 1 chữ số)
        patterns += [
            "M/d/yyyy H:mm",
            "M/d/yyyy h:mm a",
            "MM/dd/yyyy HH:mm",
            "dd/M/yyyy H:mm",
            "dd/MM/yyyy HH:mm",
            "M/d/yyyy H:mm:ss",
            "MM/dd/yyyy HH:mm:ss",
            "dd/M/yyyy H:mm:ss",
            "dd/MM/yyyy HH:mm:ss",
            "yyyy-MM-dd",
            "MM/dd/yyyy",
            "dd/MM/yyyy",
        ]

        ts_exprs = [to_timestamp(col(raw), p) for p in patterns]
        df = df.withColumn("purchase_ts", coalesce(*ts_exprs))
        df = df.withColumn("purchase_date", date_format(col("purchase_ts"), "yyyy-MM-dd"))

        # Cảnh báo nếu gần như không parse được
        null_ratio = df.select((col("purchase_ts").isNull()).cast("int").alias("isnull")) \
                       .agg({"isnull": "avg"}).collect()[0][0]
        if null_ratio and null_ratio >= 0.9999:
            st.warning(
                "Không parse được cột thời gian 'Purchase Date'. "
                "Hãy nhập đúng định dạng ở ô 'Định dạng thời gian' (ví dụ: M/d/yyyy H:mm). "
                "App vẫn ghi Parquet nhưng các biểu đồ theo ngày sẽ rỗng."
            )
    else:
        df = df.withColumn("purchase_ts", None).withColumn("purchase_date", None)

    # --- Bổ khuyết churn theo recency nếu thiếu hoặc phần lớn NULL ---
    needs_fill = True
    if "churn" in df.columns:
        null_ratio = df.select((col("churn").isNull()).cast("int").alias("isnull")) \
                       .agg({"isnull": "avg"}).collect()[0][0]
        needs_fill = (null_ratio is None) or (null_ratio > 0.5)

    cutoff_days = 180  # có thể đưa ra sidebar nếu muốn điều chỉnh động
    if needs_fill:
        df = add_churn_from_recency(df, customer_col="customer_id",
                                    ts_col="purchase_ts", cutoff_days=cutoff_days)

    # --- Điền NA cho một số cột phân loại ---
    for c in ["gender", "product_category", "payment_method"]:
        if c in df.columns:
            df = df.fillna({c: "unknown"})

    # --- Ghi Parquet (overwrite), partition theo ngày nếu có ---
    writer = df.write.mode("overwrite")
    if "purchase_date" in df.columns:
        writer = writer.partitionBy("purchase_date")
    writer.parquet(dst_parquet)

    st.success(f"ETL hoàn tất → {dst_parquet}")
    return dst_parquet

# =========================
# Analytics
# =========================
def run_analytics(spark, parquet_path: str):
    st.subheader("Analytics")
    if not _spark_is_alive(spark):
        sparkContext = ensure_spark(master)  # hoặc raise lỗi tuỳ bạn đã viết
        spark = ensure_spark(master)
    app_id = spark.sparkContext.applicationId
    df = load_parquet_cached(app_id, parquet_path)

    # Tổng quan
    total_rows = df.count()
    n_cols = len(df.columns)
    st.write(f"**Số dòng:** {total_rows:,} | **Số cột:** {n_cols}")
    with st.expander("Danh sách cột"):
        st.code(", ".join(df.columns))

    # —— KPI ——
    st.markdown("### KPIs tổng quan")
    kpi = {}
    if "customer_id" in df.columns:
        kpi["Số khách hàng"] = df.select(countDistinct("customer_id").alias("n")).collect()[0]["n"]
    if "product_category" in df.columns:
        kpi["Số category"] = df.select(countDistinct("product_category").alias("n")).collect()[0]["n"]
    if "total_amount" in df.columns:
        kpi["Tổng doanh thu"] = df.agg(_sum("total_amount").alias("rev")).collect()[0]["rev"]
    if "returns" in df.columns:
        kpi["Tỉ lệ trả hàng (%)"] = round((df.agg(avg(col("returns").cast("double")).alias("r"))
                                             .collect()[0]["r"] or 0) * 100, 2)
    if "churn" in df.columns:
        kpi["Tỉ lệ churn (%)"] = round((df.agg(avg(col("churn").cast("double")).alias("c"))
                                         .collect()[0]["c"] or 0) * 100, 2)

    if kpi:
        for k, v in kpi.items():
            st.write(f"- **{k}:** {v}")
    else:
        st.write("Không đủ cột để tính KPI.")

    # —— Doanh thu theo ngày (Biểu đồ tương tác của Streamlit) ——
    if "purchase_date" in df.columns and "total_amount" in df.columns:
        st.markdown("### Doanh thu & Đơn hàng theo ngày")

        daily = (
            df.groupBy("purchase_date")
              .agg(_sum("total_amount").alias("revenue"),
                   count("*").alias("orders"))
              .orderBy("purchase_date")
        )
        pdf = daily.toPandas()

        if pdf.empty or pdf["purchase_date"].isna().all():
            st.warning("Không có dữ liệu ngày (có thể do không parse được 'Purchase Date').")
        else:
            # Chuẩn hoá thời gian & sort
            pdf["purchase_date"] = pd.to_datetime(pdf["purchase_date"], errors="coerce")
            pdf = pdf.dropna(subset=["purchase_date"]).sort_values("purchase_date")
            
            if pdf.empty:
                st.warning("Không có bản ghi hợp lệ sau khi chuẩn hoá thời gian.")
            else:
                # Tính đường trung bình trượt cho Revenue
                pdf["rev_ma7"] = pdf["revenue"].rolling(7, min_periods=1).mean()
                
                # Đặt 'purchase_date' làm index để Streamlit vẽ biểu đồ
                pdf_indexed = pdf.set_index("purchase_date")

                # --- Biểu đồ 1: Doanh thu (Revenue) ---
                st.markdown("#### Xu hướng doanh thu (Revenue & MA-7)")
                st.write("Biểu đồ đường thể hiện doanh thu hàng ngày và đường trung bình trượt 7 ngày (MA-7).")
                # st.line_chart sẽ tự động vẽ 2 cột 'revenue' và 'rev_ma7'
                st.line_chart(pdf_indexed[["revenue", "rev_ma7"]])

                # --- Biểu đồ 2: Số lượng đơn hàng (Orders) ---
                st.markdown("#### Số lượng đơn hàng hàng ngày")
                st.write("Biểu đồ cột thể hiện tổng số đơn hàng mỗi ngày.")
                st.bar_chart(pdf_indexed[["orders"]])


    # —— Top Category theo số đơn & doanh thu ——
    if "product_category" in df.columns:
        st.markdown("### Top Category")
        top_n = 10
        top_cat = (df.groupBy("product_category")
                     .agg(count("*").alias("cnt"), _sum("total_amount").alias("revenue"))
                     .orderBy(col("cnt").desc())
                     .limit(top_n))
        pdf = top_cat.toPandas()
        st.dataframe(pdf, use_container_width=True)

        if not pdf.empty:
            # theo số đơn (barh, dễ đọc label dài)
            pdf_cnt = pdf.sort_values("cnt", ascending=True)
            fig4, ax4 = plt.subplots()
            ax4.barh(pdf_cnt["product_category"].astype(str), pdf_cnt["cnt"])
            ax4.set_title(f"Top {top_n} Category (theo số đơn)")
            ax4.set_xlabel("Count"); ax4.set_ylabel("Category")
            st.pyplot(fig4)

            # theo doanh thu (barh)
            pdf_rev = pdf.sort_values("revenue", ascending=True)
            fig5, ax5 = plt.subplots()
            ax5.barh(pdf_rev["product_category"].astype(str), pdf_rev["revenue"])
            ax5.set_title(f"Top {top_n} Category (theo doanh thu)")
            ax5.set_xlabel("Revenue"); ax5.set_ylabel("Category")
            st.pyplot(fig5)


    # —— Phương thức thanh toán ——
    if "payment_method" in df.columns:
        st.markdown("### Phân bố phương thức thanh toán")
        pm = df.groupBy("payment_method").agg(count("*").alias("cnt"),
                                              _sum("total_amount").alias("revenue")) \
               .orderBy(col("cnt").desc())
        pdf = pm.toPandas()
        st.dataframe(pdf, use_container_width=True)

        if not pdf.empty:
            fig6, ax6 = plt.subplots()
            ax6.bar(pdf["payment_method"].astype(str), pdf["cnt"])
            ax6.set_title("Số đơn theo phương thức thanh toán")
            ax6.set_xlabel("Payment Method"); ax6.set_ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig6)

            fig7, ax7 = plt.subplots()
            ax7.bar(pdf["payment_method"].astype(str), pdf["revenue"])
            ax7.set_title("Doanh thu theo phương thức thanh toán")
            ax7.set_xlabel("Payment Method"); ax7.set_ylabel("Revenue")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig7)

    # —— Returns & Churn theo ngày (nếu có cột) ——
    if "purchase_date" in df.columns and "returns" in df.columns:
        st.markdown("### Tỉ lệ trả hàng theo ngày")
        rate = (df.groupBy("purchase_date")
                  .agg(avg(col("returns").cast("double")).alias("return_rate"))
                  .orderBy("purchase_date")).toPandas()
        
        if not rate.empty and rate["purchase_date"].notna().any():
            # Chuẩn hoá thời gian & set index
            rate["purchase_date"] = pd.to_datetime(rate["purchase_date"], errors="coerce")
            rate = rate.dropna(subset=["purchase_date"]).set_index("purchase_date")
            
            if not rate.empty:
                st.write("Biểu đồ đường thể hiện tỉ lệ trả hàng (return rate) theo thời gian.")
                st.line_chart(rate[["return_rate"]])
            else:
                st.warning("Không có dữ liệu tỉ lệ trả hàng hợp lệ sau khi chuẩn hoá thời gian.")
        else:
            st.warning("Không có dữ liệu để vẽ biểu đồ tỉ lệ trả hàng.")

    if "churn" in df.columns and "gender" in df.columns:
        st.markdown("### Tỉ lệ churn theo giới tính")
        churn_g = (df.groupBy("gender")
                     .agg(avg(col("churn").cast("double")).alias("churn_rate"))
                     .orderBy("gender")).toPandas()
        if not churn_g.empty:
            fig9, ax9 = plt.subplots()
            ax9.bar(churn_g["gender"].astype(str), churn_g["churn_rate"])
            ax9.set_title("Churn rate theo giới tính")
            ax9.set_xlabel("Gender"); ax9.set_ylabel("Churn rate")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig9)

    # —— Phân bố tuổi (nếu có) ——
    if "customer_age" in df.columns:
        st.markdown("### Phân bố độ tuổi khách hàng")
        age_pdf = df.select("customer_age").toPandas().dropna()
        if not age_pdf.empty:
            data = age_pdf["customer_age"].astype(float)
            # Lọc ra các giá trị tuổi hợp lệ (ví dụ: 0-100) để tránh làm hỏng biểu đồ
            data = data[(data > 0) & (data < 100)] 
            
            if len(data) > 0:
                # Tính số bins (nhóm)
                bins_count = int(np.clip(np.sqrt(len(data)), 10, 40)) # sqrt(n) trong [10..40]
                
                # Tính toán giá trị histogram (tần suất và các cạnh của bin)
                hist, bin_edges = np.histogram(data, bins=bins_count)
                
                # Tạo DataFrame cho st.bar_chart
                # Lấy nhãn là điểm giữa của mỗi bin để biểu đồ hiển thị đúng
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                hist_df = pd.DataFrame({
                    'age_center': bin_centers, # Trục X: Tuổi (trung tâm nhóm)
                    'count': hist              # Trục Y: Số lượng khách hàng
                }).set_index('age_center')
                
                st.write(f"Biểu đồ cột thể hiện tần suất khách hàng theo các nhóm tuổi (chia thành {bins_count} nhóm).")
                st.bar_chart(hist_df[['count']])
            else:
                st.warning("Không có dữ liệu tuổi hợp lệ (0-100) để vẽ biểu đồ phân bố.")
        else:
            st.warning("Không có dữ liệu tuổi để vẽ biểu đồ phân bố.")


# ==================================================
# 🧩 PHÂN TÍCH XU HƯỚNG MUA SẮM (Trend Analysis)
# ==================================================
def run_trend_analysis(df):
    st.subheader("📈 Phân tích xu hướng mua sắm")

        # Doanh thu theo tháng
    monthly_rev = (df.withColumn("month", date_format("purchase_ts", "yyyy-MM"))
                     .groupBy("month")
                     .agg(_sum("total_amount").alias("revenue"))
                     .orderBy("month"))

    pandas_monthly = monthly_rev.toPandas()

    if not pandas_monthly.empty:
        import matplotlib.dates as mdates

        # Chuẩn hoá mốc tháng, điền tháng trống = 0
        pandas_monthly["month"] = pd.to_datetime(pandas_monthly["month"], format="%Y-%m", errors="coerce")
        pandas_monthly = pandas_monthly.dropna(subset=["month"]).sort_values("month")

        if not pandas_monthly.empty:
            # Reindex theo dải tháng đầy đủ
            full_range = pd.period_range(
                pandas_monthly["month"].min().to_period("M"),
                pandas_monthly["month"].max().to_period("M"),
                freq="M"
            ).to_timestamp()
            pm = pandas_monthly.set_index("month").reindex(full_range, fill_value=0).rename_axis("month").reset_index()

            fig, ax = plt.subplots()
            ax.plot(pm["month"], pm["revenue"], linewidth=2)
            locator = mdates.AutoDateLocator(minticks=4, maxticks=12)
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            ax.set_title("Doanh thu theo tháng")
            ax.set_xlabel("Tháng"); ax.set_ylabel("Revenue")
            st.pyplot(fig)
    else:
        st.warning("Không có dữ liệu doanh thu theo tháng.")

    # Top 5 danh mục sản phẩm
    top_cat = (df.groupBy("product_category")
                 .agg(_sum("total_amount").alias("revenue"))
                 .orderBy(F.desc("revenue"))
                 .limit(5))

    pandas_cat = top_cat.toPandas()
    if not pandas_cat.empty:
        st.markdown("**Top 5 danh mục sản phẩm có doanh thu cao nhất**")
        st.bar_chart(pandas_cat.set_index("product_category")["revenue"])

        # Phương thức thanh toán phổ biến (bar chart thay vì pie)
    pay_method = (df.groupBy("payment_method")
                    .agg(count("*").alias("count"))
                    .orderBy(F.desc("count")))
    pandas_pay = pay_method.toPandas()
    if not pandas_pay.empty:
        st.markdown("**Phân bố phương thức thanh toán**")
        fig, ax = plt.subplots()
        ax.bar(pandas_pay["payment_method"].astype(str), pandas_pay["count"])
        ax.set_title("Số đơn theo phương thức thanh toán")
        ax.set_xlabel("Payment Method"); ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)



# ==================================================
# 🧠 PHÂN TÍCH HÀNH VI NGƯỜI DÙNG (Customer Behavior)
# ==================================================
def run_customer_behavior(df):
    st.subheader("🧠 Phân tích hành vi người dùng (RFM)")

    # Chuẩn bị dữ liệu
    rfm_df = (df.groupBy("customer_id")
                .agg(
                    F.max("purchase_ts").alias("last_purchase"),
                    count("*").alias("frequency"),
                    _sum("total_amount").alias("monetary")
                ))
    
    max_date_row = df.agg(F.max("purchase_ts")).collect()
    if not max_date_row or max_date_row[0][0] is None:
        st.error("Không thể tính toán RFM do thiếu mốc thời gian (max_date). Kiểm tra lại dữ liệu 'Purchase Date'.")
        return
    max_date = max_date_row[0][0]
        
    rfm_df = rfm_df.withColumn(
        "recency_days", F.datediff(F.lit(max_date), F.col("last_purchase"))
    )

    rfm_df = rfm_df.select("customer_id", "recency_days", "frequency", "monetary")
    pandas_rfm = rfm_df.toPandas()
    if pandas_rfm.empty:
        st.warning("Không có dữ liệu hành vi người dùng để phân tích.")
        return

    # Biểu đồ phân tán R-F-M
    st.markdown("**Phân bố RFM của khách hàng (Tổng quan)**")
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        pandas_rfm["recency_days"],
        pandas_rfm["frequency"],
        c=pandas_rfm["monetary"],
        cmap="viridis",
        alpha=0.7,
    )
    plt.xlabel("Recency (days)")
    plt.ylabel("Frequency")
    plt.title("Hành vi mua sắm khách hàng (R vs F, màu là M)")
    plt.colorbar(sc, label="Monetary (Tổng chi tiêu)")
    st.pyplot(fig)

    # --- Thêm biểu đồ phân bố chi tiết ---
    st.markdown("**Phân bố chi tiết của Recency, Frequency, và Monetary**")
    
    # Chia layout thành 3 cột để hiển thị biểu đồ
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Recency (Số ngày từ lần mua cuối)")
        fig_r, ax_r = plt.subplots()
        ax_r.hist(pandas_rfm['recency_days'].dropna(), bins=30, color='skyblue', edgecolor='black')
        ax_r.set_title("Phân bố Recency")
        ax_r.set_xlabel("Số ngày")
        ax_r.set_ylabel("Số lượng khách hàng")
        st.pyplot(fig_r)

    with col2:
        st.markdown("#### Frequency (Tổng số lần mua)")
        # Lấy dữ liệu F, lọc các giá trị ngoại lệ (ví dụ: > 99th percentile) để biểu đồ dễ nhìn hơn
        f_data = pandas_rfm['frequency'].dropna()
        if not f_data.empty:
            f_q99 = f_data.quantile(0.99)
            f_data_clipped = f_data[f_data <= f_q99]
            
            fig_f, ax_f = plt.subplots()
            ax_f.hist(f_data_clipped, bins=30, color='lightgreen', edgecolor='black')
            ax_f.set_title(f"Phân bố Frequency (lọc giá trị > {f_q99:.0f})")
            ax_f.set_xlabel("Số lần mua")
            ax_f.set_ylabel("Số lượng khách hàng")
            st.pyplot(fig_f)
        else:
            st.write("Không có dữ liệu Frequency.")

    with col3:
        st.markdown("#### Monetary (Tổng chi tiêu)")
        # Lọc giá trị ngoại lệ tương tự Frequency
        m_data = pandas_rfm['monetary'].dropna()
        if not m_data.empty:
            m_q99 = m_data.quantile(0.99)
            m_data_clipped = m_data[m_data <= m_q99]
            
            fig_m, ax_m = plt.subplots()
            ax_m.hist(m_data_clipped, bins=30, color='salmon', edgecolor='black')
            ax_m.set_title(f"Phân bố Monetary (lọc giá trị > {m_q99:,.0f})")
            ax_m.set_xlabel("Tổng chi tiêu")
            ax_m.set_ylabel("Số lượng khách hàng")
            st.pyplot(fig_m)
        else:
            st.write("Không có dữ liệu Monetary.")

    # Bảng thống kê trung bình
    st.markdown("**Thống kê trung bình RFM**")
    st.dataframe(
        pandas_rfm[["recency_days", "frequency", "monetary"]]
        .describe()
        .T
        .rename(columns={"mean": "Giá trị trung bình"})
    )


# =========================
# ML: KMeans Customer Segmentation (RFM nhẹ)
# =========================
def _interpret_cluster(r, f, m, g_r, g_f, g_m):
    """Hàm trợ giúp để tự động diễn giải ý nghĩa cụm."""
    
    # So sánh R, F, M của cụm với trung bình chung
    # (Tốt/Xấu/TB)
    r_score = "Thấp (Tốt)" if r < g_r * 0.9 else "Cao (Xấu)" if r > g_r * 1.1 else "Trung bình"
    f_score = "Cao (Tốt)" if f > g_f * 1.1 else "Thấp (Xấu)" if f < g_f * 0.9 else "Trung bình"
    m_score = "Cao (Tốt)" if m > g_m * 1.1 else "Thấp (Xấu)" if m < g_m * 0.9 else "Trung bình"

    # Diễn giải logic
    if r_score == "Thấp (Tốt)" and f_score == "Cao (Tốt)" and m_score == "Cao (Tốt)":
        return "🌟 Khách hàng VIP/Trung thành"
    elif r_score == "Cao (Xấu)" and f_score == "Thấp (Xấu)":
        return "⚠️ Khách hàng có nguy cơ rời bỏ"
    elif r_score == "Thấp (Tốt)" and f_score == "Thấp (Xấu)":
        return "💡 Khách hàng mới"
    elif f_score == "Cao (Tốt)":
        return "💖 Khách hàng thân thiết"
    elif m_score == "Cao (Tốt)":
        return "💰 Khách hàng chi tiêu cao"
    elif r_score == "Cao (Xấu)":
        return "💤 Khách hàng ngủ đông"
    
    return "Khách hàng Tiềm năng/Trung bình"

def run_kmeans_segmentation(spark, parquet_path: str):
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.evaluation import ClusteringEvaluator
    from pyspark.ml import Pipeline

    st.subheader("Phân cụm khách hàng (KMeans, dựa trên RFM đơn giản)")

    # Đảm bảo Spark sống & đọc dữ liệu
    if not _spark_is_alive(spark):
        spark = ensure_spark(master)
    app_id = spark.sparkContext.applicationId
    try:
        df = load_parquet_cached(app_id, parquet_path)
    except Exception:
        df = spark.read.parquet(parquet_path)

    # Kiểm tra cột cần thiết
    required = ["customer_id", "purchase_ts", "total_amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Thiếu cột cho RFM: {missing}")
        return

    # Loại bản ghi thiếu timestamp
    df_nonull = df.na.drop(subset=["purchase_ts"])
    if df_nonull.count() == 0:
        st.error("Không có 'purchase_ts' hợp lệ. Kiểm tra định dạng thời gian khi ETL.")
        return

    # Tính RFM
    rfm = (df_nonull.groupBy("customer_id")
           .agg(_max("purchase_ts").alias("last_purchase"),
                count("*").alias("frequency"),
                _sum("total_amount").alias("monetary")))

    max_ts_row = df_nonull.agg(_max("purchase_ts").alias("mx")).collect()
    if not max_ts_row or max_ts_row[0]["mx"] is None:
        st.error("Không tính được mốc thời gian lớn nhất (max_ts).")
        return
    max_ts = max_ts_row[0]["mx"]

    rfm = rfm.withColumn("recency_days", expr(f"datediff(to_timestamp('{str(max_ts)}'), last_purchase)"))
    rfm_clean = rfm.na.drop(subset=["recency_days", "frequency", "monetary"])
    
    if rfm_clean.count() == 0:
        st.error("Không đủ dữ liệu sau khi làm sạch để phân cụm.")
        return
        
    # Lấy trung bình toàn cục để so sánh
    try:
        global_avgs = rfm_clean.agg(
            avg("recency_days"), avg("frequency"), avg("monetary")
        ).collect()[0]
        global_r, global_f, global_m = global_avgs[0], global_avgs[1], global_avgs[2]
        if global_r is None: global_r = 0
        if global_f is None: global_f = 0
        if global_m is None: global_m = 0
    except Exception:
        global_r, global_f, global_m = 0, 0, 0


    # Pipeline tiền xử lý
    features = ["recency_days", "frequency", "monetary"]
    assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False)

    # ---- TỰ ĐỘNG CHỌN k THEO SILHOUETTE ----
    evaluator = ClusteringEvaluator(featuresCol="features")
    ks = list(range(2, 9))  # dải k thử (2..8)
    scores = []

    # (Tăng tốc nhẹ với sample nếu quá lớn)
    train_df = rfm_clean
    approx = rfm_clean.count()
    if approx > 200_000:
        train_df = rfm_clean.sample(False, 200_000 / approx, seed=42)

    best_model = None
    best_k = None
    best_score = float("-inf")

    st.markdown("#### 1. Chọn số cụm (k) tự động theo Silhouette")
    progress_bar = st.progress(0, text="Đang tìm k tốt nhất...")

    for i, k in enumerate(ks):
        model = Pipeline(stages=[assembler, scaler, KMeans(featuresCol="features", k=k, seed=42)]).fit(train_df)
        pred_k = model.transform(train_df)
        sil = evaluator.evaluate(pred_k)
        scores.append((k, sil))
        if sil > best_score:
            best_score, best_k, best_model = sil, k, model
        progress_bar.progress((i + 1) / len(ks), text=f"Đã thử k={k} (Silhouette: {sil:.4f})")
    
    progress_bar.empty()

    # **CẢI TIẾN 1: Trực quan hóa chọn k**
    scores_df = pd.DataFrame(scores, columns=["k", "silhouette"]).set_index("k")
    st.line_chart(scores_df)
    st.success(f"Số cụm được chọn: **k = {best_k}** (Silhouette cao nhất = **{best_score:.4f}**)")

    # Dùng model tốt nhất để gán cụm cho TOÀN BỘ dữ liệu rfm_clean
    res = best_model.transform(rfm_clean)

    # Lấy profile cụm
    prof_spark = (res.groupBy("prediction")
              .agg(avg("recency_days").alias("avg_recency"),
                   avg("frequency").alias("avg_freq"),
                   avg("monetary").alias("avg_monetary"),
                   count("*").alias("n_customers"))
              .orderBy("prediction"))
    prof_pd = prof_spark.toPandas()


    # **CẢI TIẾN 2: Dùng Tab để tổ chức kết quả**
    st.markdown("---")
    st.markdown("#### 2. Kết quả phân cụm")
    
    tab_profile, tab_viz, tab_data = st.tabs([
        "📊 Profile Cụm (Họ là ai?)", 
        "📈 Trực quan hóa Cụm (Họ ở đâu?)", 
        "📋 Dữ liệu chi tiết"
    ])

    # **CẢI TIẾN 3: Dùng st.metric và diễn giải cụm**
    with tab_profile:
        st.subheader("Phân tích Profile từng cụm")
        st.write("""
        Dưới đây là đặc điểm trung bình của khách hàng trong từng cụm. 
        Tên cụm (ví dụ: "Khách hàng VIP") được tự động gợi ý dựa trên việc so sánh với mức trung bình chung.
        """)
        
        for idx, row in prof_pd.iterrows():
            cluster_id = row["prediction"]
            # Đổi tên biến 'count' thành 'n_customers' để tránh xung đột
            avg_r, avg_f, avg_m, n_customers = row["avg_recency"], row["avg_freq"], row["avg_monetary"], row["n_customers"]
            
            # Tự động diễn giải
            title = _interpret_cluster(avg_r, avg_f, avg_m, global_r, global_f, global_m)
            
            st.markdown(f"### Cụm {cluster_id}: {title}")
            
            # Dùng st.metric để hiển thị đẹp mắt
            c1, c2, c3, c4 = st.columns(4)
            # Sử dụng biến 'n_customers' mới
            c1.metric("Số lượng KH", f"{n_customers:,.0f} KH")
            c2.metric("Recency (TB)", f"{avg_r:,.1f} ngày", 
                      f"{avg_r - global_r:,.1f} vs TB", help=f"Trung bình chung: {global_r:,.1f} ngày")
            c3.metric("Frequency (TB)", f"{avg_f:,.1f} lần", 
                      f"{avg_f - global_f:,.1f} vs TB", help=f"Trung bình chung: {global_f:,.1f} lần")
            c4.metric("Monetary (TB)", f"{avg_m:,.0f}", 
                      f"{avg_m - global_m:,.0f} vs TB", help=f"Trung bình chung: {global_m:,.0f}")
            
            st.divider() # Ngăn cách giữa các cụm

    # **CẢI TIẾN 4: Biểu đồ Scatter Plot R-F-M**
    with tab_viz:
        st.subheader("Trực quan hóa các cụm (R-F-M)")
        st.write("""
        Biểu đồ này thể hiện vị trí của các khách hàng:
        - **Trục X (Recency):** Càng về bên trái càng tốt (mới mua gần đây).
        - **Trục Y (Frequency):** Càng lên cao càng tốt (mua nhiều lần).
        - **Kích thước (Size):** Càng lớn càng tốt (chi tiêu nhiều).
        - **Màu sắc (Color):** Cụm được gán.
        """)
        
        # ======================================================
        # START: SỬA LỖI
        # ======================================================
        # Lấy mẫu dữ liệu để vẽ (tránh crash trình duyệt nếu có > 10k điểm)
        viz_limit = 5000
        
        # CHỌN CÁC CỘT CẦN THIẾT TRƯỚC KHI .toPandas()
        # Điều này sẽ bỏ qua cột timestamp 'last_purchase' và các cột vector
        res_selected = res.select("recency_days", "frequency", "monetary", "prediction")
        
        total_res = res_selected.count() # Count from the selected DF
        if total_res > viz_limit:
            fraction = viz_limit / total_res
            # Now, sample and convert the *selected* DataFrame
            viz_df = res_selected.sample(False, fraction, seed=42).limit(viz_limit).toPandas() 
        else:
            # Convert the *selected* DataFrame
            viz_df = res_selected.toPandas()
        # ======================================================
        # END: SỬA LỖI
        # ======================================================

        # Chuyển prediction sang string để Streamlit hiểu là
        # cột phân loại (categorical) cho màu sắc
        viz_df["prediction"] = viz_df["prediction"].astype(str)
        
        if not viz_df.empty:
            st.scatter_chart(
                viz_df,
                x="recency_days",
                y="frequency",
                size="monetary",
                color="prediction",
                use_container_width=True
            )
        else:
            st.warning("Không có đủ dữ liệu để vẽ biểu đồ scatter.")


    # Tab cuối cùng: Dữ liệu thô và xuất file
    with tab_data:
        st.subheader("Dữ liệu khách hàng chi tiết (200 mẫu)")
        # Bảng kết quả từng khách
        st.dataframe(
            res.select("customer_id", "recency_days", "frequency", "monetary", "prediction")
               .orderBy("prediction", "customer_id")
               .limit(200)
               .toPandas(),
            use_container_width=True
        )

       

# =========================
# ML: Product Recommender using Clustering
# =========================
def build_product_clustering_model(spark, parquet_path: str):
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.evaluation import ClusteringEvaluator
    from pyspark.ml import Pipeline

    # Đảm bảo Spark sống & đọc dữ liệu
    if not _spark_is_alive(spark):
        spark = ensure_spark(master)
    app_id = spark.sparkContext.applicationId
    try:
        df = load_parquet_cached(app_id, parquet_path)
    except Exception:
        df = spark.read.parquet(parquet_path)

    # Kiểm tra cột cần thiết
    required = ["product_category", "product_price", "quantity", "total_amount", "returns"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Thiếu cột cho Product Clustering: {missing}")
        return

    # Tính features cho từng unique product_category (đại diện cho sản phẩm)
    # Features: avg_price, avg_quantity, total_revenue, avg_returns_rate
    prod_features = (df.groupBy("product_category")
                     .agg(
                         avg("product_price").alias("avg_price"),
                         avg("quantity").alias("avg_quantity"),
                         _sum("total_amount").alias("total_revenue"),
                         avg(col("returns").cast("double")).alias("avg_returns_rate")
                     ))

    prod_features_clean = prod_features.na.drop()
    if prod_features_clean.count() == 0:
        st.error("Không đủ dữ liệu sản phẩm sau khi làm sạch để phân cụm.")
        return

    # Pipeline tiền xử lý
    features = ["avg_price", "avg_quantity", "total_revenue", "avg_returns_rate"]
    assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)

    # Chọn k tự động (2-6 vì số category thường ít)
    evaluator = ClusteringEvaluator(featuresCol="features")
    ks = list(range(10,20))
    scores = []

    train_df = prod_features_clean  # Số lượng ít nên không cần sample

    best_model = None
    best_k = None
    best_score = float("-inf")

    for k in ks:
        model = Pipeline(stages=[assembler, scaler, KMeans(featuresCol="features", k=k, seed=42)]).fit(train_df)
        pred_k = model.transform(train_df)
        sil = evaluator.evaluate(pred_k)
        scores.append((k, sil))
        if sil > best_score:
            best_score, best_k, best_model = sil, k, model

    # Hiển thị Silhouette scores
    st.markdown("#### Chọn số cụm tự động theo Silhouette (cho sản phẩm)")
    st.dataframe(
        pd.DataFrame(scores, columns=["k", "silhouette"]).sort_values("k"),
        use_container_width=True
    )
    st.success(f"Số cụm sản phẩm được chọn: **k = {best_k}** (Silhouette = **{best_score:.4f}**)")

    # Áp dụng model cho toàn bộ
    clustered_prods = best_model.transform(prod_features_clean)

    # Lưu cluster vào session state để dùng cho UI
    st.session_state["clustered_prods"] = clustered_prods
    st.session_state["best_k"] = best_k
    st.session_state["product_model_built"] = True

    st.info("Model phân cụm sản phẩm đã được xây dựng. Bây giờ bạn có thể chọn sản phẩm để gợi ý bên dưới.")

    # THÊM: Hiển thị nội dung từng cụm
    st.markdown("### Nội dung từng cụm sản phẩm")
    from pyspark.sql.functions import collect_list, size

    # Group by cluster và collect list sản phẩm
    cluster_contents = (clustered_prods
                        .groupBy("prediction")
                        .agg(
                            collect_list("product_category").alias("products"),
                            count("*").alias("num_products")
                        )
                        .orderBy("prediction"))

    cluster_pdf = cluster_contents.toPandas()

    for idx, row in cluster_pdf.iterrows():
        cluster_id = row["prediction"]
        products = row["products"]
        num = row["num_products"]
        st.markdown(f"**Cụm {cluster_id}: {num} sản phẩm**")
        st.write(", ".join(products[:20]))  # Hiển thị tối đa 20 sản phẩm đầu để tránh dài dòng
        if len(products) > 20:
            st.write(f"... và {len(products) - 20} sản phẩm khác")
        st.markdown("---")

# =========================
# UI
# =========================
st.set_page_config(page_title="E-commerce Big Data App", layout="wide")
st.title("E-commerce Big Data App — Spark ETL • Analytics • ML")

with st.sidebar:
    st.header("Kết nối")
    master = st.text_input("Spark master", value=DEFAULT_SPARK_MASTER)
    log_level = st.selectbox("Spark log level", ["ERROR","WARN","INFO","DEBUG"], index=1)
    input_path = st.text_input("CSV HDFS path", value=DEFAULT_INPUT)
    parquet_out = st.text_input("Parquet output", value=PARQUET_DIR)
    ts_format = st.text_input("Định dạng thời gian (Purchase Date)", value="M/d/yyyy H:mm")
    # Gợi ý: dùng 'yyyy-MM-dd HH:mm:ss' nếu file có cả giờ-phút-giây

    if st.button("Khởi tạo Spark"):
        try:
            st.session_state["spark"] = start_spark(master=master, log_level=log_level)
            st.success("SparkSession đã sẵn sàng.")
        except Exception as e:
            st.error(f"Khởi tạo Spark thất bại: {e}")
            st.code(traceback.format_exc())

    if st.button("Restart SparkSession"):
        try:
            old = st.session_state.get("spark")
            if old is not None:
                try: old.stop()
                except Exception: pass
            _hard_reset_spark_jvm()
            # Xoá cache resource gắn với appId cũ (nếu có)
            try: st.cache_resource.clear()
            except Exception: pass
            st.session_state["spark"] = start_spark(master=master, log_level=log_level)
            st.success("Đã restart SparkSession.")
        except Exception as e:
            st.error(f"Restart Spark thất bại: {e}")
            st.code(traceback.format_exc())


spark = st.session_state.get("spark")
if spark is None:
    st.warning("Hãy khởi tạo Spark ở thanh bên.")
    st.stop()

st.markdown("### 1) ETL")
if st.button("Chạy ETL từ CSV → Parquet"):
    try:
        spark = ensure_spark(master, log_level=log_level)
        out = run_etl(spark, src_path=input_path, dst_parquet=parquet_out, ts_format=ts_format.strip())
        st.session_state["parquet_path"] = out
    except Exception as e:
        st.error(f"Lỗi ETL: {e}")
        st.code(traceback.format_exc())

st.markdown("### 2) Analytics")
ppath = st.text_input("Parquet đã ETL", value=st.session_state.get("parquet_path", PARQUET_DIR))
if st.button("Chạy Analytics"):
    try:
        spark = ensure_spark(master, log_level=log_level)
        run_analytics(spark, ppath)
    except Exception as e:
        st.error(f"Lỗi Analytics: {e}")
        st.code(traceback.format_exc())

st.markdown("### 3) Phân tích nâng cao")
if st.button("📊 Phân tích xu hướng mua sắm"):
    try:
        spark = start_spark(log_level=log_level)
        df = spark.read.parquet(ppath)
        run_trend_analysis(df)
    except Exception as e:
        st.error(f"Lỗi khi phân tích xu hướng: {e}")
        st.code(traceback.format_exc())

if st.button("🧠 Phân tích hành vi người dùng"):
    try:
        spark = start_spark(log_level=log_level)
        df = spark.read.parquet(ppath)
        run_customer_behavior(df)
    except Exception as e:
        st.error(f"Lỗi khi phân tích hành vi người dùng: {e}")
        st.code(traceback.format_exc())

st.markdown("### 4) Machine Learning")

if st.button("Phân cụm khách hàng (KMeans)"):
    try:
        spark = ensure_spark(master, log_level=log_level)
        run_kmeans_segmentation(spark, ppath)
    except Exception as e:
        st.error(f"Lỗi ML (KMeans): {e}")
        st.code(traceback.format_exc())

# Phần gợi ý sản phẩm: Tách build model và UI
st.markdown("### 🛒 Gợi ý sản phẩm dựa trên Clustering")
if st.button("Xây dựng model phân cụm sản phẩm"):
    try:
        spark = ensure_spark(master, log_level=log_level)
        build_product_clustering_model(spark, ppath)
    except Exception as e:
        st.error(f"Lỗi xây dựng model: {e}")
        st.code(traceback.format_exc())

# UI chọn và gợi ý: Luôn hiển thị nếu model đã build
if st.session_state.get("product_model_built", False):
    st.subheader("🛒 Gợi ý sản phẩm dựa trên Clustering (theo Product Category)")

    # Lấy unique categories từ data (cần đọc data để có list)
    try:
        spark_temp = ensure_spark(master, log_level=log_level)
        df_temp = spark_temp.read.parquet(ppath)
        unique_cats = [row["product_category"] for row in df_temp.select("product_category").distinct().collect()]
    except Exception:
        unique_cats = []
        st.warning("Không thể tải danh sách sản phẩm. Hãy chạy ETL trước.")

    if unique_cats:
        selected_category = st.selectbox("Chọn Product Category (sản phẩm đại diện):", unique_cats, key="select_product")

        if st.button("Tìm gợi ý sản phẩm tương tự", key="find_recommend"):
            # Tìm cluster của sản phẩm được chọn
            selected_cluster = (st.session_state["clustered_prods"]
                                .filter(col("product_category") == selected_category)
                                .select("prediction").collect()[0]["prediction"])

            # Lấy top 5 sản phẩm khác trong cùng cluster (theo total_revenue)
            recommendations = (st.session_state["clustered_prods"]
                               .filter(col("prediction") == selected_cluster)
                               .filter(col("product_category") != selected_category)
                               .select("product_category", "avg_price", "avg_quantity", "total_revenue", "avg_returns_rate")
                               .orderBy(F.desc("total_revenue"))
                               .limit(5))

            if recommendations.count() > 0:
                rec_pdf = recommendations.toPandas()
                st.markdown(f"**Gợi ý cho '{selected_category}' (Cụm {selected_cluster})**")
                st.dataframe(
                    rec_pdf,
                    use_container_width=True,
                    column_config={
                        "product_category": "Sản phẩm gợi ý",
                        "avg_price": "Giá TB",
                        "avg_quantity": "Số lượng TB",
                        "total_revenue": "Doanh thu tổng",
                        "avg_returns_rate": "Tỉ lệ trả TB"
                    }
                )
            else:
                st.warning("Không có sản phẩm tương tự trong cụm này.")
    else:
        st.warning("Không có dữ liệu sản phẩm để chọn.")
else:
    st.info("Nhấn 'Xây dựng model phân cụm sản phẩm' để bắt đầu.")