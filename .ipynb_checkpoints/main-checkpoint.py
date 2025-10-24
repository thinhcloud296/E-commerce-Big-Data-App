# Big Data App cho dataset: E-commerce Customer Behavior and Purchase
# - Đọc CSV từ HDFS: hdfs://namenode:8020/input/ecommerce_data.csv
# - ETL -> Parquet (partition theo PurchaseDate)
# - Analytics: KPI, Top Category/Payment, Revenue theo ngày
# - ML: Logistic Regression dự đoán Churn; KMeans phân cụm khách hàng (RFM đơn giản)
#
# Yêu cầu môi trường:
#   - Spark master: spark://spark-master:7077 (có thể đổi trong sidebar)
#   - HDFS: namenode:8020
#   - pip install streamlit pyspark pandas matplotlib

import os
import traceback
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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
DEFAULT_INPUT = "hdfs://namenode:8020/input/ecommerce_data.csv"
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

    # —— Revenue theo ngày ——
    if "purchase_date" in df.columns and "total_amount" in df.columns:
        st.markdown("### Doanh thu theo ngày")
        rev_day = (df.groupBy("purchase_date")
                     .agg(_sum("total_amount").alias("revenue"),
                          count("*").alias("orders"))
                     .orderBy("purchase_date"))
        pdf = rev_day.toPandas()

        st.dataframe(pdf, use_container_width=True)

        if not pdf.empty and pdf["purchase_date"].notna().any():
            fig, ax = plt.subplots()
            ax.plot(pdf["purchase_date"].astype(str), pdf["revenue"])
            ax.set_title("Revenue theo ngày")
            ax.set_xlabel("Ngày"); ax.set_ylabel("Revenue")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

            # Orders theo ngày
            fig2, ax2 = plt.subplots()
            ax2.plot(pdf["purchase_date"].astype(str), pdf["orders"])
            ax2.set_title("Số đơn theo ngày")
            ax2.set_xlabel("Ngày"); ax2.set_ylabel("Orders")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig2)

            # AOV theo ngày (Average Order Value)
            pdf["aov"] = pdf["revenue"] / pdf["orders"].replace(0, float("nan"))
            fig3, ax3 = plt.subplots()
            ax3.plot(pdf["purchase_date"].astype(str), pdf["aov"])
            ax3.set_title("AOV theo ngày")
            ax3.set_xlabel("Ngày"); ax3.set_ylabel("AOV")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig3)
        else:
            st.warning("Không có dữ liệu ngày (có thể do không parse được 'Purchase Date').")

    # —— Top Category theo số đơn & doanh thu ——
    if "product_category" in df.columns:
        st.markdown("### Top Category")
        top_n = st.slider("Số lượng Top", 5, 30, 10, key="top_cat")
        top_cat = (df.groupBy("product_category")
                     .agg(count("*").alias("cnt"), _sum("total_amount").alias("revenue"))
                     .orderBy(col("cnt").desc())
                     .limit(top_n))
        pdf = top_cat.toPandas()
        st.dataframe(pdf, use_container_width=True)

        if not pdf.empty:
            # theo số đơn
            fig4, ax4 = plt.subplots()
            ax4.bar(pdf["product_category"].astype(str), pdf["cnt"])
            ax4.set_title(f"Top {top_n} Category (theo số đơn)")
            ax4.set_xlabel("Category"); ax4.set_ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig4)

            # theo doanh thu
            fig5, ax5 = plt.subplots()
            ax5.bar(pdf["product_category"].astype(str), pdf["revenue"])
            ax5.set_title(f"Top {top_n} Category (theo doanh thu)")
            ax5.set_xlabel("Category"); ax5.set_ylabel("Revenue")
            plt.xticks(rotation=45, ha='right')
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
            fig8, ax8 = plt.subplots()
            ax8.plot(rate["purchase_date"].astype(str), rate["return_rate"])
            ax8.set_title("Tỉ lệ trả hàng theo ngày")
            ax8.set_xlabel("Ngày"); ax8.set_ylabel("Return rate")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig8)

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
            fig10, ax10 = plt.subplots()
            ax10.hist(age_pdf["customer_age"], bins=20)
            ax10.set_title("Histogram độ tuổi")
            ax10.set_xlabel("Tuổi"); ax10.set_ylabel("Tần suất")
            st.pyplot(fig10)


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
        st.markdown("**Biểu đồ doanh thu theo tháng**")
        st.line_chart(pandas_monthly.set_index("month")["revenue"])
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

    # Phương thức thanh toán phổ biến
    pay_method = (df.groupBy("payment_method")
                    .agg(count("*").alias("count"))
                    .orderBy(F.desc("count")))
    pandas_pay = pay_method.toPandas()
    if not pandas_pay.empty:
        st.markdown("**Phân bố phương thức thanh toán**")
        fig, ax = plt.subplots()
        ax.pie(
            pandas_pay["count"],
            labels=pandas_pay["payment_method"],
            autopct="%1.1f%%",
            startangle=140,
        )
        ax.axis("equal")
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
    max_date = df.agg(F.max("purchase_ts")).collect()[0][0]
    rfm_df = rfm_df.withColumn(
        "recency_days", F.datediff(F.lit(max_date), F.col("last_purchase"))
    )

    pandas_rfm = rfm_df.toPandas()
    if pandas_rfm.empty:
        st.warning("Không có dữ liệu hành vi người dùng để phân tích.")
        return

    # Biểu đồ phân tán R-F-M
    st.markdown("**Phân bố RFM của khách hàng**")
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
    plt.title("Hành vi mua sắm khách hàng (RFM)")
    plt.colorbar(sc, label="Monetary (Tổng chi tiêu)")
    st.pyplot(fig)

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

    max_ts = df_nonull.agg(_max("purchase_ts").alias("mx")).collect()[0]["mx"]
    if max_ts is None:
        st.error("Không tính được mốc thời gian lớn nhất.")
        return

    rfm = rfm.withColumn("recency_days", expr(f"datediff(to_timestamp('{str(max_ts)}'), last_purchase)"))
    rfm_clean = rfm.na.drop(subset=["recency_days", "frequency", "monetary"])
    if rfm_clean.count() == 0:
        st.error("Không đủ dữ liệu sau khi làm sạch để phân cụm.")
        return

    # Pipeline tiền xử lý
    features = ["recency_days", "frequency", "monetary"]
    assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False)

    # ---- TỰ ĐỘNG CHỌN k THEO SILHOUETTE ----
    evaluator = ClusteringEvaluator(featuresCol="features")
    ks = list(range(2, 9))  # dải k thử (2..8). Có thể tăng nếu muốn.
    scores = []

    # (Tăng tốc nhẹ với sample nếu quá lớn)
    train_df = rfm_clean
    approx = rfm_clean.count()
    if approx > 200_000:
        train_df = rfm_clean.sample(False, 200_000 / approx, seed=42)

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

    # Hiển thị bảng Silhouette theo k
    st.markdown("#### Chọn số cụm tự động theo Silhouette")
    st.dataframe(
        pd.DataFrame(scores, columns=["k", "silhouette"]).sort_values("k"),
        use_container_width=True
    )
    st.success(f"Số cụm được chọn: **k = {best_k}**  (Silhouette = **{best_score:.4f}**)")

    # Dùng model tốt nhất để gán cụm cho TOÀN BỘ dữ liệu rfm_clean
    res = best_model.transform(rfm_clean)

    # Bảng kết quả từng khách
    st.dataframe(
        res.select("customer_id", "recency_days", "frequency", "monetary", "prediction")
           .orderBy("prediction", "customer_id")
           .limit(200)
           .toPandas(),
        use_container_width=True
    )

    # ---- BẢNG PROFILE CỤM (bạn nói không thấy, mình buộc hiển thị ở đây) ----
    st.markdown("#### Profile từng cụm (trung bình R/F/M + số lượng)")
    prof = (res.groupBy("prediction")
              .agg(avg("recency_days").alias("avg_recency"),
                   avg("frequency").alias("avg_freq"),
                   avg("monetary").alias("avg_monetary"),
                   count("*").alias("n_customers"))
              .orderBy("prediction"))
    st.dataframe(prof.toPandas(), use_container_width=True)

    # (Tuỳ chọn) Xuất kết quả
    with st.expander("Xuất kết quả phân cụm"):
        out_dir = st.text_input("Đường dẫn xuất (HDFS hoặc file://)", value=f"{parquet_path.rstrip('/')}_kmeans")
        if st.button("Ghi Parquet"):
            (res.select("customer_id", "recency_days", "frequency", "monetary", "prediction")
               .write.mode("overwrite").parquet(out_dir))
            st.success(f"Đã ghi: {out_dir}")

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
