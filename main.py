import os
import traceback
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)
from pyspark.sql.functions import (
    col, to_timestamp, date_format, coalesce, trim, regexp_replace,
    countDistinct, count, sum as _sum, avg, expr, when, max as _max, collect_list, size
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
# Spark helpers
# =========================
def add_churn_from_recency(df, customer_col="customer_id", ts_col="purchase_ts", cutoff_days=180):
    mx = df.agg(_max(ts_col).alias("mx")).collect()[0]["mx"]
    if mx is None:
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
    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    df = spark.read.parquet(parquet_path).cache()
    _ = df.count()
    return df

def _spark_is_alive(spark):
    try:
        return spark is not None and not spark.sparkContext._jsc.sc().isStopped()
    except Exception:
        return False

def _hard_reset_spark_jvm():
    try: SparkSession.clearActiveSession(); SparkSession.clearDefaultSession()
    except Exception: pass
    try:
        from pyspark import SparkContext
        sc = SparkContext._active_spark_context
        if sc is not None: sc.stop()
    except Exception: pass

def start_spark(app_name: str = "EcomBigDataApp", master: str = DEFAULT_SPARK_MASTER, log_level: str = "WARN"):
    try:
        spark = (SparkSession.builder.appName(app_name).master(master)
                 .config("spark.sql.shuffle.partitions", "64")
                 .config("spark.ui.showConsoleProgress", "true")
                 .config("spark.sql.legacy.timeParserPolicy", "LEGACY").getOrCreate())
        if not _spark_is_alive(spark): raise RuntimeError("SC stopped")
        spark.sparkContext.setLogLevel(log_level)
        return spark
    except Exception as e1:
        print(f"[WARN] 1st attempt failed: {repr(e1)}")
        _hard_reset_spark_jvm()
        try:
            spark = (SparkSession.builder.appName(app_name).master(master)
                     .config("spark.sql.shuffle.partitions", "64")
                     .config("spark.ui.showConsoleProgress", "true")
                     .config("spark.sql.legacy.timeParserPolicy", "LEGACY").getOrCreate())
            if not _spark_is_alive(spark): raise RuntimeError("SC stopped again")
            spark.sparkContext.setLogLevel(log_level)
            return spark
        except Exception as e2:
            print(f"[WARN] 2nd attempt failed, fallback local: {repr(e2)}")
            _hard_reset_spark_jvm()
            spark = (SparkSession.builder.appName(app_name + "-local").master("local[*]")
                     .config("spark.sql.shuffle.partitions", "8")
                     .config("spark.ui.showConsoleProgress", "true")
                     .config("spark.sql.legacy.timeParserPolicy", "LEGACY").getOrCreate())
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
# ETL Function (Returns path or raises error)
# =========================
def run_etl(spark, src_path: str, dst_parquet: str, ts_format: str):
    st.info(f"Đọc CSV từ: {src_path}")
    try:
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(src_path)

        rename_pairs = {"Customer ID": "customer_id", "Customer": "customer_id", "Customer Name": "customer_name", "Customer Age": "customer_age", "Gender": "gender", "Purchase Date": "purchase_date_raw", "Product Category": "product_category", "Product Price": "product_price", "Quantity": "quantity", "Total Purchase Amount": "total_amount", "Payment Method": "payment_method", "Returns": "returns", "Churn": "churn"}
        for old, new in rename_pairs.items():
            if old in df.columns and new not in df.columns:
                df = df.withColumnRenamed(old, new)

        for num_col in ["product_price", "total_amount"]:
            if num_col in df.columns:
                df = df.withColumn(num_col, regexp_replace(col(num_col).cast("string"), r"[^0-9\.\-]", "").cast("double"))
        if "quantity" in df.columns:
            df = df.withColumn("quantity", regexp_replace(col("quantity").cast("string"), r"[^0-9\-]", "").cast("int"))
        if "returns" in df.columns: df = df.withColumn("returns", col("returns").cast("int"))
        if "churn" in df.columns: df = df.withColumn("churn", col("churn").cast("int"))

        raw = "purchase_date_raw"
        if raw in df.columns:
            df = df.withColumn(raw, trim(col(raw)))
            df = df.withColumn(raw, regexp_replace(col(raw), "T|Z$", " "))
            df = df.withColumn(raw, regexp_replace(col(raw), r"[\u00A0\u2007\u202F]", " "))
            df = df.withColumn(raw, regexp_replace(col(raw), r"\s+", " "))

            patterns = [ts_format] if ts_format else []
            patterns += ["M/d/yyyy H:mm", "M/d/yyyy h:mm a", "MM/dd/yyyy HH:mm", "dd/M/yyyy H:mm", "dd/MM/yyyy HH:mm", "M/d/yyyy H:mm:ss", "MM/dd/yyyy HH:mm:ss", "dd/M/yyyy H:mm:ss", "dd/MM/yyyy HH:mm:ss", "yyyy-MM-dd", "MM/dd/yyyy", "dd/MM/yyyy"]
            ts_exprs = [to_timestamp(col(raw), p) for p in patterns]
            df = df.withColumn("purchase_ts", coalesce(*ts_exprs))
            df = df.withColumn("purchase_date", date_format(col("purchase_ts"), "yyyy-MM-dd"))
            null_ratio = df.select(avg((col("purchase_ts").isNull()).cast("int"))).first()[0]
            if null_ratio and null_ratio >= 0.9999:
                st.warning("Không parse được cột 'Purchase Date'. Hãy nhập đúng định dạng.")
        else:
            df = df.withColumn("purchase_ts", F.lit(None)).withColumn("purchase_date", F.lit(None))

        needs_fill = True
        if "churn" in df.columns:
            null_ratio = df.select(avg((col("churn").isNull()).cast("int"))).first()[0]
            needs_fill = (null_ratio is None) or (null_ratio > 0.5)
        if needs_fill:
            df = add_churn_from_recency(df, customer_col="customer_id", ts_col="purchase_ts", cutoff_days=180)

        for c in ["gender", "product_category", "payment_method"]:
            if c in df.columns: df = df.fillna({c: "unknown"})

        writer = df.write.mode("overwrite")
        if "purchase_date" in df.columns: writer = writer.partitionBy("purchase_date")
        writer.parquet(dst_parquet)

        st.success(f"ETL hoàn tất → {dst_parquet}")
        return dst_parquet
    except Exception as e:
        st.error(f"Lỗi ETL: {e}")
        st.code(traceback.format_exc())
        raise # Re-raise exception to stop execution if needed

# =========================
# Analytics Function (Returns results dict)
# =========================
def run_analytics(spark, parquet_path: str):
    try:
        app_id = spark.sparkContext.applicationId
        df = load_parquet_cached(app_id, parquet_path)
        
        results = {}
        results['row_count'] = df.count()
        results['col_count'] = len(df.columns)
        results['column_list'] = df.columns
        
        # KPIs
        kpi = {}
        if "customer_id" in df.columns: kpi["Số khách hàng"] = df.select(countDistinct("customer_id")).first()[0]
        if "product_category" in df.columns: kpi["Số category"] = df.select(countDistinct("product_category")).first()[0]
        if "total_amount" in df.columns: kpi["Tổng doanh thu"] = df.agg(_sum("total_amount")).first()[0]
        if "returns" in df.columns: kpi["Tỉ lệ trả hàng (%)"] = round((df.agg(avg(col("returns").cast("double"))).first()[0] or 0) * 100, 2)
        if "churn" in df.columns: kpi["Tỉ lệ churn (%)"] = round((df.agg(avg(col("churn").cast("double"))).first()[0] or 0) * 100, 2)
        results['kpi_data'] = kpi

        # Daily Revenue & Orders
        if "purchase_date" in df.columns and "total_amount" in df.columns:
            daily = df.groupBy("purchase_date").agg(_sum("total_amount").alias("revenue"), count("*").alias("orders")).orderBy("purchase_date")
            pdf = daily.toPandas()
            if not pdf.empty and not pdf["purchase_date"].isna().all():
                pdf["purchase_date"] = pd.to_datetime(pdf["purchase_date"], errors="coerce")
                pdf = pdf.dropna(subset=["purchase_date"]).sort_values("purchase_date")
                if not pdf.empty:
                    pdf["rev_ma7"] = pdf["revenue"].rolling(7, min_periods=1).mean()
                    pdf_indexed = pdf.set_index("purchase_date")
                    results['daily_revenue_chart_data'] = pdf_indexed[["revenue", "rev_ma7"]]
                    results['daily_orders_chart_data'] = pdf_indexed[["orders"]]

        # Top Category
        if "product_category" in df.columns:
            top_n = 10
            top_cat_spark = df.groupBy("product_category").agg(count("*").alias("cnt"), _sum("total_amount").alias("revenue")).orderBy(col("cnt").desc()).limit(top_n)
            pdf_top_cat = top_cat_spark.toPandas()
            results['top_cat_data'] = pdf_top_cat
            if not pdf_top_cat.empty:
                pdf_cnt = pdf_top_cat.sort_values("cnt", ascending=True)
                fig_cnt, ax_cnt = plt.subplots()
                ax_cnt.barh(pdf_cnt["product_category"].astype(str), pdf_cnt["cnt"])
                ax_cnt.set_title(f"Top {top_n} Category (theo số đơn)")
                ax_cnt.set_xlabel("Count"); ax_cnt.set_ylabel("Category")
                plt.tight_layout(); results['top_cat_cnt_fig'] = fig_cnt

                pdf_rev = pdf_top_cat.sort_values("revenue", ascending=True)
                fig_rev, ax_rev = plt.subplots()
                ax_rev.barh(pdf_rev["product_category"].astype(str), pdf_rev["revenue"])
                ax_rev.set_title(f"Top {top_n} Category (theo doanh thu)")
                ax_rev.set_xlabel("Revenue"); ax_rev.set_ylabel("Category")
                plt.tight_layout(); results['top_cat_rev_fig'] = fig_rev

        # Payment Methods
        if "payment_method" in df.columns:
            pm_spark = df.groupBy("payment_method").agg(count("*").alias("cnt"), _sum("total_amount").alias("revenue")).orderBy(col("cnt").desc())
            pdf_pm = pm_spark.toPandas()
            results['payment_method_data'] = pdf_pm
            if not pdf_pm.empty:
                fig_pm_cnt, ax_pm_cnt = plt.subplots()
                ax_pm_cnt.bar(pdf_pm["payment_method"].astype(str), pdf_pm["cnt"])
                ax_pm_cnt.set_title("Số đơn theo phương thức thanh toán")
                ax_pm_cnt.set_xlabel("Payment Method"); ax_pm_cnt.set_ylabel("Count")
                plt.xticks(rotation=45, ha='right'); plt.tight_layout(); results['payment_method_cnt_fig'] = fig_pm_cnt

                fig_pm_rev, ax_pm_rev = plt.subplots()
                ax_pm_rev.bar(pdf_pm["payment_method"].astype(str), pdf_pm["revenue"])
                ax_pm_rev.set_title("Doanh thu theo phương thức thanh toán")
                ax_pm_rev.set_xlabel("Payment Method"); ax_pm_rev.set_ylabel("Revenue")
                plt.xticks(rotation=45, ha='right'); plt.tight_layout(); results['payment_method_rev_fig'] = fig_pm_rev

        # Return Rate
        if "purchase_date" in df.columns and "returns" in df.columns:
            rate_spark = df.groupBy("purchase_date").agg(avg(col("returns").cast("double")).alias("return_rate")).orderBy("purchase_date")
            pdf_rate = rate_spark.toPandas()
            if not pdf_rate.empty and pdf_rate["purchase_date"].notna().any():
                pdf_rate["purchase_date"] = pd.to_datetime(pdf_rate["purchase_date"], errors="coerce")
                pdf_rate = pdf_rate.dropna(subset=["purchase_date"]).set_index("purchase_date")
                if not pdf_rate.empty: results['return_rate_chart_data'] = pdf_rate[["return_rate"]]

        # Churn by Gender
        if "churn" in df.columns and "gender" in df.columns:
            churn_g_spark = df.groupBy("gender").agg(avg(col("churn").cast("double")).alias("churn_rate")).orderBy("gender")
            pdf_churn_g = churn_g_spark.toPandas()
            if not pdf_churn_g.empty:
                fig_churn, ax_churn = plt.subplots()
                ax_churn.bar(pdf_churn_g["gender"].astype(str), pdf_churn_g["churn_rate"])
                ax_churn.set_title("Churn rate theo giới tính")
                ax_churn.set_xlabel("Gender"); ax_churn.set_ylabel("Churn rate")
                plt.xticks(rotation=0); plt.tight_layout(); results['churn_gender_fig'] = fig_churn # No rotation needed here

        # Age Distribution
        if "customer_age" in df.columns:
            age_pdf = df.select("customer_age").toPandas().dropna()
            if not age_pdf.empty:
                data = age_pdf["customer_age"].astype(float)
                data = data[(data > 0) & (data < 100)]
                if len(data) > 0:
                    bins_count = int(np.clip(np.sqrt(len(data)), 10, 40))
                    hist, bin_edges = np.histogram(data, bins=bins_count)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    hist_df = pd.DataFrame({'age_center': bin_centers, 'count': hist}).set_index('age_center')
                    results['age_dist_chart_data'] = hist_df[['count']]
                    results['age_dist_bins_count'] = bins_count # Store bin count for context

        # Close all figures just created to prevent memory leaks in long-running app
        plt.close('all')
        return results

    except Exception as e:
        st.error(f"Lỗi Analytics: {e}")
        st.code(traceback.format_exc())
        return None # Return None on error

# =========================
# Trend Analysis Function (Returns results dict)
# =========================
def run_trend_analysis(df):
    results = {}
    try:
        # Monthly Revenue
        if "purchase_ts" in df.columns and "total_amount" in df.columns:
            monthly_rev = (df.withColumn("month", date_format("purchase_ts", "yyyy-MM"))
                             .groupBy("month").agg(_sum("total_amount").alias("revenue")).orderBy("month"))
            pandas_monthly = monthly_rev.toPandas()
            if not pandas_monthly.empty:
                pandas_monthly["month"] = pd.to_datetime(pandas_monthly["month"], format="%Y-%m", errors="coerce")
                pandas_monthly = pandas_monthly.dropna(subset=["month"]).sort_values("month")
                if not pandas_monthly.empty:
                    min_month = pandas_monthly["month"].min().to_period("M")
                    max_month = pandas_monthly["month"].max().to_period("M")
                    # Ensure range is valid
                    if min_month <= max_month:
                        full_range = pd.period_range(min_month, max_month, freq="M").to_timestamp()
                        pm = pandas_monthly.set_index("month").reindex(full_range, fill_value=0).rename_axis("month").reset_index()

                        import matplotlib.dates as mdates
                        fig_mrev, ax_mrev = plt.subplots()
                        ax_mrev.plot(pm["month"], pm["revenue"], linewidth=2)
                        locator = mdates.AutoDateLocator(minticks=4, maxticks=12)
                        formatter = mdates.ConciseDateFormatter(locator)
                        ax_mrev.xaxis.set_major_locator(locator)
                        ax_mrev.xaxis.set_major_formatter(formatter)
                        ax_mrev.set_title("Doanh thu theo tháng")
                        ax_mrev.set_xlabel("Tháng"); ax_mrev.set_ylabel("Revenue")
                        plt.tight_layout(); results['monthly_rev_fig'] = fig_mrev

        # Top 5 Categories (Bar Chart Data)
        if "product_category" in df.columns and "total_amount" in df.columns:
            top_cat_spark = df.groupBy("product_category").agg(_sum("total_amount").alias("revenue")).orderBy(F.desc("revenue")).limit(5)
            pandas_cat = top_cat_spark.toPandas()
            if not pandas_cat.empty: results['top_5_cat_chart_data'] = pandas_cat.set_index("product_category")["revenue"]

        # Payment Method Distribution (Figure)
        if "payment_method" in df.columns:
            pay_method_spark = df.groupBy("payment_method").agg(count("*").alias("count")).orderBy(F.desc("count"))
            pandas_pay = pay_method_spark.toPandas()
            if not pandas_pay.empty:
                fig_pay, ax_pay = plt.subplots()
                ax_pay.bar(pandas_pay["payment_method"].astype(str), pandas_pay["count"])
                ax_pay.set_title("Số đơn theo phương thức thanh toán")
                ax_pay.set_xlabel("Payment Method"); ax_pay.set_ylabel("Count")
                plt.xticks(rotation=45, ha='right'); plt.tight_layout(); results['payment_method_trend_fig'] = fig_pay

        plt.close('all')
        return results
    except Exception as e:
        st.error(f"Lỗi khi phân tích xu hướng: {e}")
        st.code(traceback.format_exc())
        return None

# =========================
# Customer Behavior Function (Returns results dict)
# =========================
def run_customer_behavior(df):
    results = {}
    try:
        if not all(c in df.columns for c in ["customer_id", "purchase_ts", "total_amount"]):
            st.error("Thiếu cột cần thiết cho phân tích RFM.")
            return None

        rfm_df_spark = df.groupBy("customer_id").agg(F.max("purchase_ts").alias("last_purchase"), count("*").alias("frequency"), _sum("total_amount").alias("monetary"))
        max_date_row = df.agg(F.max("purchase_ts")).collect()
        if not max_date_row or max_date_row[0][0] is None:
            st.error("Không thể tính toán RFM do thiếu mốc thời gian.")
            return None
        max_date = max_date_row[0][0]
        rfm_df_spark = rfm_df_spark.withColumn("recency_days", F.datediff(F.lit(max_date), F.col("last_purchase")))
        rfm_df_spark = rfm_df_spark.select("customer_id", "recency_days", "frequency", "monetary")
        pandas_rfm = rfm_df_spark.toPandas()
        if pandas_rfm.empty:
            st.warning("Không có dữ liệu RFM để phân tích.")
            return None

        # RFM Scatter Plot
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        sc = ax_scatter.scatter(pandas_rfm["recency_days"], pandas_rfm["frequency"], c=pandas_rfm["monetary"], cmap="viridis", alpha=0.7)
        ax_scatter.set_xlabel("Recency (days)"); ax_scatter.set_ylabel("Frequency")
        ax_scatter.set_title("Hành vi mua sắm khách hàng (R vs F, màu là M)")
        plt.colorbar(sc, label="Monetary (Tổng chi tiêu)"); plt.tight_layout(); results['rfm_scatter_fig'] = fig_scatter

        # RFM Distributions
        fig_r, ax_r = plt.subplots(); ax_r.hist(pandas_rfm['recency_days'].dropna(), bins=30, color='skyblue', edgecolor='black')
        ax_r.set_title("Phân bố Recency"); ax_r.set_xlabel("Số ngày"); ax_r.set_ylabel("Số lượng KH"); plt.tight_layout(); results['rfm_dist_r_fig'] = fig_r

        f_data = pandas_rfm['frequency'].dropna(); f_q99 = f_data.quantile(0.99); f_data_clipped = f_data[f_data <= f_q99]
        fig_f, ax_f = plt.subplots(); ax_f.hist(f_data_clipped, bins=30, color='lightgreen', edgecolor='black')
        ax_f.set_title(f"Phân bố Frequency (lọc > {f_q99:.0f})"); ax_f.set_xlabel("Số lần mua"); ax_f.set_ylabel("Số lượng KH"); plt.tight_layout(); results['rfm_dist_f_fig'] = fig_f

        m_data = pandas_rfm['monetary'].dropna(); m_q99 = m_data.quantile(0.99); m_data_clipped = m_data[m_data <= m_q99]
        fig_m, ax_m = plt.subplots(); ax_m.hist(m_data_clipped, bins=30, color='salmon', edgecolor='black')
        ax_m.set_title(f"Phân bố Monetary (lọc > {m_q99:,.0f})"); ax_m.set_xlabel("Tổng chi tiêu"); ax_m.set_ylabel("Số lượng KH"); plt.tight_layout(); results['rfm_dist_m_fig'] = fig_m

        # RFM Stats
        results['rfm_stats_data'] = pandas_rfm[["recency_days", "frequency", "monetary"]].describe().T.rename(columns={"mean": "Giá trị trung bình"})

        plt.close('all')
        return results
    except Exception as e:
        st.error(f"Lỗi khi phân tích hành vi người dùng: {e}")
        st.code(traceback.format_exc())
        return None

# =========================
# KMEANS Segmentation Function (Returns results dict)
# =========================
def _interpret_cluster(r, f, m, g_r, g_f, g_m):
    # ... (Keep the helper function as is)
    r_score = "Thấp (Tốt)" if r < g_r * 0.9 else "Cao (Xấu)" if r > g_r * 1.1 else "Trung bình"
    f_score = "Cao (Tốt)" if f > g_f * 1.1 else "Thấp (Xấu)" if f < g_f * 0.9 else "Trung bình"
    m_score = "Cao (Tốt)" if m > g_m * 1.1 else "Thấp (Xấu)" if m < g_m * 0.9 else "Trung bình"
    if r_score == "Thấp (Tốt)" and f_score == "Cao (Tốt)" and m_score == "Cao (Tốt)": return "🌟 Khách hàng VIP/Trung thành"
    elif r_score == "Cao (Xấu)" and f_score == "Thấp (Xấu)": return "⚠️ Khách hàng có nguy cơ rời bỏ"
    elif r_score == "Thấp (Tốt)" and f_score == "Thấp (Xấu)": return "💡 Khách hàng mới"
    elif f_score == "Cao (Tốt)": return "💖 Khách hàng thân thiết"
    elif m_score == "Cao (Tốt)": return "💰 Khách hàng chi tiêu cao"
    elif r_score == "Cao (Xấu)": return "💤 Khách hàng ngủ đông"
    return "Khách hàng Tiềm năng/Trung bình"

def run_kmeans_segmentation(spark, parquet_path: str):
    results = {}
    try:
        app_id = spark.sparkContext.applicationId
        df = load_parquet_cached(app_id, parquet_path)
        
        required = ["customer_id", "purchase_ts", "total_amount"]
        if not all(c in df.columns for c in required): st.error(f"Thiếu cột cho RFM: {required}"); return None
        
        df_nonull = df.na.drop(subset=["purchase_ts"])
        if df_nonull.isEmpty(): st.error("Không có 'purchase_ts' hợp lệ."); return None

        rfm = df_nonull.groupBy("customer_id").agg(_max("purchase_ts").alias("last_purchase"), count("*").alias("frequency"), _sum("total_amount").alias("monetary"))
        max_ts_row = df_nonull.agg(_max("purchase_ts").alias("mx")).first()
        if not max_ts_row or max_ts_row["mx"] is None: st.error("Không tính được max_ts."); return None
        max_ts = max_ts_row["mx"]
        rfm = rfm.withColumn("recency_days", expr(f"datediff(to_timestamp('{str(max_ts)}'), last_purchase)"))
        rfm_clean = rfm.na.drop(subset=["recency_days", "frequency", "monetary"])
        if rfm_clean.isEmpty(): st.error("Không đủ dữ liệu sạch để phân cụm."); return None

        global_avgs = rfm_clean.agg(avg("recency_days"), avg("frequency"), avg("monetary")).first()
        global_r, global_f, global_m = (global_avgs[i] or 0 for i in range(3))

        features = ["recency_days", "frequency", "monetary"]
        assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
        scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False)
        evaluator = ClusteringEvaluator(featuresCol="features")
        ks = list(range(2, 9))
        scores = []
        train_df = rfm_clean.sample(False, min(1.0, 200000 / rfm_clean.count()), seed=42) if rfm_clean.count() > 200000 else rfm_clean

        best_model = None; best_k = None; best_score = float("-inf")
        progress_bar = st.progress(0, text="Đang tìm k tốt nhất...")
        for i, k in enumerate(ks):
            model = Pipeline(stages=[assembler, scaler, KMeans(featuresCol="features", k=k, seed=42)]).fit(train_df)
            pred_k = model.transform(train_df)
            sil = evaluator.evaluate(pred_k)
            scores.append((k, sil))
            if sil > best_score: best_score, best_k, best_model = sil, k, model
            progress_bar.progress((i + 1) / len(ks), text=f"Đã thử k={k} (Silhouette: {sil:.4f})")
        progress_bar.empty()
        
        results['k_selection_chart_data'] = pd.DataFrame(scores, columns=["k", "silhouette"]).set_index("k")
        results['best_k'] = best_k
        results['best_score'] = best_score
        
        if best_model is None: # Handle case where no k was found (e.g., all silhouette scores invalid)
            st.error("Không thể tìm thấy model tốt nhất.")
            return None

        res = best_model.transform(rfm_clean) # Apply best model to full clean data
        
        prof_spark = res.groupBy("prediction").agg(avg("recency_days").alias("avg_recency"), avg("frequency").alias("avg_freq"), avg("monetary").alias("avg_monetary"), count("*").alias("n_customers")).orderBy("prediction")
        prof_pd = prof_spark.toPandas()
        
        cluster_profiles_list = []
        for _, row in prof_pd.iterrows():
            avg_r, avg_f, avg_m, n_cust = row["avg_recency"], row["avg_freq"], row["avg_monetary"], row["n_customers"]
            title = _interpret_cluster(avg_r, avg_f, avg_m, global_r, global_f, global_m)
            metrics = {
                "Recency (TB)": (f"{avg_r:,.1f} ngày", f"{avg_r - global_r:,.1f} vs TB"),
                "Frequency (TB)": (f"{avg_f:,.1f} lần", f"{avg_f - global_f:,.1f} vs TB"),
                "Monetary (TB)": (f"{avg_m:,.0f}", f"{avg_m - global_m:,.0f} vs TB")
            }
            helps = {
                 "Recency (TB)": f"Trung bình chung: {global_r:,.1f} ngày",
                 "Frequency (TB)": f"Trung bình chung: {global_f:,.1f} lần",
                 "Monetary (TB)": f"Trung bình chung: {global_m:,.0f}"
            }
            cluster_profiles_list.append({
                "cluster_id": row["prediction"], "title": title, "n_customers": f"{n_cust:,.0f} KH",
                "metrics": metrics, "helps": helps
            })
        results['cluster_profiles'] = cluster_profiles_list

        res_selected = res.select("recency_days", "frequency", "monetary", "prediction")
        viz_limit = 5000
        total_res = res_selected.count()
        if total_res > viz_limit:
            fraction = viz_limit / total_res
            viz_df = res_selected.sample(False, fraction, seed=42).limit(viz_limit).toPandas()
        else:
            viz_df = res_selected.toPandas()
        viz_df["prediction"] = viz_df["prediction"].astype(str)
        results['scatter_chart_data'] = viz_df

        results['detailed_data_sample'] = res.select("customer_id", "recency_days", "frequency", "monetary", "prediction").orderBy("prediction", "customer_id").limit(200).toPandas()
        results['full_results_spark_df_ref'] = res # Store reference for export

        return results

    except Exception as e:
        st.error(f"Lỗi ML (KMeans): {e}")
        st.code(traceback.format_exc())
        return None

# =========================
# Product Clustering Function (Returns results dict)
# =========================
def build_product_clustering_model(spark, parquet_path: str, k_value: int = 0):
    results = {}
    try:
        app_id = spark.sparkContext.applicationId
        df = load_parquet_cached(app_id, parquet_path)
        required = ["product_category", "product_price", "quantity", "total_amount", "returns"]
        if not all(c in df.columns for c in required): st.error(f"Thiếu cột cho Product Clustering: {required}"); return None

        prod_features = df.groupBy("product_category").agg(avg("product_price").alias("avg_price"), avg("quantity").alias("avg_quantity"), _sum("total_amount").alias("total_revenue"), avg(col("returns").cast("double")).alias("avg_returns_rate"))
        prod_features_clean = prod_features.na.drop()
        count_prod_features = prod_features_clean.count()
        if count_prod_features == 0: st.error("Không đủ dữ liệu sản phẩm sạch."); return None

        features = ["avg_price", "avg_quantity", "total_revenue", "avg_returns_rate"]
        assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
        scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
        train_df = prod_features_clean

        best_model = None; best_k = None; best_score = float("-inf"); scores = []; auto_k = (k_value <= 1)

        if not auto_k:
            best_k = k_value
            if best_k >= count_prod_features:
                st.warning(f"k={best_k} >= số category ({count_prod_features}). Giảm k xuống {max(2, count_prod_features - 1)}.")
                best_k = max(2, count_prod_features - 1)
            st.info(f"Sử dụng số cụm k = {best_k}")
            kmeans = KMeans(featuresCol="features", k=best_k, seed=42)
            best_model_pipeline = Pipeline(stages=[assembler, scaler, kmeans])
            best_model = best_model_pipeline.fit(train_df)
            if count_prod_features > 1 and best_k <= count_prod_features:
                try: best_score = ClusteringEvaluator(featuresCol="features").evaluate(best_model.transform(train_df))
                except Exception: best_score = -999; st.warning("Không thể tính Silhouette.")
            else: best_score = -999
            results['k_selection_prod_chart_data'] = None # No chart for manual k
        else:
            evaluator = ClusteringEvaluator(featuresCol="features")
            max_k = min(count_prod_features, 11) if count_prod_features > 1 else 2
            ks = list(range(2, max_k)) if max_k > 2 else ([2] if count_prod_features >= 2 else [])
            progress_bar_prod = st.progress(0, text="Đang tìm k tốt nhất cho sản phẩm...")
            if not ks: st.error("Chỉ có 1 category, không thể phân cụm."); return None

            for i, k in enumerate(ks):
                kmeans = KMeans(featuresCol="features", k=k, seed=42)
                model = Pipeline(stages=[assembler, scaler, kmeans]).fit(train_df)
                pred_k = model.transform(train_df)
                try: sil = evaluator.evaluate(pred_k)
                except Exception: sil = -1.0
                scores.append((k, sil))
                if sil > best_score and sil >= -1.0: best_score, best_k, best_model = sil, k, model
                progress_bar_prod.progress((i + 1) / len(ks), text=f"Đã thử k={k} (Silhouette: {sil:.4f})")
            progress_bar_prod.empty()

            if best_k is None:
                if scores:
                    scores.sort(key=lambda item: item[1], reverse=True)
                    best_k = scores[0][0]; best_score = scores[0][1]
                    st.warning(f"Silhouette không dương. Chọn k={best_k} ({best_score:.4f}).")
                    kmeans = KMeans(featuresCol="features", k=best_k, seed=42) # Retrain
                    best_model = Pipeline(stages=[assembler, scaler, kmeans]).fit(train_df)
                else: st.error("Không thể xác định số cụm."); return None
            
            results['k_selection_prod_chart_data'] = pd.DataFrame(scores, columns=["k", "silhouette"]).set_index("k") if scores else None

        results['best_k_prod'] = best_k
        results['best_score_prod'] = best_score
        
        if best_model is None: st.error("Không thể tạo model phân cụm sản phẩm."); return None

        clustered_prods = best_model.transform(prod_features_clean)
        
        pca = PCA(k=2, inputCol="features", outputCol="pca_features")
        pca_success = False
        try:
            pca_model = pca.fit(clustered_prods)
            clustered_prods_with_pca = pca_model.transform(clustered_prods)
            pca_success = True
            
            # Prepare PCA plot data and figure
            pca_data_pd = clustered_prods_with_pca.select("product_category", "pca_features", "prediction").toPandas()
            if not pca_data_pd.empty:
                pca_data_pd['x'] = pca_data_pd['pca_features'].apply(lambda v: v[0] if v and len(v)>0 else None)
                pca_data_pd['y'] = pca_data_pd['pca_features'].apply(lambda v: v[1] if v and len(v)>1 else None)
                pca_data_pd.dropna(subset=['x', 'y'], inplace=True)
                
                if not pca_data_pd.empty:
                    fig_pca, ax_pca = plt.subplots(figsize=(10, 8))
                    clusters = sorted(pca_data_pd['prediction'].unique())
                    colors = plt.cm.get_cmap('viridis', max(len(clusters), 2))
                    markers = ['^', 'o', 's', 'p', '*', 'X', 'D', 'v', '<', '>']
                    for i, cluster in enumerate(clusters):
                        cluster_data = pca_data_pd[pca_data_pd['prediction'] == cluster]
                        ax_pca.scatter(cluster_data['x'], cluster_data['y'], color=colors(i), label=f'Cụm {cluster}', alpha=0.8, marker=markers[i % len(markers)])
                        sample_labels = cluster_data.sample(min(3, len(cluster_data)), random_state=42)
                        for _, point in sample_labels.iterrows():
                            ax_pca.text(point['x'] + 0.02, point['y'], point['product_category'], fontsize=8)
                    ax_pca.set_title(f'Phân cụm sản phẩm (k={best_k}) - PCA 2D')
                    ax_pca.set_xlabel('PC1'); ax_pca.set_ylabel('PC2')
                    ax_pca.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax_pca.grid(True); fig_pca.tight_layout()
                    results['pca_plot_fig'] = fig_pca
                    results['pca_variance_info'] = f"Explained variance: PC1={pca_model.explainedVariance[0]:.2%}, PC2={pca_model.explainedVariance[1]:.2%}, Total={sum(pca_model.explainedVariance):.2%}"
                else: results['pca_plot_fig'] = None; results['pca_variance_info'] = "Không có dữ liệu PCA hợp lệ."
            else: results['pca_plot_fig'] = None; results['pca_variance_info'] = "Không có dữ liệu PCA."
            
        except Exception as e:
            st.warning(f"Không thể thực hiện/vẽ PCA: {e}")
            results['pca_plot_fig'] = None
            results['pca_variance_info'] = "PCA thất bại."
            clustered_prods_with_pca = clustered_prods # Use non-PCA df if PCA fails

        results['clustered_prods_spark_df_ref'] = clustered_prods_with_pca # Store ref with PCA if successful

        cluster_contents_spark = clustered_prods_with_pca.groupBy("prediction").agg(collect_list("product_category").alias("products"), count("*").alias("num_products")).orderBy("prediction")
        cluster_pdf = cluster_contents_spark.toPandas()
        results['cluster_contents_prod'] = [{"cluster_id": row["prediction"], "products": row["products"], "num_products": row["num_products"]} for _, row in cluster_pdf.iterrows()]

        plt.close('all') # Close PCA figure
        return results

    except Exception as e:
        st.error(f"Lỗi xây dựng model sản phẩm: {e}")
        st.code(traceback.format_exc())
        return None

# =========================
# UI Layout
# =========================
st.set_page_config(page_title="E-commerce Big Data App", layout="wide")
st.title("🛒 E-commerce Big Data App — Spark ETL • Analytics • ML")

# Initialize session state keys if they don't exist
if 'active_section' not in st.session_state: st.session_state['active_section'] = None
if 'etl_success' not in st.session_state: st.session_state['etl_success'] = False
if 'parquet_path' not in st.session_state: st.session_state['parquet_path'] = PARQUET_DIR
if 'analytics_results' not in st.session_state: st.session_state['analytics_results'] = None
if 'trends_results' not in st.session_state: st.session_state['trends_results'] = None
if 'behavior_results' not in st.session_state: st.session_state['behavior_results'] = None
if 'segmentation_results' not in st.session_state: st.session_state['segmentation_results'] = None
if 'product_clustering_results' not in st.session_state: st.session_state['product_clustering_results'] = None
if 'product_model_built' not in st.session_state: st.session_state['product_model_built'] = False # For recommendation UI trigger

# Sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình & Kết nối")
    master = st.text_input("Spark master", value=DEFAULT_SPARK_MASTER)
    log_level = st.selectbox("Spark log level", ["ERROR","WARN","INFO","DEBUG"], index=1)
    input_path = st.text_input("CSV HDFS path", value=DEFAULT_INPUT)
    parquet_out = st.text_input("Parquet output", value=st.session_state.get("parquet_path", PARQUET_DIR))
    ts_format = st.text_input("Định dạng thời gian (Purchase Date)", value="M/d/yyyy H:mm")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Khởi tạo Spark"):
            with st.spinner("Đang khởi tạo Spark..."):
                try:
                    st.session_state["spark"] = start_spark(master=master, log_level=log_level)
                    st.success("✅ SparkSession đã sẵn sàng.")
                except Exception as e:
                    st.error(f"❌ Khởi tạo thất bại: {e}")
                    # st.code(traceback.format_exc()) # Maybe too verbose for sidebar
    with col2:
         if st.button("🔄 Restart Spark"):
            with st.spinner("Đang khởi động lại Spark..."):
                try:
                    old = st.session_state.get("spark")
                    if old is not None:
                        try: old.stop()
                        except Exception: pass
                    _hard_reset_spark_jvm()
                    try: st.cache_resource.clear()
                    except Exception: pass
                    st.session_state["spark"] = start_spark(master=master, log_level=log_level)
                    st.success("✅ Đã restart Spark.")
                    # Clear potentially stale results? Optional.
                    # st.session_state['active_section'] = None 
                    # ... clear other results ...
                except Exception as e:
                    st.error(f"❌ Restart thất bại: {e}")

    st.divider()
    st.header("📊 Chạy Phân tích")
    # ETL Button
    if st.button("1️⃣ Chạy ETL (CSV → Parquet)"):
        spark = ensure_spark(master, log_level=log_level)
        if spark:
            with st.spinner("Đang chạy ETL..."):
                etl_output_path = run_etl(spark, src_path=input_path, dst_parquet=parquet_out, ts_format=ts_format.strip())
                if etl_output_path:
                    st.session_state['parquet_path'] = etl_output_path
                    st.session_state['etl_success'] = True
                    # Clear old results if ETL is rerun
                    st.session_state['analytics_results'] = None
                    st.session_state['trends_results'] = None
                    st.session_state['behavior_results'] = None
                    st.session_state['segmentation_results'] = None
                    st.session_state['product_clustering_results'] = None
                    st.session_state['product_model_built'] = False
                    st.session_state['active_section'] = None # Reset view after ETL
                else:
                    st.session_state['etl_success'] = False
        else:
            st.warning("⚠️ Spark chưa sẵn sàng.")

    # Update parquet path input based on state
    ppath_input = st.session_state.get("parquet_path", PARQUET_DIR)

    # Analytics Button
    if st.button("2️⃣ Chạy Analytics Tổng quan"):
        spark = ensure_spark(master, log_level=log_level)
        if spark and st.session_state.get('etl_success', False):
             with st.spinner("Đang chạy Analytics..."):
                results = run_analytics(spark, ppath_input)
                if results:
                    st.session_state['analytics_results'] = results
                    st.session_state['active_section'] = 'analytics'
        elif not st.session_state.get('etl_success', False):
             st.warning("⚠️ Vui lòng chạy ETL thành công trước.")
        else: st.warning("⚠️ Spark chưa sẵn sàng.")

    # Advanced Analysis Buttons in Expander
    with st.expander("📈 Phân tích Nâng cao"):
        if st.button("📊 Phân tích Xu hướng"):
            spark = ensure_spark(master, log_level=log_level)
            if spark and st.session_state.get('etl_success', False):
                with st.spinner("Đang phân tích xu hướng..."):
                    df = load_parquet_cached(spark.sparkContext.applicationId, ppath_input)
                    results = run_trend_analysis(df)
                    if results:
                        st.session_state['trends_results'] = results
                        st.session_state['active_section'] = 'trends'
            elif not st.session_state.get('etl_success', False):
                st.warning("⚠️ Vui lòng chạy ETL thành công trước.")
            else: st.warning("⚠️ Spark chưa sẵn sàng.")

        if st.button("🧠 Phân tích Hành vi (RFM)"):
            spark = ensure_spark(master, log_level=log_level)
            if spark and st.session_state.get('etl_success', False):
                 with st.spinner("Đang phân tích hành vi..."):
                    df = load_parquet_cached(spark.sparkContext.applicationId, ppath_input)
                    results = run_customer_behavior(df)
                    if results:
                        st.session_state['behavior_results'] = results
                        st.session_state['active_section'] = 'behavior'
            elif not st.session_state.get('etl_success', False):
                 st.warning("⚠️ Vui lòng chạy ETL thành công trước.")
            else: st.warning("⚠️ Spark chưa sẵn sàng.")

    # Machine Learning Buttons in Expander
    with st.expander("🤖 Machine Learning"):
        if st.button("👥 Phân cụm Khách hàng (KMeans)"):
            spark = ensure_spark(master, log_level=log_level)
            if spark and st.session_state.get('etl_success', False):
                with st.spinner("Đang phân cụm khách hàng..."):
                    results = run_kmeans_segmentation(spark, ppath_input)
                    if results:
                        st.session_state['segmentation_results'] = results
                        st.session_state['active_section'] = 'segmentation'
            elif not st.session_state.get('etl_success', False):
                 st.warning("⚠️ Vui lòng chạy ETL thành công trước.")
            else: st.warning("⚠️ Spark chưa sẵn sàng.")

        st.markdown("---")
        st.subheader("🛒 Gợi ý Sản phẩm")
        k_input_prod = st.number_input("Số cụm SP (k) (0=auto)", min_value=0, value=0, step=1, key="k_prod_input", help="Nhập k > 1 hoặc 0 để tự động.")
        if st.button("🛠️ Xây dựng Model Cụm SP"):
            spark = ensure_spark(master, log_level=log_level)
            if spark and st.session_state.get('etl_success', False):
                with st.spinner("Đang xây dựng model cụm sản phẩm..."):
                    results = build_product_clustering_model(spark, ppath_input, k_value=k_input_prod)
                    if results:
                        st.session_state['product_clustering_results'] = results
                        # Store necessary part for recommendation UI
                        st.session_state['clustered_prods_df_for_rec'] = results.get('clustered_prods_spark_df_ref')
                        st.session_state['product_model_built'] = True
                        st.session_state['active_section'] = 'product_clustering'
                    else: # Handle build failure
                        st.session_state['product_model_built'] = False

            elif not st.session_state.get('etl_success', False):
                 st.warning("⚠️ Vui lòng chạy ETL thành công trước.")
            else: st.warning("⚠️ Spark chưa sẵn sàng.")


# =========================
# Main Display Area Logic
# =========================
active_section = st.session_state.get('active_section')

if active_section == 'analytics':
    st.header("📊 Kết quả Analytics Tổng quan")
    results = st.session_state.get('analytics_results')
    if results:
        st.write(f"**Số dòng:** {results.get('row_count', 'N/A'):,} | **Số cột:** {results.get('col_count', 'N/A')}")
        with st.expander("Danh sách cột"): st.code(", ".join(results.get('column_list',[])))
        
        st.markdown("### KPIs tổng quan")
        kpi_data = results.get('kpi_data', {})
        if kpi_data:
            cols_kpi = st.columns(len(kpi_data))
            for i, (k, v) in enumerate(kpi_data.items()):
                 # Format currency and percentages nicely
                 if "thu" in k.lower() or "amount" in k.lower():
                     val_str = f"{v:,.0f}" if v is not None else "N/A"
                 elif "%" in k:
                     val_str = f"{v:.2f}%" if v is not None else "N/A"
                 else:
                     val_str = f"{v:,}" if v is not None else "N/A"
                 cols_kpi[i].metric(label=k, value=val_str)
        else: st.write("Không đủ dữ liệu tính KPI.")

        st.markdown("### Doanh thu & Đơn hàng theo ngày")
        if 'daily_revenue_chart_data' in results and results['daily_revenue_chart_data'] is not None:
             st.markdown("#### Xu hướng doanh thu (Revenue & MA-7)")
             st.line_chart(results['daily_revenue_chart_data'])
        else: st.warning("Không có dữ liệu doanh thu theo ngày.")
        if 'daily_orders_chart_data' in results and results['daily_orders_chart_data'] is not None:
             st.markdown("#### Số lượng đơn hàng hàng ngày")
             st.bar_chart(results['daily_orders_chart_data'])
        else: st.warning("Không có dữ liệu đơn hàng theo ngày.")

        st.markdown("### Top Category")
        if 'top_cat_data' in results and results['top_cat_data'] is not None:
             st.dataframe(results['top_cat_data'], use_container_width=True)
             col_tc1, col_tc2 = st.columns(2)
             with col_tc1: 
                  if results.get('top_cat_cnt_fig'): st.pyplot(results['top_cat_cnt_fig'])
             with col_tc2:
                  if results.get('top_cat_rev_fig'): st.pyplot(results['top_cat_rev_fig'])
        else: st.warning("Không có dữ liệu Top Category.")
        
        st.markdown("### Phân bố phương thức thanh toán")
        if 'payment_method_data' in results and results['payment_method_data'] is not None:
             st.dataframe(results['payment_method_data'], use_container_width=True)
             col_pm1, col_pm2 = st.columns(2)
             with col_pm1:
                  if results.get('payment_method_cnt_fig'): st.pyplot(results['payment_method_cnt_fig'])
             with col_pm2:
                   if results.get('payment_method_rev_fig'): st.pyplot(results['payment_method_rev_fig'])
        else: st.warning("Không có dữ liệu phương thức thanh toán.")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
             st.markdown("### Tỉ lệ trả hàng theo ngày")
             if 'return_rate_chart_data' in results and results['return_rate_chart_data'] is not None:
                  st.line_chart(results['return_rate_chart_data'])
             else: st.warning("Không có dữ liệu tỉ lệ trả hàng.")
        with col_r2:
             st.markdown("### Tỉ lệ churn theo giới tính")
             if 'churn_gender_fig' in results and results.get('churn_gender_fig'):
                  st.pyplot(results['churn_gender_fig'])
             else: st.warning("Không có dữ liệu churn theo giới tính.")

        st.markdown("### Phân bố độ tuổi khách hàng")
        if 'age_dist_chart_data' in results and results['age_dist_chart_data'] is not None:
             st.write(f"Biểu đồ cột thể hiện tần suất theo nhóm tuổi (chia {results.get('age_dist_bins_count','N/A')} nhóm).")
             st.bar_chart(results['age_dist_chart_data'])
        else: st.warning("Không có dữ liệu phân bố tuổi.")
        
    else:
        st.info("Nhấn nút 'Chạy Analytics Tổng quan' ở thanh bên để xem kết quả.")

elif active_section == 'trends':
    st.header("📈 Kết quả Phân tích Xu hướng")
    results = st.session_state.get('trends_results')
    if results:
        if 'monthly_rev_fig' in results and results.get('monthly_rev_fig'):
             st.markdown("**Doanh thu theo tháng**"); st.pyplot(results['monthly_rev_fig'])
        else: st.warning("Không có dữ liệu doanh thu theo tháng.")
        
        if 'top_5_cat_chart_data' in results and results['top_5_cat_chart_data'] is not None:
             st.markdown("**Top 5 danh mục SP (Doanh thu)**"); st.bar_chart(results['top_5_cat_chart_data'])
        else: st.warning("Không có dữ liệu top 5 category.")

        if 'payment_method_trend_fig' in results and results.get('payment_method_trend_fig'):
             st.markdown("**Phân bố phương thức thanh toán (Số đơn)**"); st.pyplot(results['payment_method_trend_fig'])
        else: st.warning("Không có dữ liệu phương thức thanh toán.")
    else:
        st.info("Nhấn nút 'Phân tích Xu hướng' ở thanh bên để xem kết quả.")

elif active_section == 'behavior':
    st.header("🧠 Kết quả Phân tích Hành vi (RFM)")
    results = st.session_state.get('behavior_results')
    if results:
        if results.get('rfm_scatter_fig'):
             st.markdown("**Phân bố RFM tổng quan**"); st.pyplot(results['rfm_scatter_fig'])
        
        st.markdown("**Phân bố chi tiết R, F, M**")
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1: 
             if results.get('rfm_dist_r_fig'): st.pyplot(results['rfm_dist_r_fig'])
        with col_b2:
             if results.get('rfm_dist_f_fig'): st.pyplot(results['rfm_dist_f_fig'])
        with col_b3:
             if results.get('rfm_dist_m_fig'): st.pyplot(results['rfm_dist_m_fig'])
             
        if results.get('rfm_stats_data') is not None:
             st.markdown("**Thống kê RFM**"); st.dataframe(results['rfm_stats_data'])
    else:
        st.info("Nhấn nút 'Phân tích Hành vi (RFM)' ở thanh bên để xem kết quả.")

elif active_section == 'segmentation':
    st.header("👥 Kết quả Phân cụm Khách hàng")
    results = st.session_state.get('segmentation_results')
    if results:
        st.markdown("#### 1. Chọn số cụm (k)")
        if results.get('k_selection_chart_data') is not None:
             st.line_chart(results['k_selection_chart_data'])
        st.success(f"Số cụm được chọn: **k = {results.get('best_k', 'N/A')}** (Silhouette = **{results.get('best_score', -999):.4f}**)")
        
        st.markdown("---")
        st.markdown("#### 2. Kết quả phân cụm")
        tab_profile, tab_viz, tab_data = st.tabs(["📊 Profile Cụm", "📈 Trực quan hóa Cụm", "📋 Dữ liệu chi tiết"])
        
        with tab_profile:
             st.subheader("Profile từng cụm")
             cluster_profiles = results.get('cluster_profiles', [])
             if cluster_profiles:
                 for profile in cluster_profiles:
                     st.markdown(f"### Cụm {profile['cluster_id']}: {profile['title']}")
                     cols = st.columns(len(profile['metrics']) + 1)
                     cols[0].metric("Số lượng KH", profile['n_customers'])
                     i = 1
                     for metric_name, (value, delta) in profile['metrics'].items():
                         cols[i].metric(metric_name, value, delta, help=profile['helps'].get(metric_name))
                         i += 1
                     st.divider()
             else: st.warning("Không có dữ liệu profile cụm.")
        
        with tab_viz:
             st.subheader("Trực quan hóa các cụm (R-F-M)")
             scatter_data = results.get('scatter_chart_data')
             if scatter_data is not None and not scatter_data.empty:
                  st.scatter_chart(scatter_data, x="recency_days", y="frequency", size="monetary", color="prediction", use_container_width=True)
             else: st.warning("Không có dữ liệu để vẽ biểu đồ scatter.")
             
        with tab_data:
             st.subheader("Dữ liệu chi tiết (200 mẫu)")
             if results.get('detailed_data_sample') is not None:
                  st.dataframe(results['detailed_data_sample'], use_container_width=True)
             
             
    else:
        st.info("Nhấn nút 'Phân cụm Khách hàng (KMeans)' ở thanh bên để xem kết quả.")

elif active_section == 'product_clustering':
    st.header("🛠️ Kết quả Xây dựng Model Cụm Sản phẩm")
    results = st.session_state.get('product_clustering_results')
    if results:
        k_chart_data = results.get('k_selection_prod_chart_data')
        if k_chart_data is not None: # Only show if auto-k was run
             st.markdown("#### Chọn số cụm (k) tự động theo Silhouette")
             st.line_chart(k_chart_data)
        st.success(f"Số cụm sản phẩm được sử dụng: **k = {results.get('best_k_prod', 'N/A')}** (Silhouette = **{results.get('best_score_prod', -999):.4f}**)")
        
        st.markdown("### Nội dung từng cụm sản phẩm")
        cluster_contents = results.get('cluster_contents_prod', [])
        if cluster_contents:
            for content in cluster_contents:
                with st.expander(f"Cụm {content['cluster_id']}: {content['num_products']} sản phẩm"):
                     st.write(", ".join(content['products']))
        else: st.warning("Không có dữ liệu nội dung cụm.")
        
        st.markdown("### Trực quan hóa các cụm sản phẩm (PCA 2D)")
        pca_fig = results.get('pca_plot_fig')
        if pca_fig:
            st.pyplot(pca_fig)
            st.write(results.get('pca_variance_info', ''))
        else:
            st.warning("Không thể vẽ biểu đồ PCA (có thể do lỗi hoặc dữ liệu quá ít).")
            st.write(results.get('pca_variance_info', '')) # Show reason if available
    else:
        st.info("Nhấn nút 'Xây dựng Model Cụm SP' ở thanh bên để xem kết quả.")

else:
    # Default view when no section is active
    st.info("⬅️ Chào mừng! Vui lòng chọn một hành động từ thanh bên để bắt đầu.")
    if not st.session_state.get('etl_success', False):
         st.warning("⚠️ Bước đầu tiên là chạy ETL.")

# Recommendation UI (Needs product model results in session state)
if st.session_state.get("product_model_built", False):
    st.header("💡 Gợi ý Sản phẩm Tương tự")
    
    clustered_prods_df = st.session_state.get('clustered_prods_df_for_rec')
    if clustered_prods_df is None:
        st.warning("⚠️ Không tìm thấy dữ liệu model sản phẩm. Vui lòng chạy lại bước 'Xây dựng Model Cụm SP'.")
    else:
        try:
            # Lấy unique categories từ DataFrame đã lưu
            unique_cats = [row["product_category"] for row in clustered_prods_df.select("product_category").distinct().collect()]
        except Exception as e:
            unique_cats = []
            st.warning(f"Không thể tải danh sách sản phẩm từ model đã lưu: {e}")

        if unique_cats:
            selected_category = st.selectbox("Chọn Sản phẩm:", sorted(unique_cats), key="select_product_rec")
            if st.button("🔍 Tìm gợi ý", key="find_recommend"):
                 with st.spinner("Đang tìm sản phẩm tương tự..."):
                    try:
                        selected_row = clustered_prods_df.filter(col("product_category") == selected_category).select("prediction").first()
                        if selected_row: selected_cluster = selected_row["prediction"]
                        else: st.error(f"Lỗi: Không tìm thấy cụm cho '{selected_category}'."); st.stop()
                        
                        recommendations = (clustered_prods_df
                                           .filter((col("prediction") == selected_cluster) & (col("product_category") != selected_category))
                                           .select("product_category", "avg_price", "avg_quantity", "total_revenue", "avg_returns_rate")
                                           .orderBy(F.desc("total_revenue"))
                                           .limit(5))
                        rec_count = recommendations.count() # Check count before toPandas
                        if rec_count > 0:
                            rec_pdf = recommendations.toPandas()
                            st.markdown(f"**Gợi ý cho '{selected_category}' (từ Cụm {selected_cluster})**")
                            st.dataframe(rec_pdf, use_container_width=True, column_config={
                                "product_category": "Sản phẩm gợi ý", "avg_price": "Giá TB", "avg_quantity": "SL TB",
                                "total_revenue": "Doanh thu", "avg_returns_rate": "Tỉ lệ trả TB"})
                        else: st.warning("Không tìm thấy sản phẩm tương tự trong cùng cụm.")
                    except Exception as e: st.error(f"Lỗi khi tìm gợi ý: {e}"); st.code(traceback.format_exc())
        else: st.warning("Không có dữ liệu sản phẩm trong model để chọn.")