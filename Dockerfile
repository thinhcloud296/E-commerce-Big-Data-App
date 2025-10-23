# Dựa trên image jupyter + pyspark + hadoop
FROM jupyter/pyspark-notebook:hadoop-3

# Cài thư viện bằng pip ở build-time để không mất khi recreate container
USER root
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Trả quyền chạy lại cho user mặc định (jovyan)
USER ${NB_UID}
