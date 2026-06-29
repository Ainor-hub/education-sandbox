from datetime import datetime, timedelta
import os
import time
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = os.path.join(REPO_ROOT, "tmp", "retail_customer_project")

RAW_DIR = os.path.join(BASE_DIR, "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def extract_retail_dataset():
    os.makedirs(RAW_DIR, exist_ok=True)

    input_file = os.path.join(REPO_ROOT, "Online Retail.xlsx")
    output_file = os.path.join(RAW_DIR, "retail_raw.csv")

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            "Dataset not found. Upload Online Retail.xlsx inside the include folder."
        )

    df = pd.read_excel(input_file)
    df.to_csv(output_file, index=False)

    time.sleep(10)
    print("Raw retail dataset extracted successfully.")
    print(f"Raw data shape: {df.shape}")


def clean_retail_data():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    input_file = os.path.join(RAW_DIR, "retail_raw.csv")
    output_file = os.path.join(CLEAN_DIR, "clean_retail.csv")

    df = pd.read_csv(input_file)
    before_rows = len(df)

    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    after_rows = len(df)

    df.to_csv(output_file, index=False)

    time.sleep(10)
    print("Retail data cleaned successfully.")
    print(f"Rows before cleaning: {before_rows}")
    print(f"Rows after cleaning: {after_rows}")
    print(f"Removed rows: {before_rows - after_rows}")


def calculate_rfm_metrics():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_file = os.path.join(CLEAN_DIR, "clean_retail.csv")
    output_file = os.path.join(OUTPUT_DIR, "rfm_customers.csv")

    df = pd.read_csv(input_file)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    reference_date = df["InvoiceDate"].max() + timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    rfm.to_csv(output_file, index=False)

    time.sleep(10)
    print("RFM metrics calculated successfully.")
    print(rfm.head())


def segment_customers():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_file = os.path.join(OUTPUT_DIR, "rfm_customers.csv")
    output_file = os.path.join(OUTPUT_DIR, "customer_segments.csv")

    rfm = pd.read_csv(input_file)

    features = rfm[["Recency", "Frequency", "Monetary"]]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Segment"] = kmeans.fit_predict(scaled_features)

    rfm.to_csv(output_file, index=False)

    time.sleep(10)
    print("Customer segmentation completed successfully.")
    print(rfm.head())


def identify_churn_risk():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_file = os.path.join(OUTPUT_DIR, "customer_segments.csv")
    output_file = os.path.join(OUTPUT_DIR, "churn_risk_customers.csv")

    df = pd.read_csv(input_file)

    recency_limit = df["Recency"].quantile(0.75)
    frequency_limit = df["Frequency"].quantile(0.25)

    df["ChurnRisk"] = df.apply(
        lambda row: "High Risk"
        if row["Recency"] >= recency_limit and row["Frequency"] <= frequency_limit
        else "Low Risk",
        axis=1,
    )

    df.to_csv(output_file, index=False)

    time.sleep(10)
    print("Churn risk analysis completed successfully.")
    print(df["ChurnRisk"].value_counts())


def save_summary_results():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_file = os.path.join(OUTPUT_DIR, "churn_risk_customers.csv")
    output_file = os.path.join(OUTPUT_DIR, "pipeline_summary.csv")

    df = pd.read_csv(input_file)

    summary = df.groupby(["Segment", "ChurnRisk"]).agg(
        Customers=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        TotalRevenue=("Monetary", "sum"),
    ).reset_index()

    summary.to_csv(output_file, index=False)

    time.sleep(10)
    print("Final pipeline summary saved successfully.")
    print(summary)


default_args = {
    "owner": "retail_project_team",
    "start_date": days_ago(1),
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="retail_customer_segmentation_churn",
    default_args=default_args,
    description="Retail customer segmentation and churn risk pipeline using ETL, RFM, and K-Means.",
    schedule_interval=None,
    catchup=False,
    tags=["retail", "rfm", "segmentation", "churn"],
) as dag:

    start = BashOperator(
        task_id="start_pipeline",
        bash_command='echo "Retail customer pipeline started."',
    )

    extract = PythonOperator(
        task_id="extract_retail_dataset",
        python_callable=extract_retail_dataset,
    )

    clean = PythonOperator(
        task_id="clean_retail_data",
        python_callable=clean_retail_data,
    )

    rfm = PythonOperator(
        task_id="calculate_rfm_metrics",
        python_callable=calculate_rfm_metrics,
    )

    segment = PythonOperator(
        task_id="segment_customers",
        python_callable=segment_customers,
    )

    churn = PythonOperator(
        task_id="identify_churn_risk",
        python_callable=identify_churn_risk,
    )

    summary = PythonOperator(
        task_id="save_summary_results",
        python_callable=save_summary_results,
    )

    end = BashOperator(
        task_id="end_pipeline",
        bash_command='echo "Retail customer pipeline completed successfully."',
    )

    start >> extract >> clean >> rfm >> segment >> churn >> summary >> end