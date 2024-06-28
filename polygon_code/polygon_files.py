import boto3
from botocore.config import Config
from datetime import datetime, timedelta
import os
import threading
from queue import Queue
import pandas as pd

def initialize_session(aws_access_key_id, aws_secret_access_key, endpoint_url):
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    s3 = session.client(
        's3',
        endpoint_url=endpoint_url,
        config=Config(signature_version='s3v4'),
    )
    return s3

def list_objects(s3, bucket_name, prefix):
    keys = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            keys.append(obj['Key'])
    return keys

def download_file(s3, bucket_name, object_key, local_file_path):
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    s3.download_file(bucket_name, object_key, local_file_path)
    print(f"Downloaded {object_key} to {local_file_path}")

def generate_dates(start_date, end_date):
        # Generate a date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Format the dates as strings
    dates = date_range.strftime('%Y-%m-%d').tolist()
    return dates

def worker(s3, bucket_name, prefix, date_queue, agg_type): # eventually will be data_type
    while not date_queue.empty():
        date = date_queue.get()
        year = date[:4]
        month = date[5:7]
        object_key = f'{prefix}/{agg_type}_aggs_v1/{year}/{month}/{date}.csv.gz'
        local_file_path = os.path.join('polygon_data', f'{agg_type}_aggs', year, month, f'{date}.csv.gz')
        # check if local file path exists already
        if not os.path.exists(local_file_path):
            print(f"Processing {date}")
            object_keys = list_objects(s3, bucket_name, f'{prefix}/{agg_type}_aggs_v1/{year}/{month}/')
            if object_key in object_keys:
                download_file(s3, bucket_name, object_key, local_file_path)
            else:
                print(f"{object_key} does not exist in S3 bucket")
            date_queue.task_done()

def thread_stocks():
    aws_access_key_id = '307a150f-b709-4bdb-bf95-f550bb58af36'
    aws_secret_access_key = 'q42WQRfUjTeouBEoaCSC_LHNhDFhzIIb'
    endpoint_url = 'https://files.polygon.io'
    bucket_name = 'flatfiles'
    prefix = 'us_stocks_sip'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    s3 = initialize_session(aws_access_key_id, aws_secret_access_key, endpoint_url)
    dates = generate_dates(start_date, end_date)

    agg_types = ['day', 'minute']
    # data_types = ['trades', 'quotes', 'day_aggs_v1', 'minute_aggs_v1']
    # for data_type in data_types:
    for agg_type in agg_types:
        date_queue = Queue()
        for date in dates:
            date_queue.put(date)
        
        threads = []
        for _ in range(18):
            thread = threading.Thread(target=worker, args=(s3, bucket_name, prefix, date_queue, agg_type))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()

if __name__ == "__main__":
    thread_stocks()
