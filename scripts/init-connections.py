#!/usr/bin/env python3
"""
Initialize Airflow connections for MLOps setup
"""

import os
import sys
from airflow.models import Connection
from airflow import settings

def create_connection(conn_id, conn_type, host, port=None, schema=None, login=None, password=None, extra=None):
    """Create or update an Airflow connection"""
    session = settings.Session()
    
    # Check if connection already exists
    existing_conn = session.query(Connection).filter(Connection.conn_id == conn_id).first()
    
    if existing_conn:
        print(f"Connection '{conn_id}' already exists, updating...")
        existing_conn.conn_type = conn_type
        existing_conn.host = host
        existing_conn.port = port
        existing_conn.schema = schema
        existing_conn.login = login
        existing_conn.password = password
        existing_conn.extra = extra
    else:
        print(f"Creating new connection '{conn_id}'...")
        new_conn = Connection(
            conn_id=conn_id,
            conn_type=conn_type,
            host=host,
            port=port,
            schema=schema,
            login=login,
            password=password,
            extra=extra
        )
        session.add(new_conn)
    
    session.commit()
    session.close()
    print(f"✅ Connection '{conn_id}' created/updated successfully")

def main():
    """Initialize all MLOps connections"""
    print("🚀 Initializing MLOps connections...")
    
    # MLflow connection
    create_connection(
        conn_id='mlflow_default',
        conn_type='http',
        host='mlflow',
        port=5000,
        extra='{"endpoint": "http://mlflow:5000"}'
    )
    
    # PostgreSQL connection
    create_connection(
        conn_id='postgres_mlflow',
        conn_type='postgres',
        host='postgres',
        port=5432,
        schema='mlflow',
        login='mlflow',
        password='mlflow'
    )
    
    # Additional MLflow tracking connection
    create_connection(
        conn_id='mlflow_tracking',
        conn_type='http',
        host='mlflow',
        port=5000,
        extra='{"tracking_uri": "http://mlflow:5000", "artifact_root": "/mlflow/artifacts"}'
    )
    
    # MinIO connection (for future use)
    create_connection(
        conn_id='minio_default',
        conn_type='s3',
        host='minio',
        port=9000,
        login='minioadmin',
        password='minioadmin',
        extra='{"endpoint_url": "http://minio:9000", "aws_access_key_id": "minioadmin", "aws_secret_access_key": "minioadmin"}'
    )
    
    print("✅ All MLOps connections initialized successfully!")

if __name__ == '__main__':
    main()