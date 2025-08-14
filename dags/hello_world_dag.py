from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'hello_world_dag',
    default_args=default_args,
    description='A simple Hello World DAG',
    schedule=timedelta(hours=1),
    catchup=False,
    tags=['example', 'hello_world'],
)

def print_hello():
    return 'Hello World from Airflow!'

def print_date():
    import datetime
    return f'Current date and time: {datetime.datetime.now()}'

hello_task = PythonOperator(
    task_id='hello_task',
    python_callable=print_hello,
    dag=dag,
)

date_task = PythonOperator(
    task_id='date_task', 
    python_callable=print_date,
    dag=dag,
)

bash_task = BashOperator(
    task_id='bash_task',
    bash_command='echo "Hello from Bash operator!"',
    dag=dag,
)

# Task dependencies
hello_task >> [date_task, bash_task]