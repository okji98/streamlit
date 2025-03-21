import pandas as pd
import mysql.connector
from mysql.connector import Error
import pymysql
import streamlit as st

DB_HOST='localhost'
DB_PORT='3306'
DB_NAME='tabledb'
DB_USER='root'
DB_PASS='Dhrgusdn1!'
DB_TABLE='cars'

print("DB_HOST:", DB_HOST)
print("DB_PORT:", DB_PORT)
print("DB_NAME:", DB_NAME)
print("DB_USER:", DB_USER)
print("DB_PASS:", DB_PASS)
print("DB_PASS:", DB_TABLE)

# CSV 파일 경로
CSV_FILE_PATH = "./data/cars.csv"

def create_database_connection():
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        if connection.is_connected():
            print("MySQL 데이터베이스에 성공적으로 연결되었습니다.")
        return connection
    except Error as e:
        print(f"데이터베이스 연결 중 오류 발생: {e}")
        return None
    
def create_table(connection):
    try:
        cursor = connection.cursor()
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {DB_TABLE} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            foreign_local_used VARCHAR(20),
            color VARCHAR(20),
            wheel_drive INT,
            automation VARCHAR(20),
            seat_make VARCHAR(20),
            price BIGINT,
            description VARCHAR(100),
            make_year INT,
            manufacturer VARCHAR(50)
        );
        """
        cursor.execute(create_table_query)
        connection.commit()
        print(f"'{DB_TABLE}' 테이블이 성공적으로 생성되었습니다.")
    except Error as e:
        print(f"테이블 생성 중 오류 발생: {e}")
    finally:
        cursor.close()

def insert_data(connection, df):
    try:
        cursor = connection.cursor()
        insert_query = f"""
        INSERT INTO {DB_TABLE} (
            foreign_local_used, color, wheel_drive, automation, seat_make, 
            price, description, make_year, manufacturer
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        data = [tuple(row) for row in df.values]
        cursor.executemany(insert_query, data)
        connection.commit()
        print(f"{cursor.rowcount}개의 레코드가 '{DB_TABLE}' 테이블에 삽입되었습니다.")
    except Error as e:
        print(f"데이터 삽입 중 오류 발생: {e}")
    finally:
        cursor.close()

def main():
    # CSV 파일 읽기 (index_col=0으로 첫 번째 열을 인덱스로 설정)
    try:
        df = pd.read_csv(CSV_FILE_PATH, index_col=0)
        print('CSV 파일을 성공적으로 읽었습니다.')
    except FileNotFoundError:
        print(f'파일 {CSV_FILE_PATH}을 찾을 수 없습니다.')
        return
    except Exception as e:
        print(f'CSV 파일 읽기 중 오류 발생: {e}')
        return
    
    # 열 이름 조정
    expected_columns = [
        "foreign_local_used", "color", "wheel_drive", "automation",
        "seat_make", "price", "description", "make_year", "manufacturer"
    ]
    df.columns = expected_columns # 열 이름 매핑

    # 데이터베이스 연결
    connection = create_database_connection()
    if connection is None:
        return
    
    # 테이블 생성
    create_table(connection)

    # 데이터 삽입
    insert_data(connection, df)

    # 연결 종료
    if connection.is_connected():
        connection.close()
        print('MySQL 연결이 종료되었습니다.')

if __name__ == "__main__":
    main()