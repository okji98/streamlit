import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

# MySQL 연결 정보
DB_HOST = 'localhost'
DB_NAME = 'tabledb'
DB_USER = 'root'
DB_TABLE = "cars"
DB_PORT='3306'
DB_PASS='Dhrgusdn1!'

# 페이지 설정
st.set_page_config(page_title='자동차 재고 현황', page_icon='bar_chart', layout='wide')

# MySQL 연결 함수
def create_database_connection():
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        if connection.is_connected():
            print('MySQL 데이터베이스에 성공적으로 연결되었습니다.')
        return connection
    except Error as e:
        print(f'데이터베이스 연결 중 오류 발생: {e}')
        return None
    
st.write(create_database_connection())

# SQLAlchemy 엔진 생성 (MySQL용)
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# 데이터 가져오기 (캐싱 적용)
@st.cache_data
def get_data():
    connection = create_database_connection()
    if connection is None:
        return pd.DataFrame()

def fetch_data():
    query = f"SELECT * FROM {DB_TABLE};"
    df = pd.read_sql(query, engine)
    return df
df = fetch_data()
st.dataframe(df)
# 필터링된 데이터 캐싱

# 데이터 가져오기

# 데이터가 비어있는 경우 처리

# 사이드바 생성
st.sidebar.header('선택하세요')

# 제조사 멀티셀렉트
st.sidebar.multiselect(
    '제조사를 선택하세요.',
    options = df["manufacturer"].unique(),
    default = df["manufacturer"].unique()
)
# 변속기 라디오 버튼
st.sidebar.radio(
    '변속기를 선택하세요.',
    options = df['automation'].unique()  
)

# 사용 카테고리 라디오 버튼
st.sidebar.radio(
    '카테고리를 선택하세요.',
    options = df['foreign_local_used'].unique()
)

# 데이터 필터링

# 필터링된 데이터가 비어있는 경우 처리

# 메인 화면

# (핵심 성과 지표) 계산

# KPI 표시
col1, col2, col3 = st.columns(3)
col1.metric(label='평균 가격:', value=f'US $ {int(df['price'].mean())}')
col2.metric(label='수량:', value=f'{len(df['foreign_local_used'])} Cars')
col3.metric(label='최초 생산 년도', value=f'{df['make_year'].min()}')

# 색상별 가격 막대 차트
def color_price():
    fig, ax = plt.subplots()
    ax.barh(df['color'], df['price'])
    return st.pyplot(fig)
# 제조사별 가격 막대 차트
def description_price():
    fig, ax = plt.subplots()
    ax.bar(df['description'], df['price'])
    return st.pyplot(fig)

# 컬럼 생성
price1, price2 = st.columns(2)
with price1:
    st.write('색상별 가격 추이'), 
    color_price()

with price2:
    st.write('제조사별 가격 추이'), 
    description_price()

# 시트 유형별 파이 차트

# 가격 지표

# 3열 레이아웃
col1_1, col1_2, col1_3 = st.columns(3)
with col1_1:
    
with col1_2: 
plt.pie(df['seat_make'], labels=df['price'])
with col1_3: 


# 생산 연도별 히스토그램

# Streamlit 스타일 숨기기 및 푸터 추가