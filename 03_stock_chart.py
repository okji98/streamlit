import streamlit as st
import FinanceDataReader as fdr
import datetime
import pandas as pd

# 한국거래소 상장종목 전체 가져오기
df = fdr.StockListing("KRX")

# 5건 화면에 출력합니다.
df_head = df.head()
st.dataframe(df_head)

date = st.date_input(
    '조회 시작일을 선택해 주세요.',
    datetime.datetime(2022, 1, 1)
)

code = st.text_input(
    '종목코드',
    value = '',
    placeholder = '종목코드를 입력해 주세요.'
)

if code and date:
    df = fdr.DataReader(code, date)
    # 날짜를 기준으로 종가를 가져옴
    data = df.sort_index(ascending = True).loc[:, 'Close']
    st.line_chart(data)