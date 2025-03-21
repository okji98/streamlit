import streamlit as st
import FinanceDataReader as fdr
import datetime
import pandas as pd
import numpy as np

with st.sidebar:
    date =st.date_input(
    '조회시작일을 선택해 주세요.',
    datetime.datetime(2020, 1, 1)
)
    side_df = fdr.StockListing("KRX")
    stock_list = side_df[['Code', 'Name']].copy()
    stock_list['Code_Name'] = stock_list['Code'] + ' - ' + stock_list['Name']
    selected_stock = st.selectbox(
        '종목 선택',
        options=stock_list['Code_Name'].tolist(),
        placeholder='종목을 선택해주세요'
    )
    
    # 선택된 종목에서 종목코드만 추출
    code = selected_stock.split(' - ')[0] if selected_stock else ''


#     code = st.text_input(
#     '종목코드',
#     value='',
#     placeholder='종목코드를 입력해주세요'
# )
        
st.title('종목 차트 검색')

select, compare = st.tabs(["차트", "데이터"])

with select: 
    st.write("#### 주식 그래프")
    if code and date:
        df = fdr.DataReader(code, date)
        data = df.sort_index(ascending=True).loc[:, 'Close']
        st.line_chart(data)
    

with compare:
    st.write("#### 대한민국 주식 데이터")

    df = fdr.StockListing("KRX")
    st.write(df.head(100))

with st.expander('칼럼 설명'):
        st.markdown('''
        - Open: 시가
        - High: 고가
        - Low: 저가
        - Close: 종가
        - Volumn: 거래량
        - Adj Close: 수정 종가
        ''')


    
  



