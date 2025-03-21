import streamlit as st
import numpy as np
import pandas as pd

st.title('데이터프레임 튜토리얼')

# DateFrame 생성
dataframe = pd.DataFrame({
    'first column': np.arange(1, 5),
    'second column': [10, 20, 30, 40],
})

st.dataframe(dataframe)
st.write('##### use_container_width=True 기능은 데이터프레임을 컨테이너 크기에 확장할 때 사용(True/False)')
st.dataframe(dataframe, use_container_width=True)

st.title('Table')
st.table(dataframe)

# 매트릭(value를 중심으로 위아래 표시)
st.title('Metric')
st.metric(label='온도', value='10℃',delta='1.2℃')
st.metric(label='삼성전자', value='61,000원',delta='-1,200원')

# layout 컬럼으로 영역을 나누어 표기
st.title('Columns & metric')
col1, col2, col3 = st.columns(3)
col1.metric(label='달러USD', value='1,410 원', delta='12.00 원')
col2.metric(label='일본JPY(100엔)', value='928.0 원', delta='-7.44 원')
col3.metric(label='유럽연합EUR', value='1,335.82 원', delta='11.44 원')

st.write('#### 컬럼 영역의 너비를 지정')
col1, col2, col3 = st.columns([1, 5, 1])
with col1:
    st.write('**왼쪽**')
    st.write('안녕하세요 왼쪽 영역의 내용입니다.')
with col2:
    st.write('**가운데**')
    st.write('안녕하세요 가운데 영역의 내용입니다.')
with col3:
    st.write('**오른쪽**')
    st.write('안녕하세요 오른쪽 영역의 내용입니다.')

st.write('그리고 다시 전체 영역으로 내용이 표시됩니다.')

# 화면 왼쪽의 영역을 나누어 사용
with st.sidebar:
    tab1, tab2 = st.tabs(["1번 탭", "2번 탭"])
    with tab1:
        st.header("header1")
    with tab2:
        st.header("header2")
st.write('이 영역은 메인 페이지입니다.')