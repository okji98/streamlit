import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 한글 깨짐 방지
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 탭 분류
analyze, risk = st.tabs(["위험도 분석 데이터", "위험도 진단하기"])
### 첫번째 탭 ###
# 각 컬럼마다의 상관관계 분석
with analyze:
# 데이터 불러와서 dataframe으로 만들기
    data_file = "./Maternal Health Risk Data Set.csv"
    data = pd.read_csv(data_file)
    df = pd.DataFrame(data)
    st.header("산모 건강 데이터 분석 시각화")
    st.dataframe(df)
# 산모 건강 지표 간 상관관계
    st.write('#### 산모 건강 지표 간 상관관계')
    def heat_map():
        df_drop = df.drop('RiskLevel', axis=1)
        fig, ax = plt.subplots(figsize = (8, 6))
        corr_matrix = df_drop[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']].corr()
        # 히트맵 시각화
        sns.heatmap(data=corr_matrix, annot=True, fmt='.2f', linewidths=0.5, 
                    cbar_kws={'shrink': 0.5}, cmap='RdYlBu_r', vmin=-1, vmax=1, ax=ax)
        ax.set_title('산모 건강 지표 간 상관관계', fontsize=15)
        return fig
    st.pyplot(heat_map())
# 산모의 나이대 분류
    bins = [20, 30, 40, 50 ,60, 100]
    labels = ['20대', '30대', '40대', '50대', '60대']
    df['AgeGroup'] = pd.cut(df['Age'], bins = bins, labels = labels, right = False)
# 나이에 따른 혈압 시각화(bar 그래프)
    st.write('#### 나이에 따른 혈압 그래프')
    def age_bp():
        # DataFrame을 긴 형식으로 변환 (melt)
        df_melted = pd.melt(df, id_vars=['AgeGroup'], value_vars=['SystolicBP', 'DiastolicBP'], 
                            var_name='혈압 수치', value_name='BloodPressure')
        fig, ax = plt.subplots(figsize = (8, 6))
        plt.xlabel('나이')
        plt.ylabel('혈압')
        barplot = sns.barplot(x='AgeGroup', y='BloodPressure', hue='혈압 수치', data=df_melted, ax=ax, palette='Set2')
        fig = barplot.get_figure()
        return fig
    st.pyplot(age_bp())
# 나이에 따른 혈당 수치 시각화(나이대로 분류하여 산점도)
    st.write('#### 혈당 수치 산점도')
    def age_bs():
        # 그래프 설정
        fig, ax = plt.subplots(figsize=(10, 6))
        # 산점도 그리기
        sns.scatterplot(data=df, x='Age', y='BS', hue='AgeGroup', palette='Set2', ax=ax)

        # 그래프 제목과 축 레이블 설정
        ax.set_title('나이에 따른 혈당 수치')
        ax.set_xlabel('연령')
        ax.set_ylabel('혈당 수치')
        return fig
    st.pyplot(age_bs())
# 나이에 따른 심박수(산점도 그래프)
    st.write('#### 심박수 산점도')
    def age_dt():
        fig, ax = plt.subplots(figsize = (10, 6))
        sns.scatterplot(data = df, x = 'Age', y = 'HeartRate', hue = 'AgeGroup', palette = 'deep', ax = ax)
        ax.set_title('나이에 따른 심박수')
        ax.set_xlabel('연령')
        ax.set_ylabel('심박수')
        return fig
    st.pyplot(age_dt())
# 나이에 따른 체온(도씨로 변경 필요, 막대 그래프(평균을 구해서))
    st.write('#### 체온 그래프')
    def age_hr():
        def fahrenheit_to_celsius(fahrenheit):
            return (fahrenheit - 32) * 5.0/9.0  
        df['BodyTemp_Celsius'] = df['BodyTemp'].apply(fahrenheit_to_celsius)
        hr_mean = df.groupby('AgeGroup')['BodyTemp_Celsius'].mean().reset_index()
        fig, ax = plt.subplots(figsize = (8, 6))
        plt.xlabel('나이')
        plt.ylabel('체온(℃)')
        ax.set_ylim(35, 38)
        barplot = sns.barplot(x='AgeGroup', y='BodyTemp_Celsius', data = hr_mean, ax=ax, palette='Set2')
        fig = barplot.get_figure()
        return fig
    st.write(age_hr())
# 총 다해서 위험도 분류

### 2번째 탭 ###
with risk:
    # 최고혈압, 최소혈압, 혈당 수치, 체온, 심박수를 입력받아서 위험도 분류
    risk_mapping = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
    df['RiskLevel'] = df['RiskLevel'].map(risk_mapping)
    X = df.drop(columns = ['RiskLevel'])
    y = df['RiskLevel']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    model = xgb.XGBClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 3, objective = 'multi:softmax', num_class = 3, random_state = 42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'모델 정확도: {accuracy:.2f}')
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'Mid Risk', 'High Risk']))