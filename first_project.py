import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# 한글 깨짐 방지
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 탭 분류
analyze, risk, risk2 = st.tabs(["위험도 분석 데이터", "위험도 분석 예측 모델", "위험도 분석하기"])
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
    # 'RiskLevel'을 숫자로 변환
    risk_mapping = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
    df['RiskLevel'] = df['RiskLevel'].map(risk_mapping)
    # AgeGroup을 숫자로 변환
    le = LabelEncoder()
    df['AgeGroup'] = le.fit_transform(df['AgeGroup'])
    # 머신러닝 데이터 준비
    X = df.drop(columns=['RiskLevel'])
    y = df['RiskLevel']
    # 학습 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # XGBoost 모델 생성
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        objective='multi:softmax',
        num_class=3,
        random_state=42
    )
    # 모델 학습
    model.fit(X_train, y_train)
    # 예측 및 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"모델 정확도: {accuracy:.2f}")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'Mid Risk', 'High Risk']))

    st.title("산모 건강 위험도 예측 시스템 🚑")

    # 모델 성능 평가
    st.subheader("모델 성능 평가")
    st.write(f"모델 정확도: {accuracy:.2f}")

    # Confusion Matrix 시각화
    st.write("#### 혼동 행렬 (Confusion Matrix)")
    conf_matrix = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Mid', 'High'], yticklabels=['Low', 'Mid', 'High'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

    # 특성 중요도(Feature Importance) 시각화
    st.subheader("주요 건강 지표의 중요도")

    importance = model.feature_importances_
    features = X.columns

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=importance, y=features, palette="viridis")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    st.pyplot(fig)

# 3번째 탭
with risk2:
    # 사용자가 입력한 건강 지표를 바탕으로 위험도 예측
    st.subheader("🤰 사용자 건강 정보 입력")

    age = st.slider("나이", 20, 60, 30)
    systolic_bp = st.slider("최고 혈압 (Systolic BP)", 90, 180, 120)
    diastolic_bp = st.slider("최저 혈압 (Diastolic BP)", 60, 120, 80)
    bs = st.slider("혈당 수치 (BS)", 3.0, 10.0, 5.5)
    body_temp = st.slider("체온 (섭씨)", 35.0, 40.0, 36.5)
    heart_rate = st.slider("심박수 (Heart Rate)", 50, 150, 80)

    # 연령대 변환 (입력된 나이에 따라 AgeGroup 설정)
    def get_age_group(age):
        if age < 30: return 0  # 20대
        elif age < 40: return 1  # 30대
        elif age < 50: return 2  # 40대
        elif age < 60: return 3  # 50대
        else: return 4  # 60대 이상
    age_group = get_age_group(age)
    # 입력 데이터를 모델 학습한 특성과 동일하게 변환
    input_data = pd.DataFrame([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate, age_group, (body_temp * 9/5) + 32]],
                            columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate', 'AgeGroup', 'BodyTemp_Celsius'])

    # 예측 수행
    prediction = model.predict(input_data)[0]
    risk_label = {0: "Low Risk 🟢", 1: "Mid Risk 🟡", 2: "High Risk 🔴"}

    # 결과 출력
    st.subheader("📌 예측된 위험도")
    st.write(f"### {risk_label[prediction]}")