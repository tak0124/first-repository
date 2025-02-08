import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# 데이터 파일 경로
normal_data_path = '/Users/gyohyeon/Downloads/normal_data.csv'
outlier_data_path = '/Users/gyohyeon/Downloads/outlier_data.csv'

# 파일 읽기
normal_data = pd.read_csv(normal_data_path, encoding='utf-8')
outlier_data = pd.read_csv(outlier_data_path, encoding='utf-8')

# 데이터 요약
normal_data_summary = normal_data.describe().T
outlier_data_summary = outlier_data.describe().T

# 연속형 변수
normal_num = normal_data[['DV_R', 'DA_R', 'AV_R', 'AA_R', 'PM_R']]
outlier_num = outlier_data[['DV_R', 'DA_R', 'AV_R', 'AA_R', 'PM_R']]

# Streamlit 앱 구성
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

st.title("Data Analysis Dashboard")

# PCA Biplot 섹션
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.header("3D Biplot of PCA (Normal Data)")

        # PCA 모델 생성
        pca = PCA(n_components=3)
        pca.fit(normal_num)

        # PCA로 변환된 주성분
        components = pca.transform(normal_num)

        # 주성분의 가중치
        scale_factor = 20  # 벡터의 길이를 늘리기 위한 배수
        weights = pca.components_.T * np.sqrt(pca.explained_variance_) * scale_factor

        # Biplot 시각화
        fig = px.scatter_3d(
            x=components[:, 0], y=components[:, 1], z=components[:, 2], 
            title='3D Biplot of PCA (Normal Data)',
            opacity=0.2,  # 불투명도를 낮춰서 벡터가 더 잘 보이도록 설정
            color_discrete_sequence=['blue']
        )

        # 벡터 추가
        for i, (weight1, weight2, weight3) in enumerate(weights):
            # 벡터 본체
            fig.add_trace(
                go.Scatter3d(
                    x=[0, weight1],
                    y=[0, weight2],
                    z=[0, weight3],
                    mode='lines',
                    line=dict(color='red', width=4)
                )
            )
            # 화살표 효과를 위한 작은 선분
            fig.add_trace(
                go.Scatter3d(
                    x=[weight1, weight1 * 1.1],
                    y=[weight2, weight2 * 1.1],
                    z=[weight3, weight3 * 1.1],
                    mode='lines',
                    line=dict(color='red', width=4)
                )
            )
            # 끝점에 텍스트 추가
            fig.add_trace(
                go.Scatter3d(
                    x=[weight1 * 1.1],
                    y=[weight2 * 1.1],
                    z=[weight3 * 1.1],
                    mode='text',
                    text=[normal_num.columns[i]],
                    textposition='top center',
                    textfont=dict(color='red', size=12)
                )
            )

        fig.update_traces(marker=dict(size=5, color='blue', symbol='circle'), selector=dict(mode='markers'))

        fig.update_layout(
                scene=dict(
                    xaxis_title="PC1",
                    yaxis_title="PC2",
                    zaxis_title="PC3"
                ),
                showlegend=False,
                # 초기 카메라 설정 추가
                scene_camera=dict(
                    eye=dict(x=0.5, y=0.5, z=0.5)
                )
            )

        st.plotly_chart(fig)

    with col2:
        st.header("3D Biplot of PCA (Outlier Data)")

        # PCA 모델 생성
        pca = PCA(n_components=3)
        pca.fit(outlier_num)

        # PCA로 변환된 주성분
        components = pca.transform(outlier_num)

        # 주성분의 가중치
        scale_factor = 0.7  # 벡터의 길이를 늘리기 위한 배수
        weights = pca.components_.T * np.sqrt(pca.explained_variance_) * scale_factor

        # Biplot 시각화
        fig = px.scatter_3d(
            x=components[:, 0], y=components[:, 1], z=components[:, 2], 
            title='3D Biplot of PCA (Outlier Data)',
            opacity=0.2,  # 불투명도를 낮춰서 벡터가 더 잘 보이도록 설정
            color_discrete_sequence=['blue']
        )

        # 벡터 추가
        for i, (weight1, weight2, weight3) in enumerate(weights):
            # 벡터 본체
            fig.add_trace(
                go.Scatter3d(
                    x=[0, weight1],
                    y=[0, weight2],
                    z=[0, weight3],
                    mode='lines',
                    line=dict(color='red', width=4)
                )
            )
            # 화살표 효과를 위한 작은 선분
            fig.add_trace(
                go.Scatter3d(
                    x=[weight1, weight1 * 1.1],
                    y=[weight2, weight2 * 1.1],
                    z=[weight3, weight3 * 1.1],
                    mode='lines',
                    line=dict(color='red', width=4)
                )
            )
            # 끝점에 텍스트 추가
            fig.add_trace(
                go.Scatter3d(
                    x=[weight1 * 1.1],
                    y=[weight2 * 1.1],
                    z=[weight3 * 1.1],
                    mode='text',
                    text=[outlier_num.columns[i]],
                    textposition='top center',
                    textfont=dict(color='red', size=12)
                )
            )

        fig.update_traces(marker=dict(size=5, color='blue', symbol='circle'), selector=dict(mode='markers'))

    
        
        fig.update_layout(
                scene=dict(
                    xaxis_title="PC1",
                    yaxis_title="PC2",
                    zaxis_title="PC3"
                ),
                showlegend=False,
                # 초기 카메라 설정 추가
                scene_camera=dict(
                    eye=dict(x=0.3, y=0.3, z=0.3)
                )
            )

        st.plotly_chart(fig)




# 데이터 요약 섹션
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.header("Normal Data Summary")
        st.dataframe(normal_data_summary)

    with col2:
        st.header("Outlier Data Summary")
        st.dataframe(outlier_data_summary)

# 연속형 변수 시각화 섹션
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        selected_var_normal = st.selectbox("Select a variable to plot (Normal Data)", normal_num.columns.tolist())
        if selected_var_normal:
            st.subheader(f'Normal Data: {selected_var_normal}')
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.lineplot(x=normal_num.index, y=normal_num[selected_var_normal], ax=ax)
            ax.set_title(f'{selected_var_normal} over Index (Normal Data)')
            ax.set_xlabel('Index')
            ax.set_ylabel(selected_var_normal)
            st.pyplot(fig)

    with col2:
        selected_var_outlier = st.selectbox("Select a variable to plot (Outlier Data)", outlier_num.columns.tolist())
        if selected_var_outlier:
            st.subheader(f'Outlier Data: {selected_var_outlier}')
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.lineplot(x=outlier_num.index, y=outlier_num[selected_var_outlier], ax=ax)
            ax.set_title(f'{selected_var_outlier} over Index (Outlier Data)')
            ax.set_xlabel('Index')
            ax.set_ylabel(selected_var_outlier)
            st.pyplot(fig)

# 데이터 불균형 시각화 섹션
with st.container():
    st.header("Data Imbalance Visualization")
    col1, col2 = st.columns(2)
    with col1:
        # 데이터 길이 확인
        normal_data_len = len(normal_data)
        outlier_data_len = len(outlier_data)

        # 데이터 불균형 시각화
        labels = ['Normal Data', 'Outlier Data']
        sizes = [normal_data_len, outlier_data_len]
        colors = ['skyblue', 'lightcoral']
        explode = (0.1, 0)  # 첫 번째 조각을 약간 띄우기

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=140)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

    # 상관관계 행렬 시각화 섹션
    with col2:
        st.header("Correlation Matrix Visualization")

        # 상관관계 행렬 계산
        correlation_matrix_normal = normal_num.corr()
        correlation_matrix_outlier = outlier_num.corr()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # Normal data 상관관계 행렬 시각화
        sns.heatmap(correlation_matrix_normal, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8}, linewidths=0.5, ax=ax[0])
        ax[0].set_title('Correlation Matrix (Normal Data)')

        # Outlier data 상관관계 행렬 시각화
        sns.heatmap(correlation_matrix_outlier, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8}, linewidths=0.5, ax=ax[1])
        ax[1].set_title('Correlation Matrix (Outlier Data)')

        plt.tight_layout()
        st.pyplot(fig)

