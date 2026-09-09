import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="기온 예측기", layout="wide")

st.title("🌡️ 서울 기온 예측기")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    return df

df = load_data()

# 연도별 집계: 평균기온 평균, 관측일 수
yearly = df.groupby("연도").agg(
    평균기온=("평균기온", "mean"),
    관측일수=("평균기온", "count")
).reset_index()

# 필터링: 2025년까지, 관측일 300일 이상
BASE_YEAR = 2025
filtered = yearly[(yearly["연도"] <= BASE_YEAR) & (yearly["관측일수"] >= 300)].copy()
filtered = filtered.sort_values("연도")

n_years = len(filtered)
start_year = int(filtered["연도"].min())
end_year = int(filtered["연도"].max())

st.markdown(
    f"**회귀 직선 산출 기준:** 총 **{n_years}개** 연도 사용 "
    f"(시작 연도: **{start_year}년**, 끝 연도: **{end_year}년**)"
)

# 선형 회귀
slope, intercept, r_value, p_value, std_err = stats.linregress(
    filtered["연도"], filtered["평균기온"]
)
r_squared = r_value ** 2

col1, col2, col3 = st.columns(3)
col1.metric("상관계수 (r)", f"{r_value:.4f}")
col2.metric("결정계수 (R²)", f"{r_squared:.4f}")
col3.metric("회귀식", f"y = {slope:.4f}x + {intercept:.2f}")

# 슬라이더 (1900 ~ 2100)
st.subheader("📅 연도 선택하여 예상 기온 확인")
selected_year = st.slider(
    "연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temp = slope * selected_year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding: 30px; background-color:#f0f2f6; border-radius:15px; margin-bottom:20px;">
        <h2 style="color:#555;">{selected_year}년 예상 평균기온</h2>
        <h1 style="font-size:80px; color:#e74c3c; margin:0;">{predicted_temp:.2f} °C</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Plotly 그래프
fig = go.Figure()

# 산점도 (실제 데이터)
fig.add_trace(go.Scatter(
    x=filtered["연도"],
    y=filtered["평균기온"],
    mode="markers",
    name="연도별 평균기온 (실측)",
    marker=dict(size=8, color="royalblue")
))

# 회귀 직선 (데이터 범위)
x_line = np.array([start_year, end_year])
y_line = slope * x_line + intercept
fig.add_trace(go.Scatter(
    x=x_line,
    y=y_line,
    mode="lines",
    name="회귀 직선",
    line=dict(color="red", width=2)
))

# 선택한 연도 예측점 강조
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode="markers",
    name=f"{selected_year}년 예측",
    marker=dict(size=16, color="gold", symbol="star", line=dict(width=2, color="black"))
))

fig.update_layout(
    title=f"서울 연도별 평균기온 추세 ({start_year}~{end_year}년 기준)",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# 상세 데이터 확인
with st.expander("📊 사용된 연도별 데이터 보기"):
    st.dataframe(filtered.reset_index(drop=True))
