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
filtered = filtered.sort_values("연도").reset_index(drop=True)

n_years = len(filtered)
start_year = int(filtered["연도"].min())
end_year = int(filtered["연도"].max())

st.markdown(
    f"**회귀 직선 산출 기준:** 총 **{n_years}개** 연도 사용 "
    f"(시작 연도: **{start_year}년**, 끝 연도: **{end_year}년**)"
)

# ===== 전체 기간 회귀 =====
slope_all, intercept_all, r_all, p_all, se_all = stats.linregress(
    filtered["연도"], filtered["평균기온"]
)
r_squared_all = r_all ** 2
warming_per_100y_all = slope_all * 100

# ===== 최근 20년 회귀 =====
recent20 = filtered[filtered["연도"] >= (end_year - 19)].copy()
recent_start = int(recent20["연도"].min())
recent_end = int(recent20["연도"].max())
n_recent = len(recent20)

if n_recent >= 2:
    slope_recent, intercept_recent, r_recent, p_recent, se_recent = stats.linregress(
        recent20["연도"], recent20["평균기온"]
    )
    r_squared_recent = r_recent ** 2
    warming_per_100y_recent = slope_recent * 100
else:
    slope_recent = intercept_recent = r_recent = r_squared_recent = warming_per_100y_recent = None

# ===== 상관계수/결정계수 표시 =====
col1, col2, col3 = st.columns(3)
col1.metric("상관계수 (r) - 전체", f"{r_all:.4f}")
col2.metric("결정계수 (R²) - 전체", f"{r_squared_all:.4f}")
col3.metric("회귀식 - 전체", f"y = {slope_all:.4f}x + {intercept_all:.2f}")

st.divider()

# ===== 100년당 기온 상승 비교 (크게 표시) =====
st.subheader("🔥 100년당 기온 상승 비교")

comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 25px; background-color:#eaf2f8; border-radius:15px;">
            <h3 style="color:#2874a6; margin-bottom:5px;">전체 기간 ({start_year}~{end_year})</h3>
            <p style="color:#555; margin:0;">사용 연도 수: {n_years}개</p>
            <h1 style="font-size:60px; color:#1a5276; margin:10px 0;">
                {warming_per_100y_all:+.2f} °C
            </h1>
            <p style="color:#555; margin:0;">/ 100년</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with comp_col2:
    if warming_per_100y_recent is not None:
        st.markdown(
            f"""
            <div style="text-align:center; padding: 25px; background-color:#fdedec; border-radius:15px;">
                <h3 style="color:#c0392b; margin-bottom:5px;">최근 20년 ({recent_start}~{recent_end})</h3>
                <p style="color:#555; margin:0;">사용 연도 수: {n_recent}개</p>
                <h1 style="font-size:60px; color:#922b21; margin:10px 0;">
                    {warming_per_100y_recent:+.2f} °C
                </h1>
                <p style="color:#555; margin:0;">/ 100년</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("최근 20년 데이터가 회귀분석을 하기에 부족합니다.")

st.divider()

# ===== 슬라이더 (1900 ~ 2100) =====
st.subheader("📅 연도 선택하여 예상 기온 확인 (전체 기간 회귀 기준)")
selected_year = st.slider(
    "연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temp = slope_all * selected_year + intercept_all

st.markdown(
    f"""
    <div style="text-align:center; padding: 30px; background-color:#f0f2f6; border-radius:15px; margin-bottom:20px;">
        <h2 style="color:#555;">{selected_year}년 예상 평균기온</h2>
        <h1 style="font-size:80px; color:#e74c3c; margin:0;">{predicted_temp:.2f} °C</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ===== Plotly 그래프 =====
fig = go.Figure()

# 산점도 (실제 데이터, 최근 20년 강조)
fig.add_trace(go.Scatter(
    x=filtered["연도"],
    y=filtered["평균기온"],
    mode="markers",
    name="연도별 평균기온 (전체)",
    marker=dict(size=8, color="royalblue")
))

fig.add_trace(go.Scatter(
    x=recent20["연도"],
    y=recent20["평균기온"],
    mode="markers",
    name="최근 20년 데이터",
    marker=dict(size=10, color="darkorange", symbol="circle-open", line=dict(width=2))
))

# 전체 기간 회귀 직선
x_line_all = np.array([start_year, end_year])
y_line_all = slope_all * x_line_all + intercept_all
fig.add_trace(go.Scatter(
    x=x_line_all,
    y=y_line_all,
    mode="lines",
    name="회귀 직선 (전체 기간)",
    line=dict(color="blue", width=2)
))

# 최근 20년 회귀 직선
if slope_recent is not None:
    x_line_recent = np.array([recent_start, recent_end])
    y_line_recent = slope_recent * x_line_recent + intercept_recent
    fig.add_trace(go.Scatter(
        x=x_line_recent,
        y=y_line_recent,
        mode="lines",
        name="회귀 직선 (최근 20년)",
        line=dict(color="red", width=3, dash="dash")
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
    title=f"서울 연도별 평균기온 추세 비교",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ===== 상세 데이터 확인 =====
with st.expander("📊 사용된 연도별 데이터 보기"):
    st.dataframe(filtered)
