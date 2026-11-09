import streamlit as st
import pandas as pd
import numpy as np
import shap
import altair as alt
import os
from xgboost import XGBRegressor

st.set_page_config(page_title="Dự đoán doanh thu quảng cáo", page_icon="📈", layout="wide")

SOURCE = "Kaggle: nudratabbas/global-ads-performance-google-meta-tiktok (CC0, tác giả Nudrat Abbas)"

st.markdown("""
<style>
.hero {
    background: linear-gradient(120deg, #6a11cb 0%, #2575fc 60%, #00c6ff 100%);
    padding: 28px 34px; border-radius: 18px; color: white; margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(37,117,252,0.30);
}
.hero h1 { color: white; margin: 0; font-size: 2.1rem; }
.hero p { color: rgba(255,255,255,0.92); margin: 6px 0 0 0; font-size: 1rem; }
.card { border-radius: 16px; padding: 20px 22px; color: white; box-shadow: 0 6px 18px rgba(0,0,0,0.25); }
.card .t { font-size: 0.9rem; opacity: 0.9; }
.card .v { font-size: 2rem; font-weight: 700; margin: 4px 0; }
.card .s { font-size: 0.85rem; opacity: 0.9; }
.c1 { background: linear-gradient(135deg, #11998e, #38ef7d); }
.c2 { background: linear-gradient(135deg, #f7971e, #ffd200); color: #3a2c00; }
.c3 { background: linear-gradient(135deg, #ee0979, #ff6a00); }
.stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📈 Dự đoán doanh thu chiến dịch quảng cáo</h1>
  <p>Mô hình XGBoost huấn luyện trên 1.800 chiến dịch, giải thích bằng SHAP</p>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_raw():
    d = pd.read_csv("global_ads_performance_dataset.csv")
    d["date"] = pd.to_datetime(d["date"])
    return d


@st.cache_resource
def load_model():
    df = load_raw().copy()
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    cats = {c: sorted(df[c].unique()) for c in ["platform", "campaign_type", "industry", "country"]}
    y = df["revenue"]
    X = df.drop(columns=["date", "revenue", "ROAS"])
    X = pd.get_dummies(X, columns=list(cats), drop_first=True)
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X, y)
    return model, X.columns, cats, float(df["ROAS"].mean())


raw = load_raw()
model, cols, cats, mean_roas = load_model()

def stat(col, label, value):
    col.markdown(
        f"<div style='font-size:0.85rem;opacity:0.7'>{label}</div>"
        f"<div style='font-size:1.4rem;font-weight:600;line-height:1.3'>{value}</div>",
        unsafe_allow_html=True,
    )


d1, d2, d3, d4 = st.columns(4)
stat(d1, "Số chiến dịch", f"{len(raw):,}")
stat(d2, "Thời gian", f"{raw['date'].min():%m/%Y} – {raw['date'].max():%m/%Y}")
stat(d3, "Nền tảng / Loại CD", f"{raw['platform'].nunique()} / {raw['campaign_type'].nunique()}")
stat(d4, "Ngành / Quốc gia", f"{raw['industry'].nunique()} / {raw['country'].nunique()}")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Dự đoán", "📊 So sánh mô hình", "🔍 SHAP tổng thể", "🗂️ Dữ liệu"])

with st.sidebar:
    st.header("⚙️ Thông tin chiến dịch")
    platform = st.selectbox("Nền tảng", cats["platform"])
    campaign_type = st.selectbox("Loại chiến dịch", cats["campaign_type"])
    industry = st.selectbox("Ngành", cats["industry"])
    country = st.selectbox("Quốc gia", cats["country"])
    month = st.slider("Tháng", 1, 12, 6)
    dayofweek = st.slider("Thứ trong tuần (0 = Thứ Hai)", 0, 6, 2)

    st.header("💰 Chỉ số hiệu suất")
    ad_spend = st.number_input("Chi phí quảng cáo", 50.0, 100000.0, 6000.0, step=100.0)
    impressions = st.number_input("Lượt hiển thị", 1000, 500000, 100000, step=1000)
    clicks = st.number_input("Lượt click", 10, 50000, 4000, step=100)
    conversions = st.number_input("Lượt chuyển đổi", 1, 5000, 180, step=10)

with tab1:
    if clicks > impressions:
        st.warning("Số click đang lớn hơn lượt hiển thị, hãy kiểm tra lại.")
    if conversions > clicks:
        st.warning("Số chuyển đổi đang lớn hơn số click, hãy kiểm tra lại.")

    ranges = {"ad_spend": ad_spend, "impressions": impressions, "clicks": clicks, "conversions": conversions}
    out = [k for k, v in ranges.items() if v < raw[k].min() or v > raw[k].max()]
    if out:
        st.warning("Giá trị nhập nằm ngoài vùng dữ liệu huấn luyện ("
                   + ", ".join(out) + "), dự đoán có thể kém tin cậy.")

    row = pd.DataFrame([{
        "platform": platform, "campaign_type": campaign_type,
        "industry": industry, "country": country,
        "impressions": impressions, "clicks": clicks,
        "CTR": clicks / impressions, "CPC": ad_spend / clicks,
        "ad_spend": ad_spend, "conversions": conversions,
        "CPA": ad_spend / conversions, "month": month, "dayofweek": dayofweek,
    }])
    row = pd.get_dummies(row, columns=["platform", "campaign_type", "industry", "country"])
    row = row.reindex(columns=cols, fill_value=0)

    pred = max(float(model.predict(row)[0]), 0.0)
    roas = pred / ad_spend
    cpa = ad_spend / conversions
    badge = "Cao hơn" if roas >= mean_roas else "Thấp hơn"

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="card c1"><div class="t">Doanh thu dự đoán</div>
        <div class="v">{pred:,.0f}</div><div class="s">Dựa trên thông số bạn nhập</div></div>""",
                unsafe_allow_html=True)
    c2.markdown(f"""<div class="card c2"><div class="t">ROAS dự kiến</div>
        <div class="v">{roas:.2f}</div>
        <div class="s">{badge} mức trung bình toàn bộ dữ liệu ({mean_roas:.2f})</div></div>""",
                unsafe_allow_html=True)
    c3.markdown(f"""<div class="card c3"><div class="t">Chi phí mỗi chuyển đổi (CPA)</div>
        <div class="v">{cpa:,.1f}</div><div class="s">Chi phí / lượt chuyển đổi</div></div>""",
                unsafe_allow_html=True)

    st.info("Độ chính xác mô hình trên tập kiểm tra: R² ≈ 0,66, sai số trung bình (MAE) ≈ 11.500 "
            "(doanh thu trung bình của dữ liệu khoảng 30.100).")

    st.markdown("### Yếu tố tác động đến dự đoán này")
    sv = shap.TreeExplainer(model).shap_values(row)[0]
    top = pd.Series(sv, index=cols).sort_values(key=abs, ascending=False).head(8)
    chart_df = top.rename("Tác động").reset_index()
    chart_df.columns = ["Biến", "Tác động"]
    chart_df["Chiều"] = np.where(chart_df["Tác động"] >= 0, "Tăng doanh thu", "Giảm doanh thu")

    chart = alt.Chart(chart_df).mark_bar(cornerRadiusEnd=6).encode(
        x=alt.X("Tác động:Q", title="Mức đóng góp vào doanh thu dự đoán"),
        y=alt.Y("Biến:N", sort="-x", title=None, axis=alt.Axis(labelLimit=250)),
        color=alt.Color("Chiều:N",
                        scale=alt.Scale(domain=["Tăng doanh thu", "Giảm doanh thu"],
                                        range=["#2ecc71", "#e74c3c"]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["Biến", alt.Tooltip("Tác động:Q", format=",.0f"), "Chiều"],
    ).properties(height=320)
    st.altair_chart(chart, width="stretch")
    st.caption("Màu xanh đẩy doanh thu dự đoán lên, màu đỏ kéo xuống so với mức nền của mô hình.")

with tab2:
    st.markdown("### Kết quả 5 mô hình (mục tiêu: revenue gốc)")
    if os.path.exists("ket_qua_mo_hinh.csv"):
        res = pd.read_csv("ket_qua_mo_hinh.csv")
        res = res[res["Mục tiêu"] == "revenue gốc"].drop(columns=["Mục tiêu"])
        st.dataframe(res, width="stretch", hide_index=True)
        bar = alt.Chart(res).mark_bar(cornerRadiusEnd=6).encode(
            x=alt.X("R2 (test):Q", scale=alt.Scale(domain=[0.5, 0.75])),
            y=alt.Y("Mô hình:N", sort="-x", title=None),
            color=alt.Color("Mô hình:N", legend=None, scale=alt.Scale(scheme="tableau10")),
            tooltip=["Mô hình", "R2 (test)", "RMSE", "MAE"],
        ).properties(height=260, title="R² trên tập test")
        st.altair_chart(bar, width="stretch")
    else:
        st.info("Chưa thấy file ket_qua_mo_hinh.csv trong thư mục. Hãy chạy buoc 3.py trước.")

with tab3:
    st.markdown("### Yếu tố quan trọng nhất trên toàn bộ dữ liệu")
    a, b = st.columns(2)
    if os.path.exists("shap_bar.png"):
        a.image("shap_bar.png", caption="Mức độ quan trọng trung bình", width="stretch")
    else:
        a.info("Chưa có shap_bar.png (chạy buoc 4.py).")
    if os.path.exists("shap_beeswarm.png"):
        b.image("shap_beeswarm.png", caption="Chiều tác động của từng biến", width="stretch")
    else:
        b.info("Chưa có shap_beeswarm.png (chạy buoc 4.py).")

with tab4:
    st.markdown("### Khám phá dữ liệu")
    st.caption(f"Nguồn: {SOURCE}")
    st.dataframe(raw.head(20), width="stretch", hide_index=True)

    m = raw.assign(Tháng=raw["date"].dt.to_period("M").astype(str)) \
           .groupby("Tháng", as_index=False)["revenue"].mean()
    line = alt.Chart(m).mark_line(point=True, color="#2575fc").encode(
        x=alt.X("Tháng:N", title=None),
        y=alt.Y("revenue:Q", title="Doanh thu trung bình"),
        tooltip=["Tháng", alt.Tooltip("revenue:Q", format=",.0f")],
    ).properties(height=280, title="Doanh thu trung bình theo tháng")
    st.altair_chart(line, width="stretch")

    st.markdown("**Khoảng giá trị trong dữ liệu**")
    st.dataframe(raw[["ad_spend", "impressions", "clicks", "conversions", "revenue"]]
                 .describe().loc[["min", "50%", "max"]].round(1), width="stretch")
    
