import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

st.set_page_config(page_title="Từ xác suất dự báo đến quyết định - HCMUTE", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
header {visibility: hidden;}
footer {visibility: hidden;}
.block-container {
    padding-top: 0.25rem !important;
    padding-bottom: 0.15rem !important;
    padding-left: 0.8rem !important;
    padding-right: 0.8rem !important;
    max-width: 100% !important;
}
div[data-testid="stVerticalBlock"] > div {
    gap: 0.18rem;
}
.main-title {
    font-size: 20px;
    font-weight: 800;
    color: #0f172a;
    border-left: 4px solid #2563eb;
    padding-left: 8px;
    margin-bottom: 2px;
}
.main-sub {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 6px;
    padding-left: 12px;
}
.card-header {
    font-size: 13px;
    font-weight: 700;
    color: #0f172a;
    border-left: 3px solid #2563eb;
    padding-left: 6px;
    margin-bottom: 2px;
}
.card-sub {
    font-size: 11px;
    color: #64748b;
    margin-bottom: 3px;
    padding-left: 9px;
}
.metric-tile {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 7px 4px;
    text-align: center;
}
.metric-lbl {
    font-size: 10.5px;
    font-weight: bold;
    color: #475569;
    text-transform: uppercase;
}
.metric-num {
    font-size: 18px;
    font-weight: 800;
    color: #0f172a;
}
.callout-box {
    background: #f0fdf4;
    border-left: 3px solid #16a34a;
    border-radius: 5px;
    padding: 7px 9px;
    font-size: 11.5px;
    color: #166534;
    line-height: 1.45;
}
button[kind="secondary"] {
    padding: 2px 4px !important;
    font-size: 11.5px !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return pd.read_csv('du_lieu_sinh_vien.csv')

df_data = get_data()
n_total = len(df_data)

@st.cache_resource
def train_model(df):
    X_mat = df[['gio_hoc_tuan', 'ty_le_chuyen_can', 'so_bai_tap', 'diem_giua_ky']]
    y_vec = df['qua_mon']
    X_tr, X_te, y_tr, y_te = train_test_split(X_mat, y_vec, test_size=0.2, random_state=42, stratify=y_vec)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    lr = LogisticRegression(random_state=42)
    lr.fit(X_tr_s, y_tr)
    y_pr_te = lr.predict_proba(X_te_s)[:, 1]
    th_range = np.linspace(0.02, 0.98, 70)
    c_precs, c_recs, c_f1s = [], [], []
    for th_val in th_range:
        yp = (y_pr_te >= th_val).astype(int)
        c_precs.append(precision_score(y_te, yp, zero_division=0))
        c_recs.append(recall_score(y_te, yp, zero_division=0))
        c_f1s.append(f1_score(y_te, yp, zero_division=0))
    return sc, lr, X_tr, X_te, y_tr, y_te, y_pr_te, th_range, c_precs, c_recs, c_f1s

scaler, model, X_train, X_test, y_train, y_test, y_prob_test, thresholds_range, curve_precs, curve_recs, curve_f1s = train_model(df_data)

if 'tau' not in st.session_state:
    st.session_state.tau = 0.50
if 'y_feature' not in st.session_state:
    st.session_state.y_feature = 'diem_giua_ky'

def set_tau_val(val):
    st.session_state.tau = float(val)

st.markdown('<div class="main-title">Từ xác suất dự báo đến quyết định</div>', unsafe_allow_html=True)
st.markdown(f'<div class="main-sub">Dữ liệu mô phỏng gồm {n_total} sinh viên (538 Đạt, 62 Rớt), phục vụ minh họa Logistic Regression và tác động của ngưỡng phân loại; không đại diện cho hiệu quả dự kiến trên sinh viên thực tế.</div>', unsafe_allow_html=True)

col_charts, col_ctrl = st.columns([2.2, 1.0])

with col_ctrl:
    st.markdown("<div style='font-size:13px; font-weight:800; color:#1e3a8a; margin-bottom:2px;'>ĐIỀU CHỈNH NGƯỠNG</div>", unsafe_allow_html=True)

    tau = st.session_state.tau

    st.markdown(f"<div style='font-size:11.5px; font-weight:bold; color:#475569;'>Ngưỡng phân loại (Threshold τ) <span style='float:right; background:#dbeafe; color:#1e40af; padding:1px 6px; border-radius:3px; font-size:12px;'>τ = {tau:.2f}</span></div>", unsafe_allow_html=True)
    st.slider("Ngưỡng phân loại (Threshold τ)", min_value=0.05, max_value=0.95, step=0.01, key="tau", label_visibility="collapsed")

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        st.button("0.30 · Dễ dự báo Đạt", on_click=set_tau_val, args=(0.30,), width="stretch")
    with c_b2:
        st.button("0.50 · Ngưỡng mặc định", on_click=set_tau_val, args=(0.50,), width="stretch")
            
    c_b3, c_b4 = st.columns(2)
    with c_b3:
        st.button("0.85 · Ưu tiên phát hiện Rớt", on_click=set_tau_val, args=(0.85,), width="stretch")
    with c_b4:
        st.button("Đặt lại", on_click=set_tau_val, args=(0.50,), width="stretch")

    y_axis_choice = st.selectbox(f"TRỤC Y VÙNG DỮ LIỆU ({n_total} SV)", ["Điểm giữa kỳ (0-10)", "Tỷ lệ chuyên cần (%)", "Số bài tập (0-10)"], label_visibility="visible")
    y_map = {
        "Điểm giữa kỳ (0-10)": "diem_giua_ky",
        "Tỷ lệ chuyên cần (%)": "ty_le_chuyen_can",
        "Số bài tập (0-10)": "so_bai_tap"
    }
    st.session_state.y_feature = y_map[y_axis_choice]

    eval_y_pred = (y_prob_test >= tau).astype(int)
    cm_eval = confusion_matrix(y_test, eval_y_pred)
    tn_e, fp_e, fn_e, tp_e = cm_eval.ravel()
    n_test = len(y_test)

    rec_dat = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0.0
    rec_rot = tn_e / (tn_e + fp_e) if (tn_e + fp_e) > 0 else 0.0
    bal_acc = (rec_dat + rec_rot) / 2.0
    f1_dat = f1_score(y_test, eval_y_pred, zero_division=0)
    f1_rot = f1_score(y_test, eval_y_pred, pos_label=0, zero_division=0)
    macro_f1 = f1_score(y_test, eval_y_pred, average='macro', zero_division=0)
    acc_val = accuracy_score(y_test, eval_y_pred)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-lbl">RECALL LỚP ĐẠT</div>
            <div class="metric-num">{rec_dat*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-lbl">RECALL LỚP RỚT</div>
            <div class="metric-num">{rec_rot*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    m3, m4 = st.columns(2)
    with m3:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-lbl">MACRO F1</div>
            <div class="metric-num">{macro_f1*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-lbl">BALANCED ACCURACY</div>
            <div class="metric-num">{bal_acc*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    pass_real = int(np.sum(y_test == 1))
    fail_real = int(np.sum(y_test == 0))
    pred_pass = int(np.sum(eval_y_pred == 1))

    st.markdown(f"""
    <div class="callout-box" style="margin-top:5px;">
        <b>ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST (N = {n_test}):</b><br>
        • Thực tế: <b>{pass_real} Đạt</b> ({(pass_real/n_test)*100:.1f}%) | <b>{fail_real} Rớt</b> ({(fail_real/n_test)*100:.1f}%)<br>
        • Dự báo tại τ={tau:.2f}: <b>{pred_pass} Đạt</b> | <b>{n_test - pred_pass} Rớt</b><br>
        • Độ chính xác (Accuracy): <b>{acc_val*100:.1f}%</b> ({tp_e + tn_e}/{n_test} ca đúng)<br>
        • F1 lớp Rớt: <b>{f1_rot*100:.1f}%</b> | F1 lớp Đạt: <b>{f1_dat*100:.1f}%</b><br>
        • Nhận diện sinh viên nguy cơ: Bắt trúng <b>{tn_e}/{fail_real} SV rớt</b> (Recall Rớt: {rec_rot*100:.1f}%).<br>
        <i>* Lưu ý: Recall lớp Rớt được tính trên {fail_real} trường hợp, vì vậy chỉ số có thể dao động đáng kể với dữ liệu mới.</i>
    </div>
    """, unsafe_allow_html=True)

with col_charts:
    r1_c1, r1_c2 = st.columns(2)
    
    with r1_c1:
        st.markdown(f"<div class='card-header'>Phân bố toàn bộ {n_total} sinh viên: Giờ học vs {y_axis_choice}</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-sub'>* Đường biên là lát cắt 2D, hai biến còn lại được cố định tại giá trị trung bình.</div>", unsafe_allow_html=True)
        
        y_feat_col = st.session_state.y_feature
        x_col = 'gio_hoc_tuan'
        
        c_real_pass = df_data[df_data['qua_mon'] == 1]
        c_real_fail = df_data[df_data['qua_mon'] == 0]
        
        fig_2d = go.Figure()
        fig_2d.add_trace(go.Scatter(
            x=c_real_pass[x_col], y=c_real_pass[y_feat_col], mode='markers',
            marker=dict(size=5.5, color='#16a34a', opacity=0.7),
            name=f'Thực tế qua môn ({len(c_real_pass)})'
        ))
        fig_2d.add_trace(go.Scatter(
            x=c_real_fail[x_col], y=c_real_fail[y_feat_col], mode='markers',
            marker=dict(size=6.5, color='#dc2626', opacity=0.85, symbol='x'),
            name=f'Thực tế rớt môn ({len(c_real_fail)})'
        ))
        
        feat_idx_x = 0
        feat_cols = ['gio_hoc_tuan', 'ty_le_chuyen_can', 'so_bai_tap', 'diem_giua_ky']
        feat_idx_y = feat_cols.index(y_feat_col)
        w = model.coef_[0]
        b0 = model.intercept_[0]
        m_s = scaler.mean_
        s_s = scaler.scale_
        
        logit_tau = np.log(tau / (1.0 - tau + 1e-9))
        x_seq = np.linspace(0.0, 20.0, 60)
        
        mean_fixed_sum = 0.0
        for i_f, col_f in enumerate(feat_cols):
            if i_f != feat_idx_x and i_f != feat_idx_y:
                mean_val = df_data[col_f].mean()
                mean_fixed_sum += w[i_f] * (mean_val - m_s[i_f]) / s_s[i_f]
                
        y_bound_analytical = m_s[feat_idx_y] + s_s[feat_idx_y] * (
            (logit_tau - b0 - mean_fixed_sum - w[feat_idx_x] * (x_seq - m_s[feat_idx_x]) / s_s[feat_idx_x]) / w[feat_idx_y]
        )
        
        fig_2d.add_trace(go.Scatter(
            x=x_seq, y=y_bound_analytical, mode='lines',
            line=dict(color='#2563eb', width=2.5),
            name=f'Ranh giới phân loại p = {tau:.2f}'
        ))
        
        y_max_limit = 10.5 if y_feat_col in ['diem_giua_ky', 'so_bai_tap'] else 102.0
        fig_2d.update_layout(
            height=265, margin=dict(l=30, r=10, t=10, b=25),
            xaxis=dict(title="Giờ học/tuần", range=[-0.5, 20.5], showgrid=True, gridcolor='#f1f5f9', zeroline=False),
            yaxis=dict(title=y_axis_choice, range=[-0.5, y_max_limit], showgrid=True, gridcolor='#f1f5f9', zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11))
        )
        st.plotly_chart(fig_2d, width="stretch", key="fig_space_2d")

    with r1_c2:
        st.markdown(f"<div class='card-header'>Ma trận nhầm lẫn trên tập test, n = {n_test} (τ = {tau:.2f})</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-sub'>* Phân loại dương tính: p ≥ τ (Đạt); Âm tính: p &lt; τ (Rớt). Tập test có 12 ca Rớt.</div>", unsafe_allow_html=True)
        tn_pct = (tn_e / n_test) * 100
        fp_pct = (fp_e / n_test) * 100
        fn_pct = (fn_e / n_test) * 100
        tp_pct = (tp_e / n_test) * 100

        cm_box_html = f"""
        <div style='display:flex; flex-direction:column; justify-content:center; height:265px;'>
            <div style='display:grid; grid-template-columns: 85px 1fr 1fr; gap:4px; text-align:center;'>
                <div></div>
                <div style='font-size:12px; font-weight:bold; color:#1e293b; padding:3px;'>Dự báo 0 (Rớt)</div>
                <div style='font-size:12px; font-weight:bold; color:#1e293b; padding:3px;'>Dự báo 1 (Đạt)</div>
                <div style='font-size:12px; font-weight:bold; color:#1e293b; display:flex; align-items:center; justify-content:flex-end; padding-right:6px;'>Thực tế 0</div>
                <div style='background:#dcfce7; border-radius:6px; padding:12px 4px; border:1px solid #bbf7d0;'>
                    <div style='font-size:12px; font-weight:bold; color:#166534;'>TN (Đúng Rớt)</div>
                    <div style='font-size:24px; font-weight:800; color:#166534; margin:2px 0;'>{tn_e}</div>
                    <div style='font-size:11px; color:#15803d;'>{tn_pct:.1f}% · Dự: 0 | Thực: 0</div>
                </div>
                <div style='background:#fee2e2; border-radius:6px; padding:12px 4px; border:1px solid #fecaca;'>
                    <div style='font-size:12px; font-weight:bold; color:#991b1b;'>FP (Dương tính giả)</div>
                    <div style='font-size:24px; font-weight:800; color:#991b1b; margin:2px 0;'>{fp_e}</div>
                    <div style='font-size:11px; color:#b91c1c;'>{fp_pct:.1f}% · Dự: 1 | Thực: 0</div>
                </div>
                <div style='font-size:12px; font-weight:bold; color:#1e293b; display:flex; align-items:center; justify-content:flex-end; padding-right:6px;'>Thực tế 1</div>
                <div style='background:#fee2e2; border-radius:6px; padding:12px 4px; border:1px solid #fecaca;'>
                    <div style='font-size:12px; font-weight:bold; color:#991b1b;'>FN (Âm tính giả)</div>
                    <div style='font-size:24px; font-weight:800; color:#991b1b; margin:2px 0;'>{fn_e}</div>
                    <div style='font-size:11px; color:#b91c1c;'>{fn_pct:.1f}% · Dự: 0 | Thực: 1</div>
                </div>
                <div style='background:#dcfce7; border-radius:6px; padding:12px 4px; border:1px solid #bbf7d0;'>
                    <div style='font-size:12px; font-weight:bold; color:#166534;'>TP (Đúng Qua môn)</div>
                    <div style='font-size:24px; font-weight:800; color:#166534; margin:2px 0;'>{tp_e}</div>
                    <div style='font-size:11px; color:#15803d;'>{tp_pct:.1f}% · Dự: 1 | Thực: 1</div>
                </div>
            </div>
        </div>
        """
        st.markdown(cm_box_html, unsafe_allow_html=True)

    r2_c1, r2_c2 = st.columns(2)
    
    with r2_c1:
        st.markdown(f"<div class='card-header'>Đánh đổi Precision – Recall theo ngưỡng (Tập test, N = {n_test})</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-sub'>* Đường biểu diễn các chỉ số tính trên lớp Đạt theo biến thiên ngưỡng τ.</div>", unsafe_allow_html=True)
        fig_tradeoff = go.Figure()
        fig_tradeoff.add_trace(go.Scatter(
            x=thresholds_range, y=curve_precs, mode='lines',
            line=dict(color='#2563eb', width=2.5),
            name='Precision lớp Đạt'
        ))
        fig_tradeoff.add_trace(go.Scatter(
            x=thresholds_range, y=curve_recs, mode='lines',
            line=dict(color='#dc2626', width=2.5),
            name='Recall lớp Đạt'
        ))
        fig_tradeoff.add_trace(go.Scatter(
            x=thresholds_range, y=curve_f1s, mode='lines',
            line=dict(color='#7c3aed', width=2, dash='dot'),
            name='F1 lớp Đạt'
        ))
        
        cur_p_test = precision_score(y_test, eval_y_pred, zero_division=0)
        fig_tradeoff.add_vline(x=tau, line_dash="dash", line_color="#334155", line_width=1.5)
        fig_tradeoff.add_annotation(
            x=tau, y=0.52, text=f"τ = {tau:.2f}<br>Precision Đạt = {cur_p_test*100:.1f}%<br>Recall Đạt = {rec_dat*100:.1f}%<br>F1 Đạt = {f1_dat*100:.1f}%",
            showarrow=True, arrowhead=1, font=dict(size=11, color="#0f172a"),
            bgcolor="#ffffff", bordercolor="#cbd5e1"
        )
        
        fig_tradeoff.update_layout(
            height=265, margin=dict(l=30, r=10, t=10, b=25),
            xaxis=dict(title="Ngưỡng quyết định (Threshold τ)", range=[0, 1], showgrid=True, gridcolor='#f1f5f9'),
            yaxis=dict(title="Chỉ số (0–1)", range=[0, 1.05], showgrid=True, gridcolor='#f1f5f9'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11))
        )
        st.plotly_chart(fig_tradeoff, width="stretch", key="fig_tradeoff_chart")

    with r2_c2:
        st.markdown("<div class='card-header'>Hệ số Logistic chuẩn hóa (β) và tỷ số chênh (OR)</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-sub'>* OR biểu thị thay đổi Odds khi đặc trưng tăng 1 độ lệch chuẩn (dữ liệu đã chuẩn hóa).</div>", unsafe_allow_html=True)
        coefs = model.coef_[0]
        ors = np.exp(coefs)
        var_names = ['Giờ học/tuần', 'Chuyên cần', 'Bài tập', 'Điểm giữa kỳ']
        
        order_b = np.argsort(coefs)
        sorted_coefs = coefs[order_b]
        sorted_ors = ors[order_b]
        sorted_names = [var_names[i] for i in order_b]
        
        bar_colors = ['#2563eb' if c >= 0 else '#dc2626' for c in sorted_coefs]
        bar_texts = [f"OR = {o:.2f} (β = {'+' if c>=0 else ''}{c:.2f})" for o, c in zip(sorted_ors, sorted_coefs)]
        
        fig_coef = go.Figure()
        fig_coef.add_trace(go.Bar(
            y=sorted_names, x=sorted_coefs, orientation='h',
            marker=dict(color=bar_colors),
            text=bar_texts, textposition='outside',
            textfont=dict(size=11),
            cliponaxis=False
        ))
        fig_coef.add_vline(x=0, line_color="#475569", line_width=1)
        max_c = max(sorted_coefs)
        fig_coef.update_layout(
            height=265, margin=dict(l=85, r=180, t=10, b=25),
            xaxis=dict(title="Hệ số hồi quy chuẩn hóa (β)", range=[0, max_c * 1.55], showgrid=True, gridcolor='#f1f5f9'),
            yaxis=dict(showgrid=False),
            showlegend=False
        )
        st.plotly_chart(fig_coef, width="stretch", key="fig_coef_chart")
