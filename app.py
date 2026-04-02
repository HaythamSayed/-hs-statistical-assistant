import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import tempfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HS-Statistical Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0f4ff 0%, #e8eeff 100%);
    border-right: 1px solid #c7d2fe;
}
.stButton > button {
    background: linear-gradient(90deg,#2563eb,#1d4ed8) !important;
    color: white !important;
    font-size: 15px !important;
    font-weight: bold !important;
    padding: 10px !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 10px rgba(37,99,235,0.3) !important;
}
.stat-card {
    background: white;
    border-radius: 14px;
    padding: 20px 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    transition: all 0.25s ease;
    margin-bottom: 14px;
    height: 100%;
}
.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.12);
    border-color: #93c5fd;
}
.card-icon  { font-size: 2rem; margin-bottom: 6px; }
.card-title { font-size: 1rem; font-weight: 700; color: #1e3a5f; margin-bottom: 4px; }
.card-desc  { font-size: 0.82rem; color: #6b7280; line-height: 1.5; }
div[data-testid="metric-container"] {
    background: #f0f4ff;
    border-radius: 10px;
    padding: 10px 14px;
    border-left: 4px solid #2563eb;
}
h3 { color: #1e3a5f; }
h4 { color: #2563eb; margin-top: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PDF HELPER
# ─────────────────────────────────────────────
def generate_pdf(title: str, report_text: str, figures=None) -> io.BytesIO:
    buf = io.BytesIO()
    styles = getSampleStyleSheet()
    code_style = ParagraphStyle("CodeStyle", parent=styles["Normal"],
                                fontName="Courier", fontSize=8, leading=10)
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12),
             Preformatted(report_text, code_style)]
    if figures:
        for fig in figures:
            story.append(Spacer(1, 16))
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
            tmp.close()
            story.append(Image(tmp.name, width=460, height=300))
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=36, rightMargin=36,
                            topMargin=36, bottomMargin=36)
    doc.build(story)
    buf.seek(0)
    return buf

def pdf_download_button(title, report, figs=None, filename="report.pdf"):
    pdf_buf = generate_pdf(title, report, figs)
    st.download_button(label="📥  Download PDF Report", data=pdf_buf,
                       file_name=filename, mime="application/pdf",
                       use_container_width=True)

# ─────────────────────────────────────────────
# INITIALISE df
# ─────────────────────────────────────────────
df: pd.DataFrame = st.session_state.get("df", None)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#1e3a8a,#2563eb);
            padding:22px 28px; border-radius:16px; color:white;
            margin-bottom:22px; box-shadow:0 4px 18px rgba(0,0,0,0.22);">
  <h1 style="margin:0; font-size:1.9rem;">📊 HS-Statistical Assistant</h1>
  <p style="margin:6px 0 0; font-size:1rem; opacity:.88;">
      Professional Statistical Analysis Platform for Researchers &amp; Academics
  </p>
  <p style="margin:4px 0 0; font-size:.88rem; opacity:.75;">
      Developed by <b>Haytham Saleh</b> &nbsp;|&nbsp;
      DBA Candidate · MBA · MSc · MCTS · MCSE · SAS
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂  Load Data")
    uploaded_file = st.file_uploader("Upload Excel File (.xlsx / .xls)",
                                     type=["xlsx", "xls"])

    if uploaded_file:
        try:
            st.session_state["df"] = pd.read_excel(uploaded_file)
            df = st.session_state["df"]
            st.success(f"✅ {df.shape[0]} rows × {df.shape[1]} columns")
            c1, c2 = st.columns(2)
            c1.metric("Rows",    df.shape[0])
            c2.metric("Columns", df.shape[1])
            c3, c4 = st.columns(2)
            c3.metric("Missing", int(df.isna().sum().sum()))
            c4.metric("Numeric", df.select_dtypes(include=np.number).shape[1])
        except Exception as exc:
            st.error(f"Error reading file: {exc}")

    df = st.session_state.get("df", None)

    if df is not None:
        all_cols = list(df.columns)
        st.markdown("---")
        st.markdown("## 📋  Variables")
        selected_vars = st.multiselect("Independent / Analysis Variables", all_cols)
        dep_var = st.selectbox("Dependent Variable  (Y)", [""] + all_cols)
        st.markdown("---")
        st.markdown("## 🔬  Analysis")
        ANALYSES = [
            "─── Descriptive ───",
            "Descriptive Statistics",
            "Frequency Analysis",
            "─── Normality ───",
            "Normality Tests",
            "─── Graphs ───",
            "Histogram",
            "Scatter Plot",
            "Boxplot",
            "─── Scale & Validity ───",
            "Reliability  (Cronbach Alpha)",
            "Factor Analysis",
            "─── Correlation ───",
            "Correlation Analysis",
            "Multicollinearity  (VIF)",
            "─── Comparative ───",
            "T-Test",
            "Chi-Square",
            "ANOVA  (One-Way)",
            "─── Regression ───",
            "Simple Regression",
            "Multiple Regression",
            "Logistic Regression",
            "─── Mediation ───",
            "Mediation Analysis",
            "─── Data Tools ───",
            "Composite Variable",
        ]
        analysis = st.selectbox("Select Analysis", ANALYSES)
        run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)
    else:
        selected_vars, dep_var, analysis, run_btn = [], "", None, False
        st.info("📁 Please upload an Excel file to begin")

    st.markdown("---")
    st.markdown("""
<div style='text-align:center; font-size:.78rem; color:#64748b; line-height:2;'>
  <b>Haytham Saleh</b><br>
  📞 <a href="https://wa.me/201001693305" target="_blank" style="color:#2563eb;">+20 100 169 3305</a><br>
  <a href="https://www.linkedin.com/in/haytham-saleh-0b407226/" target="_blank"
     style="display:inline-flex; align-items:center; gap:5px; background:#0a66c2;
            color:white; padding:5px 12px; border-radius:6px; text-decoration:none;
            font-weight:bold; font-size:.8rem; margin-top:4px;">
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24"
         fill="white" style="vertical-align:middle;">
      <path d="M20.447 20.452H17.21v-5.569c0-1.328-.027-3.037-1.852-3.037
               -1.853 0-2.136 1.445-2.136 2.939v5.667H9.987V9h3.102v1.561h.044
               c.432-.818 1.487-1.681 3.059-1.681 3.271 0 3.874 2.152 3.874 4.95v6.622z
               M5.337 7.433a1.8 1.8 0 1 1 0-3.6 1.8 1.8 0 0 1 0 3.6z
               M6.95 20.452H3.723V9H6.95v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729
               v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271
               V1.729C24 .774 23.2 0 22.222 0h.003z"/>
    </svg>
    LinkedIn Profile
  </a><br>
  <span style="font-size:.72rem; color:#94a3b8; margin-top:4px; display:block;">
    © 2026 All Rights Reserved
  </span>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# WELCOME SCREEN
# ─────────────────────────────────────────────
if df is None:
    st.markdown("""
<div style="background:#eff6ff; border-left:5px solid #2563eb;
            padding:14px 20px; border-radius:10px; margin-bottom:20px;">
  <h3 style="margin:0; color:#1e3a8a;">👋 Welcome!</h3>
  <p style="margin:6px 0 0; color:#374151;">
      Upload your <b>Excel file</b> from the left sidebar to start
      analysing your survey&nbsp;/&nbsp;questionnaire data instantly.
      All results can be downloaded as <b>PDF reports</b>.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 🗂️ Available Analyses")

    CARDS = [
        ("📊", "Descriptive Statistics",  "Mean, Median, Mode, Std, Skewness, Kurtosis, Range"),
        ("🔢", "Frequency Analysis",       "Frequency tables, Percent, Cumulative %, Bar Charts"),
        ("📐", "Normality Tests",           "Shapiro-Wilk · KS · Anderson-Darling · QQ-Plot · Histogram"),
        ("🔗", "Correlation Analysis",      "Pearson · Spearman · Heatmap · Significance matrices"),
        ("α",  "Reliability (Cronbach α)", "Cronbach's Alpha · Item-Total · Alpha-if-Deleted"),
        ("Σ",  "Factor Analysis",           "KMO · Bartlett · Varimax Rotation · Scree Plot · Communalities"),
        ("t",  "T-Test",                    "Independent Samples · Cohen's d · Levene's Test · Means Plot"),
        ("χ²", "Chi-Square",               "Contingency Table · Cramer's V · Expected Freq · Heatmap"),
        ("F",  "ANOVA (One-Way)",           "F-statistic · Eta² · Post-Hoc Tukey HSD · Means Plot"),
        ("β",  "Simple Regression",         "R² · ANOVA Table · Coefficients · Residual Plots"),
        ("β+", "Multiple Regression",       "Std. Beta · VIF · Durbin-Watson · QQ Residuals"),
        ("L",  "Logistic Regression",       "Odds Ratio · ROC/AUC · Cox-Snell R² · Classification Table"),
        ("⟳",  "Mediation Analysis",        "Bootstrap 1000 · Sobel Test · Path Diagram · Indirect Effect"),
        ("V",  "Multicollinearity (VIF)",   "VIF per variable · Interpretation guide"),
        ("⊕",  "Composite Variable",        "Mean or Sum of selected variables · New column · Download updated dataset"),
    ]

    cols_per_row = 4
    rows = [CARDS[i:i+cols_per_row] for i in range(0, len(CARDS), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for col, (icon, title, desc) in zip(cols, row):
            with col:
                st.markdown(f"""
<div class="stat-card">
  <div class="card-icon">{icon}</div>
  <div class="card-title">{title}</div>
  <div class="card-desc">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div style='text-align:center; color:#9ca3af; font-size:.8rem;'>
  HS-Statistical Assistant &nbsp;|&nbsp; Developed by <b>Haytham Saleh</b> &nbsp;|&nbsp; © 2026
</div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# DATA PREVIEW
# ─────────────────────────────────────────────
with st.expander("📋  Data Preview  (first 30 rows)", expanded=False):
    st.dataframe(df.head(30), use_container_width=True)

if not run_btn:
    st.info("👈  Select your variables and analysis from the sidebar, then press **▶ Run Analysis**")
    st.stop()

if "───" in str(analysis):
    st.warning("Please select a specific analysis (not a section header).")
    st.stop()


# ════════════════════════════════════════════════════════════════════
#                         A N A L Y S E S
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# 1. DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────
if analysis == "Descriptive Statistics":
    if not selected_vars:
        st.error("Please select at least one variable."); st.stop()
    data   = df[selected_vars].apply(pd.to_numeric, errors="coerce")
    result = pd.DataFrame(index=selected_vars)
    result["N"]         = data.count()
    result["Missing"]   = data.isna().sum()
    result["Mean"]      = data.mean()
    result["Median"]    = data.median()
    try:    result["Mode"] = data.mode().iloc[0]
    except: result["Mode"] = np.nan
    result["Std"]       = data.std()
    result["Variance"]  = data.var()
    result["Std Error"] = data.std() / np.sqrt(data.count())
    result["Min"]       = data.min()
    result["Max"]       = data.max()
    result["Range"]     = result["Max"] - result["Min"]
    result["Skewness"]  = data.skew()
    result["Kurtosis"]  = data.kurt()
    st.subheader("📊  Descriptive Statistics")
    st.dataframe(result.round(4), use_container_width=True)
    report = "DESCRIPTIVE STATISTICS REPORT\n==============================\n\n"
    report += result.round(4).to_string()
    pdf_download_button("Descriptive Statistics Report", report,
                        filename="descriptive_report.pdf")

# ─────────────────────────────────────────────
# 2. FREQUENCY ANALYSIS
# ─────────────────────────────────────────────
elif analysis == "Frequency Analysis":
    if not selected_vars:
        st.error("Please select at least one variable."); st.stop()
    st.subheader("🔢  Frequency Analysis")
    report    = "FREQUENCY ANALYSIS REPORT\n=========================\n\n"
    first_fig = None
    for col in selected_vars:
        data    = df[col]
        freq    = data.value_counts(dropna=False).sort_index()
        pct     = (freq / freq.sum()) * 100
        cum_pct = pct.cumsum()
        table   = pd.DataFrame({"Frequency": freq,
                                 "Percent (%)": pct.round(2),
                                 "Cumulative %": cum_pct.round(2)})
        st.markdown(f"#### Variable: `{col}`")
        st.dataframe(table, use_container_width=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(freq.index.astype(str), freq.values,
               color="#3b82f6", edgecolor="white", linewidth=0.8)
        ax.set_title(f"Bar Chart — {col}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Category"); ax.set_ylabel("Frequency")
        plt.xticks(rotation=45, ha="right"); fig.tight_layout()
        st.pyplot(fig)
        if first_fig is None: first_fig = fig
        else: plt.close(fig)
        report += f"\nVARIABLE: {col}\n" + "-"*30 + "\n" + table.to_string() + "\n\n"
    pdf_download_button("Frequency Analysis Report", report,
                        figs=[first_fig] if first_fig else None,
                        filename="frequency_report.pdf")

# ─────────────────────────────────────────────
# 3. NORMALITY TESTS
# ─────────────────────────────────────────────
elif analysis == "Normality Tests":
    if not selected_vars:
        st.error("Please select at least one variable."); st.stop()
    st.subheader("📐  Normality Tests")
    report       = "NORMALITY TESTS REPORT\n======================\n\n"
    report      += f"Variables Tested: {', '.join(selected_vars)}\n\n"
    summary_rows = []
    first_fig    = None
    for col in selected_vars:
        raw  = pd.to_numeric(df[col], errors="coerce").dropna()
        n    = len(raw)
        if n < 3: st.warning(f"'{col}' — fewer than 3 values, skipped."); continue
        mean, std = raw.mean(), raw.std()
        skew, kurt = raw.skew(), raw.kurt()
        sw_stat, sw_p = stats.shapiro(raw) if n <= 2000 else (np.nan, np.nan)
        ks_stat, ks_p = stats.kstest(raw, "norm", args=(mean, std))
        ad_res  = stats.anderson(raw, dist="norm")
        ad_stat = ad_res.statistic; ad_cv5 = ad_res.critical_values[2]
        ad_ok   = ad_stat < ad_cv5
        is_normal = (sw_p > 0.05) if not np.isnan(sw_p) else (ks_p > 0.05)
        label     = "✅ YES" if is_normal else "❌ NO"
        summary_rows.append({
            "Variable": col, "N": n,
            "Skewness": round(skew,3), "Kurtosis": round(kurt,3),
            "SW Stat": round(sw_stat,4) if not np.isnan(sw_stat) else "N/A",
            "SW p":    round(sw_p,4)    if not np.isnan(sw_p)    else "N/A",
            "KS p":    round(ks_p,4),
            "AD Stat": round(ad_stat,4), "AD Normal?": "Yes" if ad_ok else "No",
            "Normal?": label,
        })
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hist(raw, bins=15, density=True, color="#93c5fd", edgecolor="white")
        xr = np.linspace(raw.min(), raw.max(), 200)
        axes[0].plot(xr, (1/(std*np.sqrt(2*np.pi)))*np.exp(-0.5*((xr-mean)/std)**2),
                     color="#1e40af", linewidth=2)
        axes[0].set_title(f"Histogram — {col}"); axes[0].set_ylabel("Density")
        (osm, osr), (slope, intercept, _) = stats.probplot(raw, dist="norm")
        axes[1].scatter(osm, osr, color="#a78bfa", s=18, alpha=0.75)
        axes[1].plot([osm.min(),osm.max()],
                     [slope*osm.min()+intercept, slope*osm.max()+intercept],
                     color="#5b21b6", linewidth=2)
        axes[1].set_title(f"Q-Q Plot — {col}")
        axes[2].boxplot(raw, patch_artist=True,
                        boxprops=dict(facecolor="#d1fae5", color="#065f46"),
                        medianprops=dict(color="#065f46", linewidth=2))
        axes[2].set_title(f"Boxplot — {col}")
        fig.tight_layout(); st.pyplot(fig)
        if first_fig is None: first_fig = fig
        else: plt.close(fig)
        sw_str = f"{sw_p:.4f}" if not np.isnan(sw_p) else "N/A"
        report += f"\n{'─'*50}\nVariable: {col}\nN={n}  Mean={mean:.4f}  Std={std:.4f}\n"
        report += f"Skewness={skew:.4f}  Kurtosis={kurt:.4f}\n"
        report += f"Shapiro-Wilk: W={sw_stat if not np.isnan(sw_stat) else 'N/A'}  p={sw_str}\n"
        report += f"KS: D={ks_stat:.4f}  p={ks_p:.4f}\n"
        report += f"Anderson-Darling: A={ad_stat:.4f}  Critical5%={ad_cv5:.4f}\n"
        report += f"CONCLUSION: {'NORMAL' if is_normal else 'NOT NORMAL'}\n"
    if summary_rows:
        st.markdown("---"); st.markdown("#### 📋  Summary Table")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
    pdf_download_button("Normality Tests Report", report,
                        figs=[first_fig] if first_fig else None,
                        filename="normality_report.pdf")

# ─────────────────────────────────────────────
# 4. HISTOGRAM
# ─────────────────────────────────────────────
elif analysis == "Histogram":
    if not selected_vars: st.error("Please select at least one variable."); st.stop()
    for col in selected_vars:
        raw = pd.to_numeric(df[col], errors="coerce").dropna()
        mean, std = raw.mean(), raw.std()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(raw, bins=15, density=True, color="#93c5fd", edgecolor="white")
        xr = np.linspace(raw.min(), raw.max(), 200)
        ax.plot(xr, (1/(std*np.sqrt(2*np.pi)))*np.exp(-0.5*((xr-mean)/std)**2),
                color="#1e40af", linewidth=2)
        ax.set_title(f"Histogram — {col}"); ax.set_xlabel("Value"); ax.set_ylabel("Density")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ─────────────────────────────────────────────
# 5. SCATTER PLOT
# ─────────────────────────────────────────────
elif analysis == "Scatter Plot":
    if len(selected_vars) < 2: st.error("Select at least 2 variables."); st.stop()
    col1, col2 = selected_vars[0], selected_vars[1]
    data = df[[col1, col2]].apply(pd.to_numeric, errors="coerce").dropna()
    b1, b0 = np.polyfit(data[col1], data[col2], 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data[col1], data[col2], alpha=0.6, color="#3b82f6")
    ax.plot(data[col1], b1*data[col1]+b0, color="#ef4444", linewidth=2)
    ax.set_xlabel(col1); ax.set_ylabel(col2)
    ax.set_title(f"Scatter Plot: {col1} vs {col2}")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ─────────────────────────────────────────────
# 6. BOXPLOT
# ─────────────────────────────────────────────
elif analysis == "Boxplot":
    if not selected_vars: st.error("Please select at least one variable."); st.stop()
    data_list = [pd.to_numeric(df[c], errors="coerce").dropna().values for c in selected_vars]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data_list, labels=selected_vars, patch_artist=True,
               boxprops=dict(facecolor="#dbeafe", color="#1e40af"),
               medianprops=dict(color="#1e40af", linewidth=2))
    ax.set_title("Boxplot"); ax.set_ylabel("Value")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

# ─────────────────────────────────────────────
# 7. RELIABILITY
# ─────────────────────────────────────────────
elif analysis == "Reliability  (Cronbach Alpha)":
    if len(selected_vars) < 2: st.error("Select at least 2 variables."); st.stop()
    data = df[selected_vars].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] == 0: st.error("No valid numeric data."); st.stop()
    k           = data.shape[1]
    variances   = data.var(axis=0, ddof=1)
    total_score = data.sum(axis=1)
    total_var   = total_score.var(ddof=1)
    if total_var == 0: st.error("Total variance = 0."); st.stop()
    alpha_val = (k / (k-1)) * (1 - variances.sum() / total_var)
    if   alpha_val >= 0.9: level = "Excellent"
    elif alpha_val >= 0.8: level = "Good"
    elif alpha_val >= 0.7: level = "Acceptable"
    elif alpha_val >= 0.6: level = "Questionable"
    elif alpha_val >= 0.5: level = "Poor"
    else:                   level = "Unacceptable"
    item_total    = {c: data[c].corr(total_score - data[c]) for c in selected_vars}
    alpha_deleted = {}
    for col in selected_vars:
        sub = data.drop(columns=[col]); k2 = sub.shape[1]
        if k2 < 2: alpha_deleted[col] = np.nan; continue
        tv2 = sub.sum(axis=1).var(ddof=1)
        alpha_deleted[col] = (k2/(k2-1))*(1 - sub.var(ddof=1).sum()/tv2) if tv2 else np.nan
    st.subheader("α  Reliability Analysis — Cronbach's Alpha")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cronbach's Alpha", f"{alpha_val:.3f}")
    c2.metric("Level", level)
    c3.metric("Items (k)", k)
    st.info(f"Cronbach's alpha = **{alpha_val:.3f}** → **{level}** internal consistency.")
    item_df = pd.DataFrame({"Item-Total Correlation": pd.Series(item_total).round(3),
                             "Alpha if Item Deleted":  pd.Series(alpha_deleted).round(3)})
    st.markdown("#### Item Statistics"); st.dataframe(item_df, use_container_width=True)
    st.markdown("#### Inter-Item Correlation Matrix")
    st.dataframe(data.corr().round(3), use_container_width=True)
    report  = "RELIABILITY ANALYSIS\n====================\n\n"
    report += f"Cronbach's Alpha = {alpha_val:.3f}  →  {level}\nItems (k) = {k}\n\n"
    report += "Item-Total Correlation\n" + pd.Series(item_total).round(3).to_string() + "\n\n"
    report += "Alpha if Item Deleted\n"  + pd.Series(alpha_deleted).round(3).to_string() + "\n\n"
    report += "Inter-Item Correlation Matrix\n" + data.corr().round(3).to_string()
    pdf_download_button("Reliability Analysis Report", report, filename="reliability_report.pdf")

# ─────────────────────────────────────────────
# 8. FACTOR ANALYSIS
# ─────────────────────────────────────────────
elif analysis == "Factor Analysis":
    try:
        from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
    except ImportError:
        st.error("'factor_analyzer' library not found."); st.stop()
    if len(selected_vars) < 3: st.error("Select at least 3 variables."); st.stop()
    data = df[selected_vars].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] == 0: st.error("No valid data."); st.stop()
    kmo_all, kmo_model  = calculate_kmo(data)
    chi2_val, bart_p    = calculate_bartlett_sphericity(data)
    fa0 = FactorAnalyzer(rotation=None); fa0.fit(data)
    ev, _ = fa0.get_eigenvalues()
    n_factors = max(1, int(sum(ev > 1)))
    fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax"); fa.fit(data)
    loadings = pd.DataFrame(fa.loadings_, index=selected_vars,
                             columns=[f"Factor {i+1}" for i in range(n_factors)])
    communalities = pd.DataFrame(fa.get_communalities(), index=selected_vars, columns=["Communality"])
    st.subheader("Σ  Factor Analysis  (Varimax Rotation)")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("KMO", f"{kmo_model:.3f}")
    c2.metric("Bartlett χ²", f"{chi2_val:.2f}")
    c3.metric("Bartlett p", f"{bart_p:.5f}")
    c4.metric("Factors Extracted", n_factors)
    if   kmo_model < 0.5: st.warning("⚠️ KMO < 0.5 — data may not be suitable.")
    elif kmo_model < 0.7: st.info("KMO Mediocre (0.5–0.7) — proceed with caution.")
    else:                  st.success("KMO acceptable — Factor Analysis appropriate.")
    fig_scree, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(ev)+1), ev, marker="o", color="#3b82f6", linewidth=2)
    ax.axhline(y=1, color="red", linestyle="--", linewidth=1.5, label="Eigenvalue = 1")
    ax.set_title("Scree Plot", fontsize=13, fontweight="bold")
    ax.set_xlabel("Factor Number"); ax.set_ylabel("Eigenvalue"); ax.legend()
    fig_scree.tight_layout(); st.pyplot(fig_scree)
    st.markdown("#### Rotated Factor Loadings (Varimax)")
    st.dataframe(loadings.round(3), use_container_width=True)
    st.markdown("#### Communalities")
    st.dataframe(communalities.round(3), use_container_width=True)
    ev_series = pd.Series(ev, index=[f"Factor {i+1}" for i in range(len(ev))])
    st.markdown("#### Eigenvalues")
    st.dataframe(ev_series.round(3).to_frame("Eigenvalue"), use_container_width=True)
    report  = "FACTOR ANALYSIS REPORT\n======================\n\n"
    report += f"KMO = {kmo_model:.3f}\nBartlett χ² = {chi2_val:.3f}  p = {bart_p:.5f}\n"
    report += f"Factors Extracted = {n_factors}\n\n"
    report += "Eigenvalues\n" + ev_series.round(3).to_string() + "\n\n"
    report += "Rotated Factor Loadings (Varimax)\n" + loadings.round(3).to_string() + "\n\n"
    report += "Communalities\n" + communalities.round(3).to_string()
    pdf_download_button("Factor Analysis Report", report,
                        figs=[fig_scree], filename="factor_analysis_report.pdf")
    plt.close(fig_scree)

# ─────────────────────────────────────────────
# 9. CORRELATION ANALYSIS
# ─────────────────────────────────────────────
elif analysis == "Correlation Analysis":
    if len(selected_vars) < 2: st.error("Select at least 2 variables."); st.stop()
    data = df[selected_vars].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] == 0: st.error("No valid data."); st.stop()
    N = data.shape[0]
    def build_matrix(method="pearson"):
        cm = pd.DataFrame(index=selected_vars, columns=selected_vars, dtype=object)
        pm = pd.DataFrame(index=selected_vars, columns=selected_vars, dtype=object)
        for i in selected_vars:
            for j in selected_vars:
                if i == j: cm.loc[i,j] = "1.000"; pm.loc[i,j] = "—"
                else:
                    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
                    r, p = fn(data[i], data[j])
                    cm.loc[i,j] = f"{r:.3f}"
                    pm.loc[i,j] = "<.001" if p < 0.001 else f"{p:.3f}"
        return cm, pm
    p_corr, p_sig = build_matrix("pearson")
    s_corr, s_sig = build_matrix("spearman")
    st.subheader("r  Correlation Analysis"); st.metric("Sample Size (N)", N)
    tab1, tab2 = st.tabs(["Pearson", "Spearman"])
    with tab1:
        st.markdown("**Pearson Correlation Matrix**"); st.dataframe(p_corr, use_container_width=True)
        st.markdown("**p-values**"); st.dataframe(p_sig, use_container_width=True)
    with tab2:
        st.markdown("**Spearman Correlation Matrix**"); st.dataframe(s_corr, use_container_width=True)
        st.markdown("**p-values**"); st.dataframe(s_sig, use_container_width=True)
    num_corr = data.corr()
    fig_heat, ax = plt.subplots(figsize=(max(6, len(selected_vars)), max(5, len(selected_vars)-1)))
    im = ax.imshow(num_corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(selected_vars))); ax.set_yticks(range(len(selected_vars)))
    ax.set_xticklabels(selected_vars, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(selected_vars, fontsize=9)
    for i in range(len(selected_vars)):
        for j in range(len(selected_vars)):
            ax.text(j, i, f"{num_corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(num_corr.values[i,j]) > 0.6 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Pearson Correlation Heatmap", fontsize=13, fontweight="bold")
    fig_heat.tight_layout(); st.pyplot(fig_heat)
    report  = "CORRELATION ANALYSIS REPORT\n===========================\n\n"
    report += f"N = {N}\n\nPearson Correlation Matrix\n" + p_corr.to_string()
    report += "\n\nPearson p-values\n"  + p_sig.to_string()
    report += "\n\nSpearman Correlation Matrix\n" + s_corr.to_string()
    report += "\n\nSpearman p-values\n" + s_sig.to_string()
    pdf_download_button("Correlation Analysis Report", report,
                        figs=[fig_heat], filename="correlation_report.pdf")
    plt.close(fig_heat)

# ─────────────────────────────────────────────
# 10. MULTICOLLINEARITY (VIF)
# ─────────────────────────────────────────────
elif analysis == "Multicollinearity  (VIF)":
    if len(selected_vars) < 2: st.error("Select at least 2 variables."); st.stop()
    data = df[selected_vars].apply(pd.to_numeric, errors="coerce").dropna()
    X    = sm.add_constant(data)
    vif  = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"]      = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["Interpretation"] = vif["VIF"].apply(
        lambda x: "✅ Low (<5)" if x<5 else ("⚠️ Moderate (5-10)" if x<10 else "❌ High (>10)"))
    st.subheader("V  Multicollinearity — VIF")
    st.dataframe(vif.round(3), use_container_width=True)
    st.caption("VIF < 5 = OK  |  5–10 = Moderate  |  > 10 = Serious Multicollinearity")
    report  = "MULTICOLLINEARITY (VIF) REPORT\n==============================\n\n"
    report += vif.round(3).to_string()
    report += "\n\nVIF < 5 → OK  |  5–10 → Moderate  |  >10 → High\n"
    pdf_download_button("VIF Report", report, filename="vif_report.pdf")

# ─────────────────────────────────────────────
# 11. T-TEST
# ─────────────────────────────────────────────
elif analysis == "T-Test":
    if dep_var == "" or len(selected_vars) != 1:
        st.error("Select ONE grouping variable + set Dependent Variable."); st.stop()
    group_var = selected_vars[0]
    data = df[[dep_var, group_var]].copy()
    data[dep_var] = pd.to_numeric(data[dep_var], errors="coerce"); data.dropna(inplace=True)
    groups = data[group_var].unique()
    if len(groups) > 10:
        st.error(f"'{group_var}' has {len(groups)} unique values — needs exactly 2."); st.stop()
    if len(groups) != 2:
        st.error(f"T-Test requires exactly 2 groups. Found: {len(groups)}."); st.stop()
    g1 = data[data[group_var] == groups[0]][dep_var]
    g2 = data[data[group_var] == groups[1]][dep_var]
    lev_stat, lev_p = stats.levene(g1, g2); equal_var = lev_p > 0.05
    t_stat, p_val   = stats.ttest_ind(g1, g2, equal_var=equal_var); p_one = p_val / 2
    pooled = np.sqrt(((len(g1)-1)*g1.std()**2+(len(g2)-1)*g2.std()**2)/(len(g1)+len(g2)-2))
    d = (g1.mean()-g2.mean()) / pooled
    eff = "Large" if abs(d)>=0.8 else ("Medium" if abs(d)>=0.5 else "Small")
    se_diff = np.sqrt(g1.var()/len(g1)+g2.var()/len(g2))
    ci_lo = (g1.mean()-g2.mean()) - 1.96*se_diff
    ci_hi = (g1.mean()-g2.mean()) + 1.96*se_diff
    min_n = min(len(g1),len(g2))
    t_pair, p_pair = stats.ttest_rel(g1.values[:min_n], g2.values[:min_n])
    desc = data.groupby(group_var)[dep_var].agg(
        N="count", Mean="mean", Std="std",
        SE=lambda x: x.std()/np.sqrt(len(x)), Min="min", Max="max").round(4)
    sig_label = "statistically significant" if p_val < 0.05 else "NOT statistically significant"
    var_label = "Equal variances assumed" if equal_var else "Equal variances NOT assumed"
    interpretation = (
        f"An independent samples t-test compared {dep_var} between '{groups[0]}' and '{groups[1]}'. "
        f"Levene's test: {var_label} (F={lev_stat:.3f}, p={lev_p:.3f}). "
        f"The difference was {sig_label} (t={t_stat:.3f}, df={len(g1)+len(g2)-2}, p={p_val:.4f}). "
        f"Cohen's d = {d:.3f} — {eff} effect.")
    st.subheader("t  Independent Samples T-Test")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("t-statistic", f"{t_stat:.4f}"); c2.metric("p-value (2-tail)", f"{p_val:.4f}")
    c3.metric("Cohen's d", f"{d:.4f}");         c4.metric("Effect Size", eff)
    st.markdown("#### Group Descriptives"); st.dataframe(desc, use_container_width=True)
    st.markdown("#### Levene's Test")
    st.dataframe(pd.DataFrame({"F":[lev_stat],"p-value":[lev_p],
                                "Equal Variances?":[var_label]}).round(4),
                 use_container_width=True)
    st.markdown(f"#### 95% CI for Mean Difference:  [{ci_lo:.4f},  {ci_hi:.4f}]")
    fig_t, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.boxplot([g1, g2], labels=[str(groups[0]), str(groups[1])], patch_artist=True,
                boxprops=dict(facecolor="#dbeafe", color="#1e40af"),
                medianprops=dict(color="#1e40af", linewidth=2))
    ax1.set_title(f"Boxplot: {dep_var} by {group_var}"); ax1.set_ylabel(dep_var)
    ax2.bar([str(groups[0]),str(groups[1])],[g1.mean(),g2.mean()],
            yerr=[g1.sem(),g2.sem()], capsize=6, color=["#3b82f6","#f59e0b"], edgecolor="white")
    ax2.set_title(f"Means ± SE: {dep_var}"); ax2.set_ylabel(dep_var)
    fig_t.tight_layout(); st.pyplot(fig_t)
    st.info(f"📝 {interpretation}")
    report  = "INDEPENDENT SAMPLES T-TEST REPORT\n==================================\n\n"
    report += f"Dependent: {dep_var}\nGrouping: {group_var}\nGroups: {groups[0]} vs {groups[1]}\n\n"
    report += "GROUP DESCRIPTIVES\n" + desc.to_string() + "\n\n"
    report += f"LEVENE'S: F={lev_stat:.4f}  p={lev_p:.4f}  {var_label}\n\n"
    report += f"T-TEST: t={t_stat:.4f}  df={len(g1)+len(g2)-2}  p2={p_val:.4f}  p1={p_one:.4f}\n"
    report += f"Mean Diff={g1.mean()-g2.mean():.4f}  95%CI[{ci_lo:.4f},{ci_hi:.4f}]\n"
    report += f"Cohen's d={d:.4f} ({eff})\n\nPAIRED: t={t_pair:.4f}  p={p_pair:.4f}\n\n"
    report += "INTERPRETATION\n" + interpretation
    pdf_download_button("T-Test Report", report, figs=[fig_t], filename="ttest_report.pdf")
    plt.close(fig_t)

# ─────────────────────────────────────────────
# 12. CHI-SQUARE
# ─────────────────────────────────────────────
elif analysis == "Chi-Square":
    if len(selected_vars) != 2: st.error("Select exactly TWO variables."); st.stop()
    var1, var2 = selected_vars[0], selected_vars[1]
    data = df[[var1, var2]].dropna()
    cross_tab = pd.crosstab(data[var1], data[var2])
    chi2_s, p_val, dof, expected = stats.chi2_contingency(cross_tab)
    n = data.shape[0]; min_dim = min(cross_tab.shape)-1
    cramers_v = np.sqrt(chi2_s/(n*min_dim)) if min_dim > 0 else 0
    eff_level = "Strong" if cramers_v>=0.5 else ("Moderate" if cramers_v>=0.3 else
                ("Weak" if cramers_v>=0.1 else "Negligible"))
    exp_df  = pd.DataFrame(expected, index=cross_tab.index, columns=cross_tab.columns)
    row_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0)*100
    low_exp = (exp_df < 5).values.sum(); pct_low = (low_exp/exp_df.size)*100
    sig_label = "statistically significant" if p_val < 0.05 else "NOT statistically significant"
    interpretation = (f"Chi-Square between '{var1}' and '{var2}': {sig_label} "
                      f"(χ²={chi2_s:.3f}, df={dof}, p={p_val:.4f}). "
                      f"Cramer's V={cramers_v:.3f} — {eff_level}.")
    if pct_low > 20:
        interpretation += f" ⚠️ {pct_low:.1f}% expected frequencies < 5."
    st.subheader("χ²  Chi-Square Test of Independence")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("χ²", f"{chi2_s:.4f}"); c2.metric("p-value", f"{p_val:.4f}")
    c3.metric("df", dof);             c4.metric("Cramer's V", f"{cramers_v:.4f}")
    tab1,tab2,tab3 = st.tabs(["Observed","Expected","Row %"])
    with tab1: st.dataframe(cross_tab, use_container_width=True)
    with tab2: st.dataframe(exp_df.round(2), use_container_width=True)
    with tab3: st.dataframe(row_pct.round(2), use_container_width=True)
    colors = ["#3b82f6","#f59e0b","#10b981","#ef4444","#8b5cf6","#ec4899"]
    fig_chi, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(cross_tab.index)); width = 0.8/len(cross_tab.columns)
    for i, col in enumerate(cross_tab.columns):
        ax.bar(x+i*width, cross_tab[col], width=width, label=str(col),
               color=colors[i%len(colors)], edgecolor="white")
    ax.set_xticks(x+width*(len(cross_tab.columns)-1)/2)
    ax.set_xticklabels([str(v) for v in cross_tab.index], rotation=30)
    ax.set_title(f"Chi-Square: {var1} × {var2}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Frequency"); ax.legend(title=var2)
    fig_chi.tight_layout(); st.pyplot(fig_chi)
    st.info(f"📝 {interpretation}")
    report  = "CHI-SQUARE TEST REPORT\n======================\n\n"
    report += f"{var1} × {var2}  N={n}\nχ²={chi2_s:.4f}  df={dof}  p={p_val:.4f}\n"
    report += f"Cramer's V={cramers_v:.4f} ({eff_level})\n\n"
    report += "Contingency Table\n" + cross_tab.to_string() + "\n\n"
    report += "Expected Frequencies\n" + exp_df.round(2).to_string() + "\n\n"
    report += "Row Percentages\n" + row_pct.round(2).to_string() + "\n\n"
    report += "INTERPRETATION\n" + interpretation
    pdf_download_button("Chi-Square Report", report, figs=[fig_chi],
                        filename="chi_square_report.pdf")
    plt.close(fig_chi)

# ─────────────────────────────────────────────
# 13. ANOVA
# ─────────────────────────────────────────────
elif analysis == "ANOVA  (One-Way)":
    if dep_var == "" or len(selected_vars) != 1:
        st.error("Select ONE grouping variable + set Dependent Variable."); st.stop()
    group_var = selected_vars[0]
    data = df[[dep_var, group_var]].dropna()
    data[dep_var] = pd.to_numeric(data[dep_var], errors="coerce"); data.dropna(inplace=True)
    groups_data = []; labels = []
    for g in sorted(data[group_var].unique()):
        grp = data[data[group_var]==g][dep_var]
        if len(grp)>0: groups_data.append(grp); labels.append(str(g))
    if len(groups_data) < 2: st.error("Need at least 2 groups."); st.stop()
    F, p = stats.f_oneway(*groups_data)
    grand_mean = data[dep_var].mean()
    ss_between = sum(len(g)*(g.mean()-grand_mean)**2 for g in groups_data)
    ss_total   = ((data[dep_var]-grand_mean)**2).sum()
    eta_sq     = ss_between/ss_total
    tukey = pairwise_tukeyhsd(endog=data[dep_var], groups=data[group_var], alpha=0.05)
    means = data.groupby(group_var)[dep_var].mean()
    sds   = data.groupby(group_var)[dep_var].std()
    cnts  = data.groupby(group_var)[dep_var].count()
    cis   = 1.96*sds/np.sqrt(cnts)
    tbl   = pd.DataFrame({"N":cnts,"Mean":means,"Std":sds,"CI95":cis})
    st.subheader("F  One-Way ANOVA")
    c1,c2,c3 = st.columns(3)
    c1.metric("F-statistic", f"{F:.4f}"); c2.metric("p-value", f"{p:.5f}")
    c3.metric("Eta Squared η²", f"{eta_sq:.4f}")
    st.markdown("#### Group Statistics"); st.dataframe(tbl.round(4), use_container_width=True)
    st.markdown("#### Post-Hoc Tukey HSD")
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:],
                             columns=tukey._results_table.data[0])
    st.dataframe(tukey_df, use_container_width=True)
    fig_an, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(range(len(means)), means, yerr=cis, fmt="o-", capsize=6,
                color="#3b82f6", ecolor="#1e40af", linewidth=2)
    ax.set_xticks(range(len(means))); ax.set_xticklabels(labels)
    ax.set_title("Means Plot with 95% CI", fontsize=13, fontweight="bold")
    ax.set_xlabel(group_var); ax.set_ylabel(dep_var)
    fig_an.tight_layout(); st.pyplot(fig_an)
    report  = "ONE-WAY ANOVA REPORT\n====================\n\n"
    report += f"Dependent: {dep_var}\nGrouping: {group_var}\n\n"
    report += f"F={F:.4f}  p={p:.5f}  Eta²={eta_sq:.4f}\n\n"
    report += "Group Statistics\n" + tbl.round(4).to_string() + "\n\n"
    report += "Post-Hoc Tukey HSD\n" + tukey.summary().as_text()
    pdf_download_button("ANOVA Report", report, figs=[fig_an], filename="anova_report.pdf")
    plt.close(fig_an)

# ─────────────────────────────────────────────
# 14. SIMPLE REGRESSION
# ─────────────────────────────────────────────
elif analysis == "Simple Regression":
    if dep_var == "" or len(selected_vars) != 1:
        st.error("Select ONE independent variable + set Dependent Variable."); st.stop()
    data = df[[dep_var]+selected_vars].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] == 0: st.error("No valid data."); st.stop()
    y = data[dep_var]; X = sm.add_constant(data[selected_vars]); md = sm.OLS(y,X).fit()
    ssr=md.ess; sse=md.ssr; sst=ssr+sse
    df_r=int(md.df_model); df_e=int(md.df_resid)
    msr=ssr/df_r; mse=sse/df_e; dw=sm.stats.stattools.durbin_watson(md.resid)
    an_tbl  = pd.DataFrame({"SS":[ssr,sse,sst],"df":[df_r,df_e,df_r+df_e],
                             "MS":[msr,mse,""],"F":[md.fvalue,"",""],
                             "Sig":[md.f_pvalue,"",""]},
                            index=["Regression","Residual","Total"])
    coef_tbl = pd.DataFrame({"Beta":md.params,"t":md.tvalues,"Sig":md.pvalues})
    st.subheader("β  Simple Linear Regression")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("R²",f"{md.rsquared:.3f}"); c2.metric("Adj R²",f"{md.rsquared_adj:.3f}")
    c3.metric("F",f"{md.fvalue:.3f}");    c4.metric("Durbin-Watson",f"{dw:.3f}")
    st.markdown("#### ANOVA Table"); st.dataframe(an_tbl.round(4), use_container_width=True)
    st.markdown("#### Coefficients"); st.dataframe(coef_tbl.round(4), use_container_width=True)
    fig_sr, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.scatter(md.fittedvalues, md.resid, alpha=0.6, color="#3b82f6")
    ax1.axhline(0, color="red", linestyle="--")
    ax1.set_title("Residual Plot"); ax1.set_xlabel("Predicted"); ax1.set_ylabel("Residual")
    ax2.hist(md.resid, bins=15, color="#a78bfa", edgecolor="white")
    ax2.set_title("Histogram of Residuals")
    fig_sr.tight_layout(); st.pyplot(fig_sr)
    report  = "SIMPLE REGRESSION REPORT\n========================\n\n"
    report += f"N={data.shape[0]}  Dep={dep_var}  Indep={selected_vars[0]}\n\n"
    report += f"R²={md.rsquared:.3f}  Adj R²={md.rsquared_adj:.3f}  F={md.fvalue:.3f}  "
    report += f"p={md.f_pvalue:.5f}  DW={dw:.3f}\n\n"
    report += "ANOVA TABLE\n" + an_tbl.round(4).to_string() + "\n\n"
    report += "COEFFICIENTS\n" + coef_tbl.round(4).to_string()
    pdf_download_button("Simple Regression Report", report, figs=[fig_sr],
                        filename="simple_regression_report.pdf")
    plt.close(fig_sr)

# ─────────────────────────────────────────────
# 15. MULTIPLE REGRESSION
# ─────────────────────────────────────────────
elif analysis == "Multiple Regression":
    if dep_var == "" or len(selected_vars) < 2:
        st.error("Select at least TWO independent variables + set Dependent Variable."); st.stop()
    data = df[[dep_var]+selected_vars].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] == 0: st.error("No valid data."); st.stop()
    y = data[dep_var]; X = sm.add_constant(data[selected_vars]); md = sm.OLS(y,X).fit()
    y_std = (y-y.mean())/y.std()
    X_std = (data[selected_vars]-data[selected_vars].mean())/data[selected_vars].std()
    md_std = sm.OLS(y_std, sm.add_constant(X_std)).fit()
    dw = sm.stats.stattools.durbin_watson(md.resid)
    coef_tbl = pd.DataFrame({"Unstd. Beta":md.params,"Std. Beta":md_std.params,
                              "t":md.tvalues,"Sig":md.pvalues})
    vif_df = pd.DataFrame({"Variable":X.columns,
                            "VIF":[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]})
    st.subheader("β+  Multiple Linear Regression")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("R²",f"{md.rsquared:.3f}"); c2.metric("Adj R²",f"{md.rsquared_adj:.3f}")
    c3.metric("F",f"{md.fvalue:.3f}");    c4.metric("Durbin-Watson",f"{dw:.3f}")
    st.markdown("#### Coefficients (Unstd. + Std. Beta)")
    st.dataframe(coef_tbl.round(4), use_container_width=True)
    st.markdown("#### VIF — Multicollinearity")
    st.dataframe(vif_df.round(3), use_container_width=True)
    fig_mr, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(md.fittedvalues, md.resid, alpha=0.6, color="#3b82f6")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title("Residual Plot"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
    axes[1].hist(md.resid, bins=15, color="#a78bfa", edgecolor="white")
    axes[1].set_title("Histogram of Residuals")
    (osm,osr),(slope,intercept,_) = stats.probplot(md.resid, dist="norm")
    axes[2].scatter(osm, osr, s=20, alpha=0.7, color="#10b981")
    axes[2].plot([osm.min(),osm.max()],
                 [slope*osm.min()+intercept, slope*osm.max()+intercept], color="red", linewidth=2)
    axes[2].set_title("QQ Plot — Residuals")
    fig_mr.tight_layout(); st.pyplot(fig_mr)
    report  = "MULTIPLE REGRESSION REPORT\n==========================\n\n"
    report += f"N={data.shape[0]}  Dep={dep_var}  Indep={', '.join(selected_vars)}\n\n"
    report += f"R²={md.rsquared:.3f}  Adj R²={md.rsquared_adj:.3f}  F={md.fvalue:.3f}  "
    report += f"p={md.f_pvalue:.5f}  DW={dw:.3f}\n\n"
    report += "COEFFICIENTS\n" + coef_tbl.round(4).to_string() + "\n\nVIF\n" + vif_df.round(3).to_string()
    pdf_download_button("Multiple Regression Report", report, figs=[fig_mr],
                        filename="multiple_regression_report.pdf")
    plt.close(fig_mr)

# ─────────────────────────────────────────────
# 16. LOGISTIC REGRESSION
# ─────────────────────────────────────────────
elif analysis == "Logistic Regression":
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
    if dep_var == "" or len(selected_vars) < 1:
        st.error("Select at least ONE independent variable + set Dependent Variable."); st.stop()
    data = df[[dep_var]+selected_vars].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] == 0: st.error("No valid data."); st.stop()
    if data[dep_var].nunique() != 2:
        st.error(f"Dependent variable must have 2 categories. Found: {data[dep_var].nunique()}."); st.stop()
    y = data[dep_var]; X = sm.add_constant(data[selected_vars])
    try: md = sm.Logit(y, X).fit(disp=0)
    except Exception as exc: st.error(f"Model failed: {exc}"); st.stop()
    or_vals = np.exp(md.params); conf = np.exp(md.conf_int())
    conf.columns = ["OR Lower 95%","OR Upper 95%"]
    coef_tbl = pd.DataFrame({"Coefficient":md.params,"Std Error":md.bse,
                              "Wald (z)":md.tvalues,"Sig (p)":md.pvalues,
                              "Odds Ratio":or_vals,
                              "OR Lower 95%":conf["OR Lower 95%"],
                              "OR Upper 95%":conf["OR Upper 95%"]})
    n = data.shape[0]; ll_null=md.llnull; ll_model=md.llf
    cox_snell  = 1 - np.exp((2/n)*(ll_null-ll_model))
    nagelkerke = cox_snell / (1 - np.exp((2/n)*ll_null))
    y_pred_prob = md.predict(X); y_pred = (y_pred_prob>=0.5).astype(int)
    cm = confusion_matrix(y, y_pred); accuracy = np.trace(cm)/cm.sum()*100
    try:
        auc_val = roc_auc_score(y, y_pred_prob)
        fpr, tpr, _ = roc_curve(y, y_pred_prob); has_roc = True
    except: has_roc = False; auc_val = np.nan
    labels_cls = sorted(y.unique())
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels_cls],
                         columns=[f"Predicted {l}" for l in labels_cls])
    st.subheader("L  Logistic Regression")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Cox & Snell R²",f"{cox_snell:.4f}"); c2.metric("Nagelkerke R²",f"{nagelkerke:.4f}")
    c3.metric("Accuracy",f"{accuracy:.2f}%");       c4.metric("AUC",f"{auc_val:.4f}" if has_roc else "N/A")
    st.markdown("#### Coefficients & Odds Ratios")
    st.dataframe(coef_tbl.round(4), use_container_width=True)
    st.markdown("#### Classification Table"); st.dataframe(cm_df, use_container_width=True)
    fig_lr, axes = plt.subplots(1, 2, figsize=(12, 4))
    vp = [v for v in coef_tbl.index if v != "const"]
    ov = or_vals[vp]
    el = ov.values - conf.loc[vp,"OR Lower 95%"].values
    eu = conf.loc[vp,"OR Upper 95%"].values - ov.values
    axes[0].barh(vp, ov, xerr=[el,eu], color="#a78bfa", edgecolor="#5b21b6", capsize=5)
    axes[0].axvline(x=1, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Odds Ratio Plot with 95% CI"); axes[0].set_xlabel("Odds Ratio")
    if has_roc:
        axes[1].plot(fpr, tpr, color="#1e40af", linewidth=2, label=f"AUC={auc_val:.3f}")
        axes[1].plot([0,1],[0,1], color="gray", linestyle="--")
        axes[1].set_title("ROC Curve"); axes[1].legend()
        axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    else: axes[1].text(0.5,0.5,"ROC N/A",ha="center",va="center",fontsize=12)
    fig_lr.tight_layout(); st.pyplot(fig_lr)
    report  = "LOGISTIC REGRESSION REPORT\n==========================\n\n"
    report += f"N={n}  Dep={dep_var}  Indep={', '.join(selected_vars)}\n\n"
    report += f"Cox & Snell R²={cox_snell:.4f}  Nagelkerke R²={nagelkerke:.4f}\n"
    report += f"Accuracy={accuracy:.2f}%  AUC={auc_val:.4f if has_roc else 'N/A'}\n\n"
    report += "COEFFICIENTS & ODDS RATIOS\n" + coef_tbl.round(4).to_string() + "\n\n"
    report += "CLASSIFICATION TABLE\n" + cm_df.to_string()
    pdf_download_button("Logistic Regression Report", report, figs=[fig_lr],
                        filename="logistic_regression_report.pdf")
    plt.close(fig_lr)

# ─────────────────────────────────────────────
# 17. MEDIATION ANALYSIS
# ─────────────────────────────────────────────
elif analysis == "Mediation Analysis":
    if dep_var == "" or len(selected_vars) != 2:
        st.error("Need: Variable1=X (Independent), Variable2=M (Mediator), Dep=Y"); st.stop()
    X_name, M_name, Y_name = selected_vars[0], selected_vars[1], dep_var
    data = df[[X_name,M_name,Y_name]].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] == 0: st.error("No valid data."); st.stop()
    if data.shape[0] < 30: st.warning(f"Small sample (N={data.shape[0]}). N≥100 recommended.")
    X = data[X_name]; M = data[M_name]; Y = data[Y_name]
    Xc = sm.add_constant(X); md_a = sm.OLS(M, Xc).fit()
    a = md_a.params[X_name]; a_se = md_a.bse[X_name]; a_p = md_a.pvalues[X_name]
    XMc = sm.add_constant(data[[X_name,M_name]]); md_b = sm.OLS(Y, XMc).fit()
    b = md_b.params[M_name]; b_se = md_b.bse[M_name]; b_p = md_b.pvalues[M_name]
    c_prime = md_b.params[X_name]; cp_p = md_b.pvalues[X_name]
    md_c = sm.OLS(Y, Xc).fit(); c = md_c.params[X_name]; c_p = md_c.pvalues[X_name]
    indirect = a*b
    sobel_se = np.sqrt(b**2*a_se**2 + a**2*b_se**2)
    sobel_z  = indirect/sobel_se if sobel_se!=0 else 0
    sobel_p  = 2*(1-stats.norm.cdf(abs(sobel_z)))
    prop_med = (indirect/c*100) if c!=0 else 0
    np.random.seed(42); boot_ind = []
    for _ in range(1000):
        idx = np.random.choice(len(data), len(data), replace=True); bd = data.iloc[idx]
        try:
            ba = sm.OLS(bd[M_name], sm.add_constant(bd[X_name])).fit().params[X_name]
            bb = sm.OLS(bd[Y_name], sm.add_constant(bd[[X_name,M_name]])).fit().params[M_name]
            boot_ind.append(ba*bb)
        except: pass
    bci_lo = np.percentile(boot_ind, 2.5); bci_hi = np.percentile(boot_ind, 97.5)
    boot_sig = "Significant ✅" if (bci_lo>0 or bci_hi<0) else "Not Significant ❌"
    st.subheader("⟳  Mediation Analysis  (Bootstrap 1 000 iterations)")
    c1,c2,c3 = st.columns(3)
    c1.metric("Path a (X→M)", f"{a:.4f}"); c2.metric("Path b (M→Y)", f"{b:.4f}")
    c3.metric("Direct c' (X→Y)", f"{c_prime:.4f}")
    c4,c5,c6 = st.columns(3)
    c4.metric("Total Effect c", f"{c:.4f}"); c5.metric("Indirect a×b", f"{indirect:.4f}")
    c6.metric("Proportion Mediated", f"{prop_med:.1f}%")
    paths_tbl = pd.DataFrame({
        "Path":["a: X→M","b: M→Y","c': Direct X→Y","c: Total X→Y"],
        "Coefficient":[a,b,c_prime,c],"p-value":[a_p,b_p,cp_p,c_p]})
    st.markdown("#### Path Coefficients"); st.dataframe(paths_tbl.round(4), use_container_width=True)
    boot_tbl = pd.DataFrame({
        "Indirect (a×b)":[round(indirect,4)],
        "Boot CI Lower":[round(bci_lo,4)],"Boot CI Upper":[round(bci_hi,4)],
        "Sobel Z":[round(sobel_z,4)],"Sobel p":[round(sobel_p,4)],
        "Bootstrap Result":[boot_sig]})
    st.markdown("#### Indirect Effect & Bootstrap 95% CI")
    st.dataframe(boot_tbl, use_container_width=True)
    fig_med, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off")
    for (x0,y0,label,fc,ec) in [(0.3,2.0,X_name,"#dbeafe","#1e40af"),
                                  (3.8,4.0,M_name,"#d1fae5","#065f46"),
                                  (7.4,2.0,Y_name,"#fef3c7","#92400e")]:
        ax.add_patch(plt.Rectangle((x0,y0),2.2,1.4,facecolor=fc,edgecolor=ec,linewidth=2,zorder=3))
        ax.text(x0+1.1,y0+0.7,label,ha="center",va="center",fontsize=10,fontweight="bold",zorder=4)
    ax.annotate("",xy=(3.8,4.6),xytext=(2.5,3.1),
                arrowprops=dict(arrowstyle="->",color="#065f46",lw=2.5))
    ax.text(2.8,4.0,f"a={a:.3f}\np={a_p:.3f}",fontsize=9,color="#065f46")
    ax.annotate("",xy=(7.4,3.1),xytext=(6.0,4.6),
                arrowprops=dict(arrowstyle="->",color="#065f46",lw=2.5))
    ax.text(7.0,4.2,f"b={b:.3f}\np={b_p:.3f}",fontsize=9,color="#065f46")
    ax.annotate("",xy=(7.4,2.7),xytext=(2.5,2.7),
                arrowprops=dict(arrowstyle="->",color="#92400e",lw=2.5))
    ax.text(4.8,2.3,f"c'={c_prime:.3f}(p={cp_p:.3f})\nc={c:.3f}(p={c_p:.3f})",
            fontsize=9,color="#92400e",ha="center")
    ax.set_title(f"Mediation Path Diagram\nIndirect={indirect:.4f}  "
                 f"Bootstrap 95%CI[{bci_lo:.4f},{bci_hi:.4f}]",fontsize=11,fontweight="bold")
    fig_med.tight_layout(); st.pyplot(fig_med)
    report  = "MEDIATION ANALYSIS REPORT\n=========================\n\n"
    report += f"X={X_name}  M={M_name}  Y={Y_name}  N={data.shape[0]}\n\n"
    report += f"Path a (X→M)  = {a:.4f}  p={a_p:.4f}\n"
    report += f"Path b (M→Y)  = {b:.4f}  p={b_p:.4f}\n"
    report += f"Direct c'     = {c_prime:.4f}  p={cp_p:.4f}\n"
    report += f"Total c       = {c:.4f}  p={c_p:.4f}\n\n"
    report += f"Indirect a×b  = {indirect:.4f}\n"
    report += f"Sobel Z={sobel_z:.4f}  p={sobel_p:.4f}\n"
    report += f"Bootstrap 95%CI = [{bci_lo:.4f}, {bci_hi:.4f}]\n"
    report += f"Result = {boot_sig}\nProportion Mediated = {prop_med:.2f}%\n"
    pdf_download_button("Mediation Analysis Report", report, figs=[fig_med],
                        filename="mediation_report.pdf")
    plt.close(fig_med)

# ─────────────────────────────────────────────
# 18. COMPOSITE VARIABLE
# ─────────────────────────────────────────────
elif analysis == "Composite Variable":
    st.subheader("⊕  Composite Variable Builder")

    # ── Step 1 : pick columns ──────────────────────────────────────────────
    num_cols = list(df.select_dtypes(include=np.number).columns)
    if len(num_cols) < 2:
        st.error("Need at least 2 numeric columns in your dataset."); st.stop()

    st.markdown("#### Step 1 — Select columns to combine")
    comp_vars = st.multiselect(
        "Choose two or more variables",
        options=list(df.columns),
        default=selected_vars if selected_vars else []
    )

    # ── Step 2 : new column name ───────────────────────────────────────────
    st.markdown("#### Step 2 — Name the new composite column")
    new_col_name = st.text_input(
        "New column name",
        value="Composite_1",
        help="This name will appear as a new column in your dataset"
    )

    # ── Step 3 : method ────────────────────────────────────────────────────
    st.markdown("#### Step 3 — Choose aggregation method")
    method = st.radio(
        "Aggregation",
        options=["Mean (average)", "Sum (total)"],
        horizontal=True
    )

    # ── Validation ─────────────────────────────────────────────────────────
    if len(comp_vars) < 2:
        st.info("👆 Please select at least 2 variables above.")
        st.stop()

    if not new_col_name.strip():
        st.warning("Please enter a name for the new column."); st.stop()

    new_col_name = new_col_name.strip()

    bad_cols = [c for c in comp_vars if not pd.api.types.is_numeric_dtype(df[c])]
    if bad_cols:
        st.error(f"These columns are not numeric: {', '.join(bad_cols)}"); st.stop()

    # ── Compute ────────────────────────────────────────────────────────────
    data_sub = df[comp_vars].apply(pd.to_numeric, errors="coerce")
    if method.startswith("Mean"):
        composite_values = data_sub.mean(axis=1)
        method_label = "Mean"
    else:
        composite_values = data_sub.sum(axis=1)
        method_label = "Sum"

    valid_n   = data_sub.dropna(how="all").shape[0]
    missing_n = data_sub.isna().any(axis=1).sum()

    df_new = df.copy()
    if new_col_name in df_new.columns:
        st.warning(f"⚠️ Column '{new_col_name}' already exists and will be overwritten.")
    df_new[new_col_name] = composite_values

    # ── Summary ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"#### ✅  New Column: `{new_col_name}`  "
        f"({method_label} of {len(comp_vars)} variables)"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Method",           method_label)
    c2.metric("Variables",        len(comp_vars))
    c3.metric("Valid Rows",       int(valid_n))
    c4.metric("Rows w/ Missing",  int(missing_n))

    clean_vals = composite_values.dropna()
    mean_v, std_v = clean_vals.mean(), clean_vals.std()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean",  f"{mean_v:.4f}")
    col2.metric("Std",   f"{std_v:.4f}")
    col3.metric("Min",   f"{clean_vals.min():.4f}")
    col4.metric("Max",   f"{clean_vals.max():.4f}")

    # ── Inter-item correlation ─────────────────────────────────────────────
    with st.expander("📊  Inter-item correlations (selected variables)", expanded=False):
        st.dataframe(data_sub.corr().round(3), use_container_width=True)

    # ── Distribution charts ────────────────────────────────────────────────
    fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(clean_vals, bins=15, density=True, color="#93c5fd", edgecolor="white")
    if std_v > 0:
        xr = np.linspace(clean_vals.min(), clean_vals.max(), 200)
        ax1.plot(xr,
                 (1/(std_v*np.sqrt(2*np.pi))) * np.exp(-0.5*((xr-mean_v)/std_v)**2),
                 color="#1e40af", linewidth=2, label="Normal curve")
        ax1.legend()
    ax1.set_title(f"Distribution of  {new_col_name}")
    ax1.set_xlabel("Value"); ax1.set_ylabel("Density")

    ax2.boxplot(clean_vals, patch_artist=True,
                boxprops=dict(facecolor="#d1fae5", color="#065f46"),
                medianprops=dict(color="#065f46", linewidth=2))
    ax2.set_title(f"Boxplot — {new_col_name}"); ax2.set_ylabel("Value")

    fig_comp.suptitle(
        f"{method_label} of: {', '.join(comp_vars)}", fontsize=10, color="#6b7280"
    )
    fig_comp.tight_layout(); st.pyplot(fig_comp)

    # ── Dataset preview ────────────────────────────────────────────────────
    st.markdown("#### 📋  Updated Dataset  (first 30 rows — selected cols + new column)")
    preview_cols = comp_vars + [new_col_name]
    st.dataframe(df_new[preview_cols].head(30), use_container_width=True)

    # ── Download Excel ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📥  Download Updated Dataset")
    st.caption(
        f"The Excel file below contains **all original columns** plus the new "
        f"`{new_col_name}` column at the end."
    )

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        df_new.to_excel(writer, index=False, sheet_name="Data")
    excel_buf.seek(0)

    st.download_button(
        label=f"📥  Download Excel  (with  '{new_col_name}')",
        data=excel_buf,
        file_name=f"data_with_{new_col_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    # ── Add to session (use in other analyses) ─────────────────────────────
    if st.button(
        f"➕  Add '{new_col_name}' to working dataset  (use in this session)",
        use_container_width=True
    ):
        st.session_state["df"] = df_new
        st.success(
            f"✅ '{new_col_name}' has been added! "
            "You can now select it in any other analysis from the sidebar."
        )
        st.rerun()

    # ── PDF report ─────────────────────────────────────────────────────────
    report  = "COMPOSITE VARIABLE REPORT\n=========================\n\n"
    report += f"New Column     : {new_col_name}\n"
    report += f"Method         : {method_label}\n"
    report += f"Variables Used : {', '.join(comp_vars)}\n"
    report += f"Valid Rows     : {valid_n}\n"
    report += f"Rows w/ Missing: {missing_n}\n\n"
    report += "DESCRIPTIVE STATISTICS\n"
    report += f"  Mean = {mean_v:.4f}\n  Std  = {std_v:.4f}\n"
    report += f"  Min  = {clean_vals.min():.4f}\n  Max  = {clean_vals.max():.4f}\n\n"
    report += "INTER-ITEM CORRELATIONS\n"
    report += data_sub.corr().round(3).to_string()

    pdf_download_button(
        f"Composite Variable Report — {new_col_name}",
        report,
        figs=[fig_comp],
        filename=f"composite_{new_col_name}_report.pdf"
    )
    plt.close(fig_comp)

else:
    st.warning("Please select a specific analysis from the sidebar.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#9ca3af; font-size:.8rem; line-height:2;'>
  HS-Statistical Assistant &nbsp;|&nbsp; Developed by <b>Haytham Saleh</b> &nbsp;|&nbsp;
  DBA Candidate · MBA · MSc · MCTS · MCSE · SAS &nbsp;|&nbsp; © 2026 All Rights Reserved<br>
  <a href="https://www.linkedin.com/in/haytham-saleh-0b407226/" target="_blank"
     style="display:inline-flex; align-items:center; gap:5px; background:#0a66c2;
            color:white; padding:5px 12px; border-radius:6px; text-decoration:none;
            font-weight:bold; font-size:.78rem; margin-top:6px;">
    <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24"
         fill="white" style="vertical-align:middle;">
      <path d="M20.447 20.452H17.21v-5.569c0-1.328-.027-3.037-1.852-3.037
               -1.853 0-2.136 1.445-2.136 2.939v5.667H9.987V9h3.102v1.561h.044
               c.432-.818 1.487-1.681 3.059-1.681 3.271 0 3.874 2.152 3.874 4.95v6.622z
               M5.337 7.433a1.8 1.8 0 1 1 0-3.6 1.8 1.8 0 0 1 0 3.6z
               M6.95 20.452H3.723V9H6.95v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729
               v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271
               V1.729C24 .774 23.2 0 22.222 0h.003z"/>
    </svg>
    Connect on LinkedIn
  </a>
</div>""", unsafe_allow_html=True)
