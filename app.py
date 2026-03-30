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
import os
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
# CSS STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2980b9 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 10px;
        margin-bottom: 18px;
    }
    .main-header h2 { margin: 0; font-size: 1.6rem; }
    .main-header p  { margin: 4px 0 0; opacity: 0.85; font-size: 0.9rem; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 10px 14px;
        border-left: 4px solid #2980b9;
    }

    /* Sidebar buttons */
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: bold;
        transition: 0.2s;
    }

    /* Code / report blocks */
    .report-box {
        background: #f8f9fa;
        border-left: 4px solid #2980b9;
        padding: 14px;
        border-radius: 6px;
        font-family: "Courier New", monospace;
        font-size: 0.82rem;
        white-space: pre-wrap;
        overflow-x: auto;
    }

    /* Section headers */
    h3 { color: #1e3a5f; }
    h4 { color: #2980b9; margin-top: 1rem; }
/* Sidebar premium background */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc, #eef2ff);
    border-right: 1px solid #dbeafe;
}

/* Premium run button */
.stButton > button {
    background: linear-gradient(90deg,#2563eb,#1d4ed8);
    color: white;
    font-size: 16px;
    font-weight: bold;
    padding: 12px;
    border-radius: 10px;
    border: none;
    box-shadow: 0 4px 10px rgba(37,99,235,0.3);
}

/* Card hover */
.dashboard-card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
    margin-bottom: 15px;
}

.dashboard-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.12);
}
        

        
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PDF HELPER
# ─────────────────────────────────────────────
def generate_pdf(title: str, report_text: str, figures=None) -> io.BytesIO:
    """Generate an in-memory PDF and return as BytesIO."""
    buf = io.BytesIO()
    styles = getSampleStyleSheet()
    code_style = ParagraphStyle(
        "CodeStyle",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10,
    )
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    story.append(Preformatted(report_text, code_style))
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


def pdf_download_button(title: str, report: str, figs=None, filename="report.pdf"):
    pdf_buf = generate_pdf(title, report, figs)
    st.download_button(
        label="📥  Download PDF Report",
        data=pdf_buf,
        file_name=filename,
        mime="application/pdf",
        use_container_width=True,
    )


# ─────────────────────────────────────────────
# HEADER

st.markdown("""
<div style="
    background: linear-gradient(90deg,#1e3a8a,#2563eb);
    padding:25px;
    border-radius:15px;
    color:white;
    margin-bottom:20px;
    box-shadow:0 4px 15px rgba(0,0,0,0.2);
">
    <h1>📊 HS-Statistical Assistant</h1>
    <p style="font-size:18px;">
        Professional Statistical Analysis Platform for Researchers & Academics
    </p>
    <p><b>Developed by Haytham Saleh</b></p>
</div>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────
# SIDEBAR  ─  Upload + Variable Selection
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂  Load Data")
    uploaded_file = st.file_uploader("Upload Excel File (.xlsx / .xls)",
                                     type=["xlsx", "xls"])

    if uploaded_file:
        try:
            st.session_state["df"] = pd.read_excel(uploaded_file)
            df = st.session_state["df"]
            st.success(f"✅  {df.shape[0]} rows  ×  {df.shape[1]} columns")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📄 Rows", df.shape[0])

            with col2:
                st.metric("📊 Columns", df.shape[1])

            with col3:
                st.metric("❗ Missing", df.isna().sum().sum())

            with col4:
                st.metric(
        "🔢 Numeric",
        df.select_dtypes(include=np.number).shape[1]
    )
        except Exception as exc:
            st.error(f"Error reading file: {exc}")

    df: pd.DataFrame = st.session_state.get("df", None)

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
        ]
        analysis = st.selectbox("Select Analysis", ANALYSES)
        run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)
    else:
        selected_vars, dep_var, analysis, run_btn = [], "", None, False
        st.info("Please upload an Excel file to begin")


# ─────────────────────────────────────────────
# WELCOME SCREEN
# ─────────────────────────────────────────────
st.success("🚀 Upload your dataset from the left sidebar and start analysis instantly")
if df is None:
    st.markdown("""
<div class="dashboard-card">
    <h2>👋 Welcome to HS Statistical Assistant</h2>
    <p style="font-size:16px;color:#6b7280;">
        🚀 Upload your Excel dataset and start advanced statistical analysis instantly.
    </p>
</div>

Upload your **Excel file** from the sidebar to start analysing your survey / questionnaire data.

#### Available Analyses
| Category | Tests |
|---|---|
| 📊 Descriptive | Descriptive Stats, Frequency Tables |
| 📐 Normality | Shapiro-Wilk, KS, Anderson-Darling, QQ-Plot |
| 🔗 Correlation | Pearson, Spearman, Heatmap |
| α Reliability | Cronbach's Alpha, Item-Total, Alpha-if-Deleted |
| Σ Factor | KMO, Bartlett, Varimax Rotation, Scree Plot |
| t Comparative | T-Test (Cohen's d, Levene), ANOVA (Tukey), Chi-Square (Cramer's V) |
| β Regression | Simple, Multiple (Std. Beta), Logistic (OR, ROC) |
| ⟳ Mediation | Bootstrap 1000, Sobel, Path Diagram |

All reports can be **downloaded as PDF** ⬇
""")

    st.markdown("---")
    st.markdown("""
<div style='text-align:center; background:#f8fafc; color:#6b7280; font-size:0.9rem; margin-top:25px; padding:18px; border-radius:12px; border-top:3px solid #2563eb; line-height:1.8;'>               
    <b>HS-Statistical Assistant</b><br>
    Developed by <b>Haytham Saleh</b><br>
    DBA Candidate | MBA | MSc | MCTS | MCSE | SAS <br>
    📞 Phone / WhatsApp: <a href="https://wa.me/201001693305" target="_blank"><b>+20 100 169 3305</b></a><br>
    © 2026 All Rights Reserved
</div>
""", unsafe_allow_html=True)

    st.stop()

# ─────────────────────────────────────────────
# DATA PREVIEW
# ─────────────────────────────────────────────
with st.expander("📋  Data Preview  (first 30 rows)", expanded=False):
    st.dataframe(df.head(30), use_container_width=True)

if not run_btn:
    st.info("👈  Select your variables and analysis from the sidebar, then press **▶ Run Analysis**")
    st.stop()

# Guard: separator lines are not runnable
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

    data = df[selected_vars].apply(pd.to_numeric, errors="coerce")

    result = pd.DataFrame(index=selected_vars)
    result["N"]          = data.count()
    result["Missing"]    = data.isna().sum()
    result["Mean"]       = data.mean()
    result["Median"]     = data.median()
    try:
        result["Mode"]   = data.mode().iloc[0]
    except Exception:
        result["Mode"]   = np.nan
    result["Std"]        = data.std()
    result["Variance"]   = data.var()
    result["Std Error"]  = data.std() / np.sqrt(data.count())
    result["Min"]        = data.min()
    result["Max"]        = data.max()
    result["Range"]      = result["Max"] - result["Min"]
    result["Skewness"]   = data.skew()
    result["Kurtosis"]   = data.kurt()

    st.subheader("📊  Descriptive Statistics")
    st.dataframe(result.round(4), use_container_width=True)

    report  = "DESCRIPTIVE STATISTICS REPORT\n"
    report += "==============================\n\n"
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
    report = "FREQUENCY ANALYSIS REPORT\n=========================\n\n"
    first_fig = None

    for col in selected_vars:
        data = df[col]
        freq    = data.value_counts(dropna=False).sort_index()
        pct     = (freq / freq.sum()) * 100
        cum_pct = pct.cumsum()

        table = pd.DataFrame({
            "Frequency":    freq,
            "Percent (%)":  pct.round(2),
            "Cumulative %": cum_pct.round(2),
        })

        st.markdown(f"#### Variable: `{col}`")
        st.dataframe(table, use_container_width=True)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(freq.index.astype(str), freq.values,
               color="#3b82f6", edgecolor="white", linewidth=0.8)
        ax.set_title(f"Bar Chart — {col}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Category"); ax.set_ylabel("Frequency")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        if first_fig is None:
            first_fig = fig
        else:
            plt.close(fig)

        report += f"\nVARIABLE: {col}\n" + "-"*30 + "\n"
        report += table.to_string() + "\n\n"

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
    report  = "NORMALITY TESTS REPORT\n======================\n\n"
    report += f"Variables Tested: {', '.join(selected_vars)}\n\n"

    summary_rows = []
    first_fig    = None

    for col in selected_vars:
        raw  = pd.to_numeric(df[col], errors="coerce").dropna()
        n    = len(raw)
        if n < 3:
            st.warning(f"'{col}' has fewer than 3 valid values — skipped.")
            continue

        mean = raw.mean(); std = raw.std()
        skew = raw.skew(); kurt = raw.kurt()

        # Shapiro-Wilk (best for n ≤ 2000)
        if n <= 2000:
            sw_stat, sw_p = stats.shapiro(raw)
        else:
            sw_stat, sw_p = np.nan, np.nan

        ks_stat, ks_p = stats.kstest(raw, "norm", args=(mean, std))

        ad_res   = stats.anderson(raw, dist="norm")
        ad_stat  = ad_res.statistic
        ad_cv5   = ad_res.critical_values[2]
        ad_ok    = ad_stat < ad_cv5

        is_normal = (sw_p > 0.05) if not np.isnan(sw_p) else (ks_p > 0.05)
        label     = "✅ YES" if is_normal else "❌ NO"

        summary_rows.append({
            "Variable":     col,
            "N":            n,
            "Skewness":     round(skew, 3),
            "Kurtosis":     round(kurt, 3),
            "SW Stat":      round(sw_stat, 4) if not np.isnan(sw_stat) else "N/A",
            "SW p":         round(sw_p,   4) if not np.isnan(sw_p)   else "N/A",
            "KS p":         round(ks_p,   4),
            "AD Stat":      round(ad_stat, 4),
            "AD Normal?":   "Yes" if ad_ok else "No",
            "Normal?":      label,
        })

        # ── Plots ──
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Histogram + normal curve
        axes[0].hist(raw, bins=15, density=True, color="#93c5fd", edgecolor="white")
        xr = np.linspace(raw.min(), raw.max(), 200)
        axes[0].plot(xr, (1/(std*np.sqrt(2*np.pi)))*np.exp(-0.5*((xr-mean)/std)**2),
                     color="#1e40af", linewidth=2)
        axes[0].set_title(f"Histogram — {col}"); axes[0].set_ylabel("Density")

        # QQ plot
        (osm, osr), (slope, intercept, _) = stats.probplot(raw, dist="norm")
        axes[1].scatter(osm, osr, color="#a78bfa", s=18, alpha=0.75)
        axes[1].plot([osm.min(), osm.max()],
                     [slope*osm.min()+intercept, slope*osm.max()+intercept],
                     color="#5b21b6", linewidth=2)
        axes[1].set_title(f"Q-Q Plot — {col}")
        axes[1].set_xlabel("Theoretical Quantiles"); axes[1].set_ylabel("Sample Quantiles")

        # Boxplot
        axes[2].boxplot(raw, patch_artist=True,
                        boxprops=dict(facecolor="#d1fae5", color="#065f46"),
                        medianprops=dict(color="#065f46", linewidth=2))
        axes[2].set_title(f"Boxplot — {col}\nSW-p={round(sw_p,4) if not np.isnan(sw_p) else 'N/A'}")

        fig.tight_layout()
        st.pyplot(fig)
        if first_fig is None:
            first_fig = fig
        else:
            plt.close(fig)

        sw_p_str = f"{sw_p:.4f}" if not np.isnan(sw_p) else "N/A"
        report += f"\n{'─'*50}\n"
        report += f"Variable  : {col}\n"
        report += f"N={n}  Mean={mean:.4f}  Std={std:.4f}\n"
        report += f"Skewness={skew:.4f}   Kurtosis={kurt:.4f}\n"
        report += f"Shapiro-Wilk  : W={sw_stat if not np.isnan(sw_stat) else 'N/A'}  p={sw_p_str}\n"
        report += f"Kolmogorov-S  : D={ks_stat:.4f}  p={ks_p:.4f}\n"
        report += f"Anderson-D    : A={ad_stat:.4f}  Critical(5%)={ad_cv5:.4f}  Normal={ad_ok}\n"
        report += f"CONCLUSION    : {'NORMAL' if is_normal else 'NOT NORMAL'}\n"

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
    if not selected_vars:
        st.error("Please select at least one variable."); st.stop()
    for col in selected_vars:
        raw  = pd.to_numeric(df[col], errors="coerce").dropna()
        mean, std = raw.mean(), raw.std()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(raw, bins=15, density=True, color="#93c5fd", edgecolor="white")
        xr = np.linspace(raw.min(), raw.max(), 200)
        ax.plot(xr, (1/(std*np.sqrt(2*np.pi)))*np.exp(-0.5*((xr-mean)/std)**2),
                color="#1e40af", linewidth=2)
        ax.set_title(f"Histogram — {col}"); ax.set_xlabel("Value"); ax.set_ylabel("Density")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ─────────────────────────────────────────────
# 5. SCATTER PLOT
# ─────────────────────────────────────────────
elif analysis == "Scatter Plot":
    if len(selected_vars) < 2:
        st.error("Select at least 2 variables."); st.stop()
    col1, col2 = selected_vars[0], selected_vars[1]
    data = df[[col1, col2]].apply(pd.to_numeric, errors="coerce").dropna()
    b1, b0 = np.polyfit(data[col1], data[col2], 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data[col1], data[col2], alpha=0.6, color="#3b82f6")
    ax.plot(data[col1], b1*data[col1]+b0, color="#ef4444", linewidth=2)
    ax.set_xlabel(col1); ax.set_ylabel(col2)
    ax.set_title(f"Scatter Plot: {col1} vs {col2}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────
# 6. BOXPLOT
# ─────────────────────────────────────────────
elif analysis == "Boxplot":
    if not selected_vars:
        st.error("Please select at least one variable."); st.stop()
    data_list = [pd.to_numeric(df[c], errors="coerce").dropna().values
                 for c in selected_vars]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data_list, labels=selected_vars, patch_artist=True,
               boxprops=dict(facecolor="#dbeafe", color="#1e40af"),
               medianprops=dict(color="#1e40af", linewidth=2))
    ax.set_title("Boxplot"); ax.set_ylabel("Value")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────
# 7. RELIABILITY  (Cronbach Alpha)
# ─────────────────────────────────────────────
elif analysis == "Reliability  (Cronbach Alpha)":
    if len(selected_vars) < 2:
        st.error("Select at least 2 variables."); st.stop()

    data  = df[selected_vars].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] == 0:
        st.error("No valid numeric data."); st.stop()

    k           = data.shape[1]
    variances   = data.var(axis=0, ddof=1)
    total_score = data.sum(axis=1)
    total_var   = total_score.var(ddof=1)

    if total_var == 0:
        st.error("Total variance = 0. Check your data."); st.stop()

    alpha_val = (k / (k-1)) * (1 - variances.sum() / total_var)

    if   alpha_val >= 0.9: level = "Excellent"
    elif alpha_val >= 0.8: level = "Good"
    elif alpha_val >= 0.7: level = "Acceptable"
    elif alpha_val >= 0.6: level = "Questionable"
    elif alpha_val >= 0.5: level = "Poor"
    else:                   level = "Unacceptable"

    # Item-Total Correlation
    item_total   = {c: data[c].corr(total_score - data[c]) for c in selected_vars}

    # Alpha if Deleted
    alpha_deleted = {}
    for col in selected_vars:
        sub = data.drop(columns=[col])
        k2  = sub.shape[1]
        if k2 < 2:
            alpha_deleted[col] = np.nan; continue
        tv2 = sub.sum(axis=1).var(ddof=1)
        alpha_deleted[col] = (k2/(k2-1))*(1 - sub.var(ddof=1).sum()/tv2) if tv2 else np.nan

    st.subheader("α  Reliability Analysis — Cronbach's Alpha")

    c1, c2, c3 = st.columns(3)
    c1.metric("Cronbach's Alpha", f"{alpha_val:.3f}")
    c2.metric("Level", level)
    c3.metric("Items (k)", k)

    st.info(
        f"**Interpretation:** Cronbach's alpha = **{alpha_val:.3f}**, "
        f"indicating **{level}** internal consistency for the scale "
        f"({', '.join(selected_vars)})."
    )

    item_df = pd.DataFrame({
        "Item-Total Correlation": pd.Series(item_total).round(3),
        "Alpha if Item Deleted":  pd.Series(alpha_deleted).round(3),
    })
    st.markdown("#### Item Statistics")
    st.dataframe(item_df, use_container_width=True)

    st.markdown("#### Inter-Item Correlation Matrix")
    st.dataframe(data.corr().round(3), use_container_width=True)

    report  = "RELIABILITY ANALYSIS\n====================\n\n"
    report += f"Cronbach's Alpha = {alpha_val:.3f}  →  {level}\n"
    report += f"Number of Items  = {k}\n\n"
    report += "Item-Total Correlation\n"
    report += pd.Series(item_total).round(3).to_string() + "\n\n"
    report += "Alpha if Item Deleted\n"
    report += pd.Series(alpha_deleted).round(3).to_string() + "\n\n"
    report += "Inter-Item Correlation Matrix\n"
    report += data.corr().round(3).to_string()

    pdf_download_button("Reliability Analysis Report", report,
                        filename="reliability_report.pdf")


# ─────────────────────────────────────────────
# 8. FACTOR ANALYSIS
# ─────────────────────────────────────────────
elif analysis == "Factor Analysis":
    try:
        from factor_analyzer import (FactorAnalyzer,
                                     calculate_kmo,
                                     calculate_bartlett_sphericity)
    except ImportError:
        st.error("'factor_analyzer' library not found. Make sure it is in requirements.txt")
        st.stop()

    if len(selected_vars) < 3:
        st.error("Select at least 3 variables."); st.stop()

    data = df[selected_vars].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] == 0:
        st.error("No valid data."); st.stop()

    kmo_all, kmo_model    = calculate_kmo(data)
    chi2_val, bart_p      = calculate_bartlett_sphericity(data)

    fa0 = FactorAnalyzer(rotation=None)
    fa0.fit(data)
    ev, _    = fa0.get_eigenvalues()
    n_factors = max(1, int(sum(ev > 1)))

    fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
    fa.fit(data)

    loadings = pd.DataFrame(
        fa.loadings_, index=selected_vars,
        columns=[f"Factor {i+1}" for i in range(n_factors)]
    )
    communalities = pd.DataFrame(
        fa.get_communalities(), index=selected_vars, columns=["Communality"]
    )

    st.subheader("Σ  Factor Analysis  (Principal Components + Varimax)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("KMO",             f"{kmo_model:.3f}")
    c2.metric("Bartlett χ²",     f"{chi2_val:.2f}")
    c3.metric("Bartlett p",      f"{bart_p:.5f}")
    c4.metric("Factors Extracted", n_factors)

    if kmo_model < 0.5:
        st.warning("⚠️ KMO < 0.5 — The data may not be suitable for Factor Analysis.")
    elif kmo_model < 0.7:
        st.info("KMO is Mediocre (0.5–0.7). Proceed with caution.")
    else:
        st.success("KMO is acceptable. Factor Analysis is appropriate.")

    # Scree Plot
    fig_scree, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(ev)+1), ev, marker="o", color="#3b82f6", linewidth=2)
    ax.axhline(y=1, color="red", linestyle="--", linewidth=1.5, label="Eigenvalue = 1")
    ax.set_title("Scree Plot", fontsize=13, fontweight="bold")
    ax.set_xlabel("Factor Number"); ax.set_ylabel("Eigenvalue")
    ax.legend()
    fig_scree.tight_layout()
    st.pyplot(fig_scree)

    st.markdown("#### Rotated Factor Loadings  (Varimax)")
    st.dataframe(loadings.round(3), use_container_width=True)

    st.markdown("#### Communalities")
    st.dataframe(communalities.round(3), use_container_width=True)

    ev_series = pd.Series(ev, index=[f"Factor {i+1}" for i in range(len(ev))])
    st.markdown("#### Eigenvalues")
    st.dataframe(ev_series.round(3).to_frame("Eigenvalue"), use_container_width=True)

    report  = "FACTOR ANALYSIS REPORT\n======================\n\n"
    report += f"KMO Measure of Sampling Adequacy  = {kmo_model:.3f}\n"
    report += f"Bartlett's Test — Chi-Square       = {chi2_val:.3f}\n"
    report += f"Bartlett's p-value                 = {bart_p:.5f}\n"
    report += f"Factors Extracted (Eigenvalue > 1) = {n_factors}\n\n"
    report += "Eigenvalues\n" + ev_series.round(3).to_string() + "\n\n"
    report += "Rotated Factor Loadings (Varimax)\n"
    report += loadings.round(3).to_string() + "\n\n"
    report += "Communalities\n"
    report += communalities.round(3).to_string()

    pdf_download_button("Factor Analysis Report", report,
                        figs=[fig_scree], filename="factor_analysis_report.pdf")
    plt.close(fig_scree)


# ─────────────────────────────────────────────
# 9. CORRELATION ANALYSIS
# ─────────────────────────────────────────────
elif analysis == "Correlation Analysis":
    if len(selected_vars) < 2:
        st.error("Select at least 2 variables."); st.stop()

    data = df[selected_vars].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] == 0:
        st.error("No valid data."); st.stop()

    N = data.shape[0]

    # Build matrices
    def build_matrix(method="pearson"):
        corr_m = pd.DataFrame(index=selected_vars, columns=selected_vars, dtype=object)
        p_m    = pd.DataFrame(index=selected_vars, columns=selected_vars, dtype=object)
        for i in selected_vars:
            for j in selected_vars:
                if i == j:
                    corr_m.loc[i,j] = "1.000"; p_m.loc[i,j] = "—"
                else:
                    if method == "pearson":
                        r, p = stats.pearsonr(data[i], data[j])
                    else:
                        r, p = stats.spearmanr(data[i], data[j])
                    corr_m.loc[i,j] = f"{r:.3f}"
                    p_m.loc[i,j]    = "<.001" if p < 0.001 else f"{p:.3f}"
        return corr_m, p_m

    p_corr, p_sig     = build_matrix("pearson")
    s_corr, s_sig     = build_matrix("spearman")

    st.subheader("r  Correlation Analysis")
    st.metric("Sample Size (N)", N)

    tab1, tab2 = st.tabs(["Pearson", "Spearman"])
    with tab1:
        st.markdown("**Pearson Correlation Matrix**")
        st.dataframe(p_corr, use_container_width=True)
        st.markdown("**Pearson Significance (p-value)**")
        st.dataframe(p_sig,  use_container_width=True)
    with tab2:
        st.markdown("**Spearman Correlation Matrix**")
        st.dataframe(s_corr, use_container_width=True)
        st.markdown("**Spearman Significance (p-value)**")
        st.dataframe(s_sig,  use_container_width=True)

    # Heatmap
    num_corr = data.corr()
    fig_heat, ax = plt.subplots(figsize=(max(6, len(selected_vars)), max(5, len(selected_vars)-1)))
    im = ax.imshow(num_corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(selected_vars))); ax.set_yticks(range(len(selected_vars)))
    ax.set_xticklabels(selected_vars, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(selected_vars, fontsize=9)
    for i in range(len(selected_vars)):
        for j in range(len(selected_vars)):
            ax.text(j, i, f"{num_corr.values[i,j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(num_corr.values[i,j]) > 0.6 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Pearson Correlation Heatmap", fontsize=13, fontweight="bold")
    fig_heat.tight_layout()
    st.pyplot(fig_heat)

    report  = "CORRELATION ANALYSIS REPORT\n===========================\n\n"
    report += f"Sample Size (N) = {N}\n\n"
    report += "Pearson Correlation Matrix\n" + p_corr.to_string() + "\n\n"
    report += "Pearson p-values\n"            + p_sig.to_string()  + "\n\n"
    report += "Spearman Correlation Matrix\n"  + s_corr.to_string() + "\n\n"
    report += "Spearman p-values\n"            + s_sig.to_string()

    pdf_download_button("Correlation Analysis Report", report,
                        figs=[fig_heat], filename="correlation_report.pdf")
    plt.close(fig_heat)


# ─────────────────────────────────────────────
# 10. MULTICOLLINEARITY  (VIF)
# ─────────────────────────────────────────────
elif analysis == "Multicollinearity  (VIF)":
    if len(selected_vars) < 2:
        st.error("Select at least 2 variables."); st.stop()

    data = df[selected_vars].apply(pd.to_numeric, errors="coerce").dropna()
    X = sm.add_constant(data)

    vif = pd.DataFrame()
    vif["Variable"]       = X.columns
    vif["VIF"]            = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["Interpretation"] = vif["VIF"].apply(
        lambda x: "✅ Low (<5)"      if x < 5  else
                  ("⚠️ Moderate (5-10)" if x < 10 else
                   "❌ High (>10)")
    )

    st.subheader("V  Multicollinearity — Variance Inflation Factor (VIF)")
    st.dataframe(vif.round(3), use_container_width=True)
    st.caption("Rule: VIF < 5 = OK  |  5–10 = Moderate  |  > 10 = Serious Multicollinearity")

    report  = "MULTICOLLINEARITY (VIF) REPORT\n==============================\n\n"
    report += vif.round(3).to_string()
    report += "\n\nRule of Thumb:\n  VIF < 5     → No multicollinearity\n"
    report +=   "  VIF 5–10   → Moderate\n"
    report +=   "  VIF > 10   → High multicollinearity\n"

    pdf_download_button("VIF Report", report, filename="vif_report.pdf")


# ─────────────────────────────────────────────
# 11. T-TEST
# ─────────────────────────────────────────────
elif analysis == "T-Test":
    if dep_var == "" or len(selected_vars) != 1:
        st.error("Select ONE grouping variable from the list AND set Dependent Variable below.")
        st.stop()

    group_var = selected_vars[0]
    data = df[[dep_var, group_var]].copy()
    data[dep_var] = pd.to_numeric(data[dep_var], errors="coerce")
    data.dropna(inplace=True)

    groups = data[group_var].unique()
    if len(groups) > 10:
        st.error(f"'{group_var}' has {len(groups)} unique values — T-Test needs exactly 2 groups.")
        st.stop()
    if len(groups) != 2:
        st.error(f"T-Test requires exactly 2 groups. Found: {len(groups)}."); st.stop()

    g1 = data[data[group_var] == groups[0]][dep_var]
    g2 = data[data[group_var] == groups[1]][dep_var]

    # Levene
    lev_stat, lev_p = stats.levene(g1, g2)
    equal_var        = lev_p > 0.05

    # Independent t-test
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var)
    p_one         = p_val / 2

    # Cohen's d
    pooled = np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) / (len(g1)+len(g2)-2))
    d      = (g1.mean() - g2.mean()) / pooled
    if   abs(d) >= 0.8: eff = "Large"
    elif abs(d) >= 0.5: eff = "Medium"
    else:               eff = "Small"

    # 95% CI for difference
    se_diff = np.sqrt(g1.var()/len(g1) + g2.var()/len(g2))
    ci_lo   = (g1.mean()-g2.mean()) - 1.96*se_diff
    ci_hi   = (g1.mean()-g2.mean()) + 1.96*se_diff

    # Paired (on matched subset)
    min_n           = min(len(g1), len(g2))
    t_pair, p_pair  = stats.ttest_rel(g1.values[:min_n], g2.values[:min_n])

    # Descriptives
    desc = data.groupby(group_var)[dep_var].agg(
        N="count", Mean="mean", Std="std",
        SE=lambda x: x.std()/np.sqrt(len(x)),
        Min="min", Max="max"
    ).round(4)

    sig_label = "statistically significant" if p_val < 0.05 else "NOT statistically significant"
    var_label = "Equal variances assumed"    if equal_var    else "Equal variances NOT assumed"

    interpretation = (
        f"An independent samples t-test compared {dep_var} between "
        f"'{groups[0]}' and '{groups[1]}'. "
        f"Levene's test indicated {var_label} (F={lev_stat:.3f}, p={lev_p:.3f}). "
        f"The difference was {sig_label} "
        f"(t={t_stat:.3f}, df={len(g1)+len(g2)-2}, p={p_val:.4f}). "
        f"Cohen's d = {d:.3f} — {eff} effect size."
    )

    st.subheader("t  Independent Samples T-Test")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("t-statistic",       f"{t_stat:.4f}")
    c2.metric("p-value (2-tailed)",f"{p_val:.4f}")
    c3.metric("Cohen's d",          f"{d:.4f}")
    c4.metric("Effect Size",        eff)

    st.markdown("#### Group Descriptives")
    st.dataframe(desc, use_container_width=True)

    st.markdown("#### Levene's Test for Equality of Variances")
    st.dataframe(
        pd.DataFrame({"F":[lev_stat],"p-value":[lev_p],
                      "Equal Variances?":[var_label]}).round(4),
        use_container_width=True
    )

    st.markdown(f"#### 95% CI for Mean Difference:  [{ci_lo:.4f},  {ci_hi:.4f}]")

    fig_t, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.boxplot([g1, g2], labels=[str(groups[0]), str(groups[1])],
                patch_artist=True,
                boxprops=dict(facecolor="#dbeafe", color="#1e40af"),
                medianprops=dict(color="#1e40af", linewidth=2))
    ax1.set_title(f"Boxplot: {dep_var} by {group_var}")
    ax1.set_ylabel(dep_var)

    means = [g1.mean(), g2.mean()]
    sems  = [g1.sem(),  g2.sem()]
    ax2.bar([str(groups[0]), str(groups[1])], means, yerr=sems, capsize=6,
            color=["#3b82f6","#f59e0b"], edgecolor="white")
    ax2.set_title(f"Means ± SE:  {dep_var}")
    ax2.set_ylabel(dep_var)

    fig_t.tight_layout()
    st.pyplot(fig_t)

    st.info(f"📝  **Interpretation:** {interpretation}")

    report  = "INDEPENDENT SAMPLES T-TEST REPORT\n==================================\n\n"
    report += f"Dependent Variable : {dep_var}\n"
    report += f"Grouping Variable  : {group_var}\n"
    report += f"Groups             : {groups[0]}  vs  {groups[1]}\n\n"
    report += "GROUP DESCRIPTIVES\n"    + desc.to_string()        + "\n\n"
    report += "LEVENE'S TEST\n"
    report += f"F = {lev_stat:.4f}   p = {lev_p:.4f}   {var_label}\n\n"
    report += "T-TEST RESULTS\n"
    report += f"t = {t_stat:.4f}\n"
    report += f"df = {len(g1)+len(g2)-2}\n"
    report += f"Sig (2-tailed) = {p_val:.4f}\n"
    report += f"Sig (1-tailed) = {p_one:.4f}\n"
    report += f"Mean Difference = {g1.mean()-g2.mean():.4f}\n"
    report += f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]\n\n"
    report += "EFFECT SIZE\n"
    report += f"Cohen's d = {d:.4f}  ({eff})\n\n"
    report += "PAIRED T-TEST (matched subset)\n"
    report += f"t = {t_pair:.4f}   p = {p_pair:.4f}\n\n"
    report += "INTERPRETATION\n" + interpretation

    pdf_download_button("T-Test Report", report, figs=[fig_t],
                        filename="ttest_report.pdf")
    plt.close(fig_t)


# ─────────────────────────────────────────────
# 12. CHI-SQUARE
# ─────────────────────────────────────────────
elif analysis == "Chi-Square":
    if len(selected_vars) != 2:
        st.error("Select exactly TWO variables for Chi-Square."); st.stop()

    var1, var2 = selected_vars[0], selected_vars[1]
    data       = df[[var1, var2]].dropna()
    cross_tab  = pd.crosstab(data[var1], data[var2])

    chi2_s, p_val, dof, expected = stats.chi2_contingency(cross_tab)

    n        = data.shape[0]
    min_dim  = min(cross_tab.shape) - 1
    cramers_v = np.sqrt(chi2_s / (n * min_dim)) if min_dim > 0 else 0
    if   cramers_v >= 0.5: eff_level = "Strong"
    elif cramers_v >= 0.3: eff_level = "Moderate"
    elif cramers_v >= 0.1: eff_level = "Weak"
    else:                   eff_level = "Negligible"

    exp_df   = pd.DataFrame(expected, index=cross_tab.index, columns=cross_tab.columns)
    low_exp  = (exp_df < 5).values.sum()
    pct_low  = (low_exp / exp_df.size) * 100
    row_pct  = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

    sig_label = "statistically significant" if p_val < 0.05 else "NOT statistically significant"
    interpretation = (
        f"A Chi-Square test of independence was conducted between '{var1}' and '{var2}'. "
        f"The association was {sig_label} "
        f"(χ²={chi2_s:.3f}, df={dof}, p={p_val:.4f}). "
        f"Cramer's V = {cramers_v:.3f} — {eff_level} association."
    )
    if pct_low > 20:
        interpretation += (
            f"\n⚠️ WARNING: {pct_low:.1f}% of expected frequencies < 5. "
            "Consider Fisher's Exact Test or merging categories."
        )

    st.subheader("χ²  Chi-Square Test of Independence")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Chi-Square (χ²)", f"{chi2_s:.4f}")
    c2.metric("p-value",          f"{p_val:.4f}")
    c3.metric("df",               dof)
    c4.metric("Cramer's V",       f"{cramers_v:.4f}")

    tab1, tab2, tab3 = st.tabs(["Observed", "Expected", "Row %"])
    with tab1: st.dataframe(cross_tab,             use_container_width=True)
    with tab2: st.dataframe(exp_df.round(2),        use_container_width=True)
    with tab3: st.dataframe(row_pct.round(2),       use_container_width=True)

    # Grouped bar
    colors = ["#3b82f6","#f59e0b","#10b981","#ef4444","#8b5cf6","#ec4899"]
    fig_chi, ax = plt.subplots(figsize=(9, 4))
    x     = np.arange(len(cross_tab.index))
    width = 0.8 / len(cross_tab.columns)
    for i, col in enumerate(cross_tab.columns):
        ax.bar(x + i*width, cross_tab[col], width=width,
               label=str(col), color=colors[i % len(colors)], edgecolor="white")
    ax.set_xticks(x + width*(len(cross_tab.columns)-1)/2)
    ax.set_xticklabels([str(v) for v in cross_tab.index], rotation=30)
    ax.set_title(f"Chi-Square: {var1} × {var2}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Frequency"); ax.legend(title=var2)
    fig_chi.tight_layout()
    st.pyplot(fig_chi)

    st.info(f"📝  **Interpretation:** {interpretation}")

    report  = "CHI-SQUARE TEST REPORT\n======================\n\n"
    report += f"Variable 1 (Rows)    : {var1}\n"
    report += f"Variable 2 (Columns) : {var2}\n"
    report += f"Sample Size (N)      : {n}\n\n"
    report += f"Chi-Square (χ²) = {chi2_s:.4f}\n"
    report += f"df              = {dof}\n"
    report += f"p-value         = {p_val:.4f}\n"
    report += f"Cramer's V      = {cramers_v:.4f}  ({eff_level})\n\n"
    report += "Contingency Table\n"    + cross_tab.to_string()    + "\n\n"
    report += "Expected Frequencies\n" + exp_df.round(2).to_string() + "\n\n"
    report += "Row Percentages\n"      + row_pct.round(2).to_string() + "\n\n"
    report += "INTERPRETATION\n"       + interpretation

    pdf_download_button("Chi-Square Report", report, figs=[fig_chi],
                        filename="chi_square_report.pdf")
    plt.close(fig_chi)


# ─────────────────────────────────────────────
# 13. ANOVA  (One-Way)
# ─────────────────────────────────────────────
elif analysis == "ANOVA  (One-Way)":
    if dep_var == "" or len(selected_vars) != 1:
        st.error("Select ONE grouping variable AND set Dependent Variable."); st.stop()

    group_var = selected_vars[0]
    data      = df[[dep_var, group_var]].dropna()
    data[dep_var] = pd.to_numeric(data[dep_var], errors="coerce")
    data.dropna(inplace=True)

    groups_data = []
    labels      = []
    for g in sorted(data[group_var].unique()):
        grp = data[data[group_var] == g][dep_var]
        if len(grp) > 0:
            groups_data.append(grp); labels.append(str(g))

    if len(groups_data) < 2:
        st.error("Need at least 2 groups."); st.stop()

    F, p = stats.f_oneway(*groups_data)

    grand_mean = data[dep_var].mean()
    ss_between = sum(len(g)*(g.mean()-grand_mean)**2 for g in groups_data)
    ss_total   = ((data[dep_var] - grand_mean)**2).sum()
    eta_sq     = ss_between / ss_total

    tukey = pairwise_tukeyhsd(endog=data[dep_var], groups=data[group_var], alpha=0.05)

    means = data.groupby(group_var)[dep_var].mean()
    sds   = data.groupby(group_var)[dep_var].std()
    cnts  = data.groupby(group_var)[dep_var].count()
    cis   = 1.96 * sds / np.sqrt(cnts)

    tbl = pd.DataFrame({"N": cnts, "Mean": means, "Std": sds, "CI95": cis})

    st.subheader("F  One-Way ANOVA")

    c1,c2,c3 = st.columns(3)
    c1.metric("F-statistic",  f"{F:.4f}")
    c2.metric("p-value",       f"{p:.5f}")
    c3.metric("Eta Squared η²",f"{eta_sq:.4f}")

    st.markdown("#### Group Statistics")
    st.dataframe(tbl.round(4), use_container_width=True)

    st.markdown("#### Post-Hoc: Tukey HSD")
    tukey_df = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )
    st.dataframe(tukey_df, use_container_width=True)

    fig_an, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(range(len(means)), means, yerr=cis,
                fmt="o-", capsize=6, color="#3b82f6",
                ecolor="#1e40af", linewidth=2)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(labels)
    ax.set_title("Means Plot with 95% CI", fontsize=13, fontweight="bold")
    ax.set_xlabel(group_var); ax.set_ylabel(dep_var)
    fig_an.tight_layout()
    st.pyplot(fig_an)

    report  = "ONE-WAY ANOVA REPORT\n====================\n\n"
    report += f"Dependent Variable : {dep_var}\n"
    report += f"Grouping Variable  : {group_var}\n\n"
    report += f"F = {F:.4f}\np = {p:.5f}\nEta Squared = {eta_sq:.4f}\n\n"
    report += "Group Statistics\n" + tbl.round(4).to_string() + "\n\n"
    report += "Post-Hoc Tukey HSD\n" + tukey.summary().as_text()

    pdf_download_button("ANOVA Report", report, figs=[fig_an],
                        filename="anova_report.pdf")
    plt.close(fig_an)


# ─────────────────────────────────────────────
# 14. SIMPLE REGRESSION
# ─────────────────────────────────────────────
elif analysis == "Simple Regression":
    if dep_var == "" or len(selected_vars) != 1:
        st.error("Select ONE independent variable AND set Dependent Variable."); st.stop()

    data = df[[dep_var] + selected_vars].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] == 0:
        st.error("No valid data."); st.stop()

    y  = data[dep_var]
    X  = sm.add_constant(data[selected_vars])
    md = sm.OLS(y, X).fit()

    ssr = md.ess;  sse = md.ssr;  sst = ssr + sse
    df_r = int(md.df_model);  df_e = int(md.df_resid)
    msr = ssr/df_r;            mse = sse/df_e
    dw  = sm.stats.stattools.durbin_watson(md.resid)

    an_tbl = pd.DataFrame({
        "SS": [ssr, sse, sst],
        "df": [df_r, df_e, df_r+df_e],
        "MS": [msr, mse, ""],
        "F":  [md.fvalue, "", ""],
        "Sig":[md.f_pvalue, "", ""],
    }, index=["Regression","Residual","Total"])

    coef_tbl = pd.DataFrame({"Beta": md.params,"t": md.tvalues,"Sig": md.pvalues})

    st.subheader("β  Simple Linear Regression")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("R²",            f"{md.rsquared:.3f}")
    c2.metric("Adjusted R²",    f"{md.rsquared_adj:.3f}")
    c3.metric("F-statistic",    f"{md.fvalue:.3f}")
    c4.metric("Durbin-Watson",  f"{dw:.3f}")

    st.markdown("#### ANOVA Table")
    st.dataframe(an_tbl.round(4), use_container_width=True)

    st.markdown("#### Coefficients")
    st.dataframe(coef_tbl.round(4), use_container_width=True)

    fig_sr, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.scatter(md.fittedvalues, md.resid, alpha=0.6, color="#3b82f6")
    ax1.axhline(0, color="red", linestyle="--")
    ax1.set_title("Residual Plot"); ax1.set_xlabel("Predicted"); ax1.set_ylabel("Residual")

    ax2.hist(md.resid, bins=15, color="#a78bfa", edgecolor="white")
    ax2.set_title("Histogram of Residuals"); ax2.set_xlabel("Residual")

    fig_sr.tight_layout()
    st.pyplot(fig_sr)

    report  = "SIMPLE REGRESSION REPORT\n========================\n\n"
    report += f"N = {data.shape[0]}\nDependent  : {dep_var}\nIndependent: {selected_vars[0]}\n\n"
    report += f"R²               = {md.rsquared:.3f}\n"
    report += f"Adjusted R²      = {md.rsquared_adj:.3f}\n"
    report += f"F-statistic      = {md.fvalue:.3f}\n"
    report += f"Sig (F)          = {md.f_pvalue:.5f}\n"
    report += f"Durbin-Watson    = {dw:.3f}\n\n"
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
        st.error("Select at least TWO independent variables AND set Dependent Variable."); st.stop()

    data = df[[dep_var] + selected_vars].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] == 0:
        st.error("No valid data."); st.stop()

    y   = data[dep_var]
    X   = sm.add_constant(data[selected_vars])
    md  = sm.OLS(y, X).fit()

    # Standardised Beta
    y_std = (y - y.mean()) / y.std()
    X_std = (data[selected_vars] - data[selected_vars].mean()) / data[selected_vars].std()
    md_std = sm.OLS(y_std, sm.add_constant(X_std)).fit()

    dw   = sm.stats.stattools.durbin_watson(md.resid)

    coef_tbl = pd.DataFrame({
        "Unstd. Beta":  md.params,
        "Std. Beta":    md_std.params,
        "t":            md.tvalues,
        "Sig":          md.pvalues,
    })

    vif_df = pd.DataFrame({
        "Variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })

    st.subheader("β+  Multiple Linear Regression")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("R²",           f"{md.rsquared:.3f}")
    c2.metric("Adjusted R²",   f"{md.rsquared_adj:.3f}")
    c3.metric("F-statistic",   f"{md.fvalue:.3f}")
    c4.metric("Durbin-Watson", f"{dw:.3f}")

    st.markdown("#### Coefficients  (Unstandardised + Standardised Beta)")
    st.dataframe(coef_tbl.round(4), use_container_width=True)

    st.markdown("#### VIF — Multicollinearity")
    st.dataframe(vif_df.round(3), use_container_width=True)

    fig_mr, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(md.fittedvalues, md.resid, alpha=0.6, color="#3b82f6")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title("Residual Plot")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")

    axes[1].hist(md.resid, bins=15, color="#a78bfa", edgecolor="white")
    axes[1].set_title("Histogram of Residuals")

    (osm, osr), (slope, intercept, _) = stats.probplot(md.resid, dist="norm")
    axes[2].scatter(osm, osr, s=20, alpha=0.7, color="#10b981")
    axes[2].plot([osm.min(), osm.max()],
                 [slope*osm.min()+intercept, slope*osm.max()+intercept],
                 color="red", linewidth=2)
    axes[2].set_title("QQ Plot — Residuals")

    fig_mr.tight_layout()
    st.pyplot(fig_mr)

    report  = "MULTIPLE REGRESSION REPORT\n==========================\n\n"
    report += f"N = {data.shape[0]}\nDependent  : {dep_var}\n"
    report += f"Independent: {', '.join(selected_vars)}\n\n"
    report += f"R²            = {md.rsquared:.3f}\n"
    report += f"Adjusted R²   = {md.rsquared_adj:.3f}\n"
    report += f"F             = {md.fvalue:.3f}\n"
    report += f"Sig (F)       = {md.f_pvalue:.5f}\n"
    report += f"Durbin-Watson = {dw:.3f}\n\n"
    report += "COEFFICIENTS\n" + coef_tbl.round(4).to_string() + "\n\n"
    report += "VIF\n"          + vif_df.round(3).to_string()

    pdf_download_button("Multiple Regression Report", report, figs=[fig_mr],
                        filename="multiple_regression_report.pdf")
    plt.close(fig_mr)


# ─────────────────────────────────────────────
# 16. LOGISTIC REGRESSION
# ─────────────────────────────────────────────
elif analysis == "Logistic Regression":
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

    if dep_var == "" or len(selected_vars) < 1:
        st.error("Select at least ONE independent variable AND set Dependent Variable."); st.stop()

    data = df[[dep_var] + selected_vars].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] == 0:
        st.error("No valid data."); st.stop()

    if data[dep_var].nunique() != 2:
        st.error(f"Dependent variable must have exactly 2 categories. "
                 f"Found: {data[dep_var].nunique()}."); st.stop()

    y = data[dep_var]
    X = sm.add_constant(data[selected_vars])

    try:
        md = sm.Logit(y, X).fit(disp=0)
    except Exception as exc:
        st.error(f"Model fitting failed: {exc}"); st.stop()

    or_vals = np.exp(md.params)
    conf    = np.exp(md.conf_int())
    conf.columns = ["OR Lower 95%", "OR Upper 95%"]

    coef_tbl = pd.DataFrame({
        "Coefficient":  md.params,
        "Std Error":    md.bse,
        "Wald (z)":     md.tvalues,
        "Sig (p)":      md.pvalues,
        "Odds Ratio":   or_vals,
        "OR Lower 95%": conf["OR Lower 95%"],
        "OR Upper 95%": conf["OR Upper 95%"],
    })

    ll_null  = md.llnull;  ll_model = md.llf
    n        = data.shape[0]
    cox_snell  = 1 - np.exp((2/n)*(ll_null-ll_model))
    nagelkerke = cox_snell / (1 - np.exp((2/n)*ll_null))

    y_pred_prob = md.predict(X)
    y_pred      = (y_pred_prob >= 0.5).astype(int)
    cm          = confusion_matrix(y, y_pred)
    accuracy    = np.trace(cm) / cm.sum() * 100

    try:
        auc_val = roc_auc_score(y, y_pred_prob)
        fpr, tpr, _ = roc_curve(y, y_pred_prob)
        has_roc = True
    except Exception:
        has_roc = False; auc_val = np.nan

    labels_cls = sorted(y.unique())
    cm_df = pd.DataFrame(cm,
                         index  =[f"Actual {l}"    for l in labels_cls],
                         columns=[f"Predicted {l}" for l in labels_cls])

    st.subheader("L  Logistic Regression")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Cox & Snell R²", f"{cox_snell:.4f}")
    c2.metric("Nagelkerke R²",  f"{nagelkerke:.4f}")
    c3.metric("Accuracy",       f"{accuracy:.2f}%")
    c4.metric("AUC",            f"{auc_val:.4f}" if has_roc else "N/A")

    st.markdown("#### Coefficients & Odds Ratios")
    st.dataframe(coef_tbl.round(4), use_container_width=True)

    st.markdown("#### Classification Table")
    st.dataframe(cm_df, use_container_width=True)

    fig_lr, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Odds Ratio Forest Plot
    vp = [v for v in coef_tbl.index if v != "const"]
    ov = or_vals[vp]
    el = ov.values - conf.loc[vp, "OR Lower 95%"].values
    eu = conf.loc[vp, "OR Upper 95%"].values - ov.values
    axes[0].barh(vp, ov, xerr=[el, eu], color="#a78bfa", edgecolor="#5b21b6", capsize=5)
    axes[0].axvline(x=1, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_title("Odds Ratio Plot with 95% CI"); axes[0].set_xlabel("Odds Ratio")

    # ROC Curve
    if has_roc:
        axes[1].plot(fpr, tpr, color="#1e40af", linewidth=2, label=f"AUC = {auc_val:.3f}")
        axes[1].plot([0,1],[0,1], color="gray", linestyle="--")
        axes[1].set_title("ROC Curve")
        axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "ROC N/A", ha="center", va="center", fontsize=12)

    fig_lr.tight_layout()
    st.pyplot(fig_lr)

    report  = "LOGISTIC REGRESSION REPORT\n==========================\n\n"
    report += f"N = {n}\nDependent  : {dep_var}\n"
    report += f"Independent: {', '.join(selected_vars)}\n\n"
    report += f"Cox & Snell R² = {cox_snell:.4f}\n"
    report += f"Nagelkerke R²  = {nagelkerke:.4f}\n"
    report += f"Accuracy       = {accuracy:.2f}%\n"
    if has_roc: report += f"AUC            = {auc_val:.4f}\n"
    report += "\nCOEFFICIENTS & ODDS RATIOS\n" + coef_tbl.round(4).to_string() + "\n\n"
    report += "CLASSIFICATION TABLE\n"          + cm_df.to_string()

    pdf_download_button("Logistic Regression Report", report, figs=[fig_lr],
                        filename="logistic_regression_report.pdf")
    plt.close(fig_lr)


# ─────────────────────────────────────────────
# 17. MEDIATION ANALYSIS
# ─────────────────────────────────────────────
elif analysis == "Mediation Analysis":
    if dep_var == "" or len(selected_vars) != 2:
        st.error(
            "Mediation Analysis needs:\n"
            "• Variable 1 selected = Independent (X)\n"
            "• Variable 2 selected = Mediator (M)\n"
            "• Dependent Variable (Y) set below"
        ); st.stop()

    X_name, M_name, Y_name = selected_vars[0], selected_vars[1], dep_var

    data = df[[X_name, M_name, Y_name]].apply(pd.to_numeric, errors="coerce").dropna()
    if data.shape[0] == 0:
        st.error("No valid data."); st.stop()
    if data.shape[0] < 30:
        st.warning(f"Small sample (N={data.shape[0]}). N ≥ 100 recommended for mediation.")

    X = data[X_name]; M = data[M_name]; Y = data[Y_name]

    # Path a: X → M
    Xc    = sm.add_constant(X)
    md_a  = sm.OLS(M, Xc).fit()
    a     = md_a.params[X_name];  a_se = md_a.bse[X_name]
    a_p   = md_a.pvalues[X_name]

    # Path b + c': X + M → Y
    XMc   = sm.add_constant(data[[X_name, M_name]])
    md_b  = sm.OLS(Y, XMc).fit()
    b     = md_b.params[M_name];  b_se = md_b.bse[M_name]
    b_p   = md_b.pvalues[M_name]
    c_prime   = md_b.params[X_name]
    cp_p      = md_b.pvalues[X_name]

    # Path c: X → Y (total)
    md_c  = sm.OLS(Y, Xc).fit()
    c     = md_c.params[X_name]
    c_p   = md_c.pvalues[X_name]

    indirect = a * b
    sobel_se = np.sqrt(b**2 * a_se**2 + a**2 * b_se**2)
    sobel_z  = indirect / sobel_se if sobel_se != 0 else 0
    sobel_p  = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
    prop_med = (indirect / c * 100) if c != 0 else 0

    # Bootstrap (1 000 iterations)
    np.random.seed(42)
    boot_ind = []
    for _ in range(1000):
        idx  = np.random.choice(len(data), len(data), replace=True)
        bd   = data.iloc[idx]
        try:
            ba = sm.OLS(bd[M_name], sm.add_constant(bd[X_name])).fit().params[X_name]
            bb = sm.OLS(bd[Y_name], sm.add_constant(bd[[X_name, M_name]])).fit().params[M_name]
            boot_ind.append(ba * bb)
        except Exception:
            pass

    bci_lo  = np.percentile(boot_ind, 2.5)
    bci_hi  = np.percentile(boot_ind, 97.5)
    boot_sig = "Significant ✅" if (bci_lo > 0 or bci_hi < 0) else "Not Significant ❌"

    st.subheader("⟳  Mediation Analysis  (Bootstrap 1 000 iterations)")

    c1,c2,c3 = st.columns(3)
    c1.metric("Path a  (X → M)",  f"{a:.4f}")
    c2.metric("Path b  (M → Y)",  f"{b:.4f}")
    c3.metric("Direct c'  (X → Y)", f"{c_prime:.4f}")

    c4,c5,c6 = st.columns(3)
    c4.metric("Total Effect  c", f"{c:.4f}")
    c5.metric("Indirect  a×b",   f"{indirect:.4f}")
    c6.metric("Proportion Mediated", f"{prop_med:.1f}%")

    paths_tbl = pd.DataFrame({
        "Path": ["a: X → M", "b: M → Y", "c': Direct X → Y", "c: Total X → Y"],
        "Coefficient": [a, b, c_prime, c],
        "p-value":     [a_p, b_p, cp_p, c_p],
    })
    st.markdown("#### Path Coefficients")
    st.dataframe(paths_tbl.round(4), use_container_width=True)

    boot_tbl = pd.DataFrame({
        "Indirect Effect (a×b)": [round(indirect, 4)],
        "Bootstrap CI Lower":    [round(bci_lo, 4)],
        "Bootstrap CI Upper":    [round(bci_hi, 4)],
        "Sobel Z":               [round(sobel_z, 4)],
        "Sobel p":               [round(sobel_p, 4)],
        "Bootstrap Result":      [boot_sig],
    })
    st.markdown("#### Indirect Effect & Bootstrap 95% CI")
    st.dataframe(boot_tbl, use_container_width=True)

    # Path Diagram
    fig_med, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    # Boxes
    for (x0, y0, label, fc, ec) in [
        (0.3, 2.0, X_name, "#dbeafe", "#1e40af"),
        (3.8, 4.0, M_name, "#d1fae5", "#065f46"),
        (7.4, 2.0, Y_name, "#fef3c7", "#92400e"),
    ]:
        ax.add_patch(plt.Rectangle((x0, y0), 2.2, 1.4,
                                   facecolor=fc, edgecolor=ec, linewidth=2, zorder=3))
        ax.text(x0+1.1, y0+0.7, label, ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=4)

    # Arrows
    ax.annotate("", xy=(3.8, 4.6), xytext=(2.5, 3.1),
                arrowprops=dict(arrowstyle="->", color="#065f46", lw=2.5))
    ax.text(2.8, 4.0, f"a = {a:.3f}\np = {a_p:.3f}", fontsize=9, color="#065f46")

    ax.annotate("", xy=(7.4, 3.1), xytext=(6.0, 4.6),
                arrowprops=dict(arrowstyle="->", color="#065f46", lw=2.5))
    ax.text(7.0, 4.2, f"b = {b:.3f}\np = {b_p:.3f}", fontsize=9, color="#065f46")

    ax.annotate("", xy=(7.4, 2.7), xytext=(2.5, 2.7),
                arrowprops=dict(arrowstyle="->", color="#92400e", lw=2.5))
    ax.text(4.8, 2.3,
            f"c' = {c_prime:.3f} (p={cp_p:.3f})\nc = {c:.3f} (p={c_p:.3f})",
            fontsize=9, color="#92400e", ha="center")

    ax.set_title(
        f"Mediation Path Diagram\n"
        f"Indirect = {indirect:.4f}   Bootstrap 95% CI [{bci_lo:.4f}, {bci_hi:.4f}]",
        fontsize=11, fontweight="bold"
    )
    fig_med.tight_layout()
    st.pyplot(fig_med)

    report  = "MEDIATION ANALYSIS REPORT\n=========================\n\n"
    report += f"X (Independent) = {X_name}\n"
    report += f"M (Mediator)    = {M_name}\n"
    report += f"Y (Dependent)   = {Y_name}\n"
    report += f"N               = {data.shape[0]}\n\n"
    report += "PATH COEFFICIENTS\n"
    report += f"Path a  (X→M)         = {a:.4f}   p = {a_p:.4f}\n"
    report += f"Path b  (M→Y)         = {b:.4f}   p = {b_p:.4f}\n"
    report += f"Direct c' (X→Y)       = {c_prime:.4f}   p = {cp_p:.4f}\n"
    report += f"Total Effect c (X→Y)  = {c:.4f}   p = {c_p:.4f}\n\n"
    report += "INDIRECT EFFECT\n"
    report += f"a × b              = {indirect:.4f}\n"
    report += f"Sobel Z            = {sobel_z:.4f}   p = {sobel_p:.4f}\n"
    report += f"Bootstrap 95% CI   = [{bci_lo:.4f},  {bci_hi:.4f}]\n"
    report += f"Bootstrap Result   = {boot_sig}\n\n"
    report += f"Proportion Mediated = {prop_med:.2f}%\n"

    pdf_download_button("Mediation Analysis Report", report, figs=[fig_med],
                        filename="mediation_report.pdf")
    plt.close(fig_med)

else:
    st.warning("Please select a specific analysis from the sidebar.")

# ─────────────────────────────────────────────
# FOOTER

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center; color:#6b7280; font-size:0.9rem; margin-top:20px;'>
        <b>HS-Statistical Assistant</b> | Developed by <b>Haytham Saleh</b><br>
        DBA Candidate | MBA | MSc | MCTS | MCSE | SAS © 2026 All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True
)
