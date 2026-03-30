import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Palo Alto Networks — Attrition Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: linear-gradient(180deg,#0c2340 0%,#1a3a5c 100%); }
  [data-testid="stSidebar"] * { color: #e8f0fe !important; }

  /* Cards */
  .kpi-card {
    background: linear-gradient(135deg,#0c2340,#1a3a5c);
    border-radius: 14px; padding: 22px 18px;
    color: white; text-align: center;
    border: 1px solid rgba(255,255,255,.12);
    box-shadow: 0 4px 20px rgba(0,0,0,.25);
  }
  .kpi-card .val { font-size: 2.4rem; font-weight: 700; color: #4fc3f7; }
  .kpi-card .lbl { font-size: .8rem; color: #aac6e0; letter-spacing:.5px; margin-top:4px; }

  .risk-high   { background:#ff4d4d22; border-left:4px solid #ff4d4d; padding:10px 14px; border-radius:8px; }
  .risk-medium { background:#ffa50022; border-left:4px solid #ffa500; padding:10px 14px; border-radius:8px; }
  .risk-low    { background:#00c85322; border-left:4px solid #00c853; padding:10px 14px; border-radius:8px; }

  .section-title {
    font-size:1.3rem; font-weight:600;
    color:#0c2340; margin:18px 0 10px;
    border-bottom:2px solid #4fc3f7; padding-bottom:6px;
  }
  .stButton>button {
    background:linear-gradient(90deg,#0c2340,#1565c0);
    color:white; border:none; border-radius:8px;
    padding:8px 22px; font-weight:600;
  }
  .stButton>button:hover { opacity:.88; }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading & Preprocessing ─────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("Palo_Alto_Networks.csv")

    # --- feature engineering ---
    df["IncomePerExp"]     = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
    df["PromotionDelay"]   = (df["YearsSinceLastPromotion"] > 3).astype(int)
    df["EngagementScore"]  = (df["JobSatisfaction"] + df["JobInvolvement"] +
                              df["EnvironmentSatisfaction"] + df["RelationshipSatisfaction"]) / 4
    df["WorkloadStress"]   = ((df["OverTime"] == "Yes") &
                              (df["WorkLifeBalance"] <= 2)).astype(int)

    # encode
    cat_cols = ["BusinessTravel","Department","EducationField",
                "Gender","JobRole","MaritalStatus","OverTime"]
    le_map = {}
    df_enc = df.copy()
    for c in cat_cols:
        le = LabelEncoder()
        df_enc[c] = le.fit_transform(df_enc[c])
        le_map[c] = le

    target = "Attrition"
    drop_cols = [target]
    X = df_enc.drop(columns=drop_cols)
    y = df_enc[target]

    # scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return df, X_scaled, y, scaler, le_map, X.columns.tolist()

@st.cache_resource
def train_models(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=200, use_label_encoder=False,
                                             eval_metric="logloss", random_state=42),
    }
    results = {}
    for name, mdl in models.items():
        mdl.fit(X_res, y_res)
        y_pred  = mdl.predict(X_test)
        y_prob  = mdl.predict_proba(X_test)[:, 1]
        results[name] = {
            "model":     mdl,
            "y_test":    y_test,
            "y_pred":    y_pred,
            "y_prob":    y_prob,
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_prob),
        }
    return results, X_test, y_test

def assign_risk(prob):
    if prob >= 0.60: return "High Risk",   "🔴"
    if prob >= 0.30: return "Medium Risk", "🟡"
    return "Low Risk", "🟢"

# ─── Load Data ─────────────────────────────────────────────────────────────────
df, X_scaled, y, scaler, le_map, feature_names = load_and_preprocess()

with st.spinner("Training ML models — please wait…"):
    model_results, X_test, y_test = train_models(X_scaled, y)

best_model_name = max(model_results, key=lambda k: model_results[k]["roc_auc"])
best_model      = model_results[best_model_name]["model"]

# Full-dataset predictions for the risk dashboard
all_probs = best_model.predict_proba(X_scaled)[:, 1]
df["AttritionProbability"] = all_probs
df["RiskCategory"] = df["AttritionProbability"].apply(lambda p: assign_risk(p)[0])
df["RiskIcon"]     = df["AttritionProbability"].apply(lambda p: assign_risk(p)[1])

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ PAN Attrition Intel")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Executive Dashboard",
        "🧑‍💼 Employee Risk Profiles",
        "🏢 Department Analytics",
        "🤖 Model Performance",
        "🔬 Feature Explainability",
        "🎛️ What-If Simulator",
    ])
    st.markdown("---")
    st.markdown("**Active Model**")
    st.info(f"✅ {best_model_name}")
    st.markdown("**Dataset**")
    st.caption(f"{len(df):,} employees · {df['Attrition'].sum()} churned")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Dashboard":
    st.markdown("# 📊 Attrition Risk Dashboard")
    st.markdown("*Predictive workforce intelligence for Palo Alto Networks HR leadership*")

    # KPI row
    total      = len(df)
    high_risk  = (df["RiskCategory"] == "High Risk").sum()
    med_risk   = (df["RiskCategory"] == "Medium Risk").sum()
    low_risk   = (df["RiskCategory"] == "Low Risk").sum()
    actual_att = df["Attrition"].sum()
    avg_prob   = df["AttritionProbability"].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, lbl in [
        (c1, total,       "Total Employees"),
        (c2, f"{high_risk} 🔴", "High Risk"),
        (c3, f"{med_risk} 🟡",  "Medium Risk"),
        (c4, f"{low_risk} 🟢",  "Low Risk"),
        (c5, f"{avg_prob:.1%}", "Avg Attrition Prob"),
    ]:
        col.markdown(f"""<div class="kpi-card">
            <div class="val">{val}</div><div class="lbl">{lbl}</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    col_a, col_b = st.columns([1,1])
    with col_a:
        st.markdown('<p class="section-title">Risk Distribution</p>', unsafe_allow_html=True)
        risk_counts = df["RiskCategory"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        color_map = {"High Risk":"#ff4d4d","Medium Risk":"#ffa500","Low Risk":"#00c853"}
        fig = px.pie(risk_counts, names="Risk", values="Count",
                     color="Risk", color_discrete_map=color_map, hole=0.5)
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-title">Attrition Probability Distribution</p>', unsafe_allow_html=True)
        fig = px.histogram(df, x="AttritionProbability", nbins=40,
                           color_discrete_sequence=["#1565c0"],
                           labels={"AttritionProbability":"Attrition Probability"})
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300, bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    # Charts row 2
    col_c, col_d = st.columns([1,1])
    with col_c:
        st.markdown('<p class="section-title">Risk by Department</p>', unsafe_allow_html=True)
        dept_risk = df.groupby(["Department","RiskCategory"]).size().reset_index(name="Count")
        fig = px.bar(dept_risk, x="Department", y="Count", color="RiskCategory",
                     color_discrete_map=color_map, barmode="stack")
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300, legend_title="Risk")
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown('<p class="section-title">Monthly Income vs Attrition Probability</p>', unsafe_allow_html=True)
        fig = px.scatter(df, x="MonthlyIncome", y="AttritionProbability",
                         color="RiskCategory", color_discrete_map=color_map,
                         opacity=0.65, size_max=8)
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300)
        st.plotly_chart(fig, use_container_width=True)

    # High-risk employee table
    st.markdown('<p class="section-title">⚠️ Top 20 High-Risk Employees</p>', unsafe_allow_html=True)
    top20 = df[df["RiskCategory"]=="High Risk"].sort_values(
        "AttritionProbability", ascending=False).head(20)[
        ["Department","JobRole","Age","MonthlyIncome","YearsAtCompany",
         "OverTime","JobSatisfaction","AttritionProbability","RiskCategory","RiskIcon"]
    ].copy()
    top20["AttritionProbability"] = top20["AttritionProbability"].map("{:.1%}".format)
    st.dataframe(top20.reset_index(drop=True), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EMPLOYEE RISK PROFILES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧑‍💼 Employee Risk Profiles":
    st.markdown("# 🧑‍💼 Employee Risk Profiles")

    col_f, col_g = st.columns([1,2])
    with col_f:
        dept_sel  = st.selectbox("Filter by Department", ["All"] + sorted(df["Department"].unique()))
        role_sel  = st.selectbox("Filter by Job Role",   ["All"] + sorted(df["JobRole"].unique()))
        risk_sel  = st.multiselect("Risk Category", ["High Risk","Medium Risk","Low Risk"],
                                   default=["High Risk","Medium Risk","Low Risk"])
        thresh    = st.slider("Min Attrition Probability", 0.0, 1.0, 0.0, 0.05)

    filt = df.copy()
    if dept_sel != "All": filt = filt[filt["Department"] == dept_sel]
    if role_sel != "All": filt = filt[filt["JobRole"]    == role_sel]
    filt = filt[filt["RiskCategory"].isin(risk_sel)]
    filt = filt[filt["AttritionProbability"] >= thresh]

    with col_g:
        st.metric("Matching Employees", len(filt))
        st.metric("High Risk in Selection",
                  (filt["RiskCategory"]=="High Risk").sum())

    # Profile table
    show_cols = ["Department","JobRole","Age","Gender","MaritalStatus",
                 "MonthlyIncome","OverTime","JobSatisfaction","WorkLifeBalance",
                 "YearsAtCompany","YearsSinceLastPromotion",
                 "AttritionProbability","RiskCategory","RiskIcon"]
    display = filt[show_cols].copy()
    display["AttritionProbability"] = display["AttritionProbability"].map("{:.1%}".format)
    st.dataframe(display.sort_values("RiskCategory").reset_index(drop=True),
                 use_container_width=True, height=420)

    # Deep-dive single employee
    st.markdown("---")
    st.markdown("### 🔍 Individual Employee Deep-Dive")
    emp_idx = st.number_input("Employee Row Index (0-based)", 0, len(df)-1, 0)
    row     = df.iloc[emp_idx]
    prob    = row["AttritionProbability"]
    risk, icon = assign_risk(prob)

    css_cls = {"High Risk":"risk-high","Medium Risk":"risk-medium","Low Risk":"risk-low"}[risk]
    st.markdown(f"""<div class="{css_cls}">
        <b>{icon} {risk}</b> &nbsp;|&nbsp; Attrition Probability: <b>{prob:.1%}</b>
    </div>""", unsafe_allow_html=True)

    r1,r2,r3,r4 = st.columns(4)
    r1.metric("Age",         int(row["Age"]))
    r2.metric("Department",  row["Department"])
    r3.metric("Monthly Income", f"${row['MonthlyIncome']:,}")
    r4.metric("Years at Company", int(row["YearsAtCompany"]))

    r5,r6,r7,r8 = st.columns(4)
    r5.metric("Job Satisfaction",  row["JobSatisfaction"])
    r6.metric("Work-Life Balance", row["WorkLifeBalance"])
    r7.metric("OverTime",          row["OverTime"])
    r8.metric("Last Promotion (yrs)", int(row["YearsSinceLastPromotion"]))

    # Radar chart
    radar_features = ["JobSatisfaction","EnvironmentSatisfaction",
                      "RelationshipSatisfaction","JobInvolvement","WorkLifeBalance"]
    emp_vals  = [row[f] for f in radar_features]
    avg_vals  = [df[f].mean() for f in radar_features]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=emp_vals+[emp_vals[0]], theta=radar_features+[radar_features[0]],
                                  fill="toself", name="Employee", line_color="#4fc3f7"))
    fig.add_trace(go.Scatterpolar(r=avg_vals+[avg_vals[0]], theta=radar_features+[radar_features[0]],
                                  fill="toself", name="Avg Employee",
                                  line_color="#ffa500", opacity=0.5))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,4.5])),
                      title="Engagement Radar vs Company Average", height=380)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DEPARTMENT ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏢 Department Analytics":
    st.markdown("# 🏢 Department-Level Risk Analytics")

    # Dept summary
    dept_sum = df.groupby("Department").agg(
        Total=("Attrition","count"),
        ActualAttrition=("Attrition","sum"),
        AvgRiskScore=("AttritionProbability","mean"),
        HighRisk=("RiskCategory", lambda x: (x=="High Risk").sum()),
    ).reset_index()
    dept_sum["ActualRate"] = dept_sum["ActualAttrition"]/dept_sum["Total"]
    dept_sum["HighRiskPct"] = dept_sum["HighRisk"]/dept_sum["Total"]

    st.dataframe(dept_sum.style.format({
        "AvgRiskScore":"{:.1%}","ActualRate":"{:.1%}","HighRiskPct":"{:.1%}"
    }), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(dept_sum, x="Department", y="AvgRiskScore",
                     color="AvgRiskScore", color_continuous_scale="Reds",
                     title="Average Attrition Risk Score by Department",
                     labels={"AvgRiskScore":"Avg Risk Score"})
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        role_risk = df.groupby("JobRole")["AttritionProbability"].mean().sort_values(ascending=False)
        fig = px.bar(role_risk.reset_index(), x="AttritionProbability", y="JobRole",
                     orientation="h", color="AttritionProbability",
                     color_continuous_scale="OrRd",
                     title="Average Risk Score by Job Role",
                     labels={"AttritionProbability":"Avg Risk"})
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Overtime vs attrition
    col3, col4 = st.columns(2)
    with col3:
        ot = df.groupby(["Department","OverTime"])["AttritionProbability"].mean().reset_index()
        fig = px.bar(ot, x="Department", y="AttritionProbability", color="OverTime",
                     barmode="group", title="Overtime Impact on Risk by Department",
                     color_discrete_map={"Yes":"#ff4d4d","No":"#4fc3f7"})
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        heatmap_data = df.pivot_table(values="AttritionProbability",
                                      index="JobSatisfaction", columns="WorkLifeBalance",
                                      aggfunc="mean")
        fig = px.imshow(heatmap_data, color_continuous_scale="RdYlGn_r",
                        title="Risk Heatmap: Job Satisfaction × Work-Life Balance",
                        labels=dict(x="Work-Life Balance",y="Job Satisfaction",
                                    color="Avg Risk"))
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("# 🤖 Model Performance Comparison")

    # Metrics table
    metrics_df = pd.DataFrame([{
        "Model":     name,
        "Accuracy":  r["accuracy"],
        "Precision": r["precision"],
        "Recall":    r["recall"],
        "F1-Score":  r["f1"],
        "ROC-AUC":   r["roc_auc"],
    } for name, r in model_results.items()])

    best_flag = metrics_df["ROC-AUC"] == metrics_df["ROC-AUC"].max()
    metrics_df["Best"] = best_flag.map({True:"⭐","False":""})

    st.dataframe(metrics_df.style.format({
        "Accuracy":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}",
        "F1-Score":"{:.3f}","ROC-AUC":"{:.3f}",
    }).highlight_max(subset=["Accuracy","Precision","Recall","F1-Score","ROC-AUC"],
                     color="#d4f1d4"), use_container_width=True)

    # Radar comparison
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        cats = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
        colors = ["#4fc3f7","#ffa500","#ff4d4d","#ab47bc"]
        for (name, r), clr in zip(model_results.items(), colors):
            vals = [r["accuracy"],r["precision"],r["recall"],r["f1"],r["roc_auc"]]
            fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]],
                                          name=name, line_color=clr, fill="toself",
                                          opacity=0.65))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])),
                          title="Model Comparison Radar", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        colors = ["#4fc3f7","#ffa500","#ff4d4d","#ab47bc"]
        for (name, r), clr in zip(model_results.items(), colors):
            fpr, tpr, _ = roc_curve(r["y_test"], r["y_prob"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{name} (AUC={r['roc_auc']:.3f})",
                                     line=dict(color=clr, width=2)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                                 line=dict(dash="dash",color="gray"),name="Random"))
        fig.update_layout(title="ROC Curves", xaxis_title="FPR",
                          yaxis_title="TPR", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix for best model
    st.markdown(f"### Confusion Matrix — {best_model_name}")
    r = model_results[best_model_name]
    cm = confusion_matrix(r["y_test"], r["y_pred"])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted",y="Actual"),
                    x=["No Attrition","Attrition"], y=["No Attrition","Attrition"])
    fig.update_layout(height=350, width=450)
    st.plotly_chart(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — FEATURE EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Feature Explainability":
    st.markdown("# 🔬 Feature Explainability")

    # Feature importance from Random Forest (always available)
    rf_model = model_results["Random Forest"]["model"]
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    top20_imp = importances.nlargest(20).sort_values()

    col1, col2 = st.columns([3,2])
    with col1:
        fig = px.bar(top20_imp.reset_index(), x=0, y="index",
                     orientation="h", color=0,
                     color_continuous_scale="Blues",
                     title="Top 20 Feature Importances (Random Forest)",
                     labels={0:"Importance","index":"Feature"})
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📌 Key Drivers")
        for feat, imp in importances.nlargest(10).items():
            bar_w = int(imp / importances.max() * 100)
            st.markdown(f"""
            **{feat}**
            <div style="background:#e0e0e0;border-radius:6px;height:10px;width:100%">
              <div style="background:#1565c0;width:{bar_w}%;height:10px;border-radius:6px"></div>
            </div>
            <small style="color:#666">{imp:.4f}</small><br>
            """, unsafe_allow_html=True)

    # SHAP
    st.markdown("---")
    st.markdown("### 🧠 SHAP Value Analysis (XGBoost)")
    with st.spinner("Computing SHAP values…"):
        xgb_model  = model_results["XGBoost"]["model"]
        sample_idx = np.random.choice(len(X_scaled), size=min(200, len(X_scaled)), replace=False)
        X_sample   = X_scaled.iloc[sample_idx]

        explainer   = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)

        shap_df = pd.DataFrame(np.abs(shap_values), columns=feature_names)
        mean_shap = shap_df.mean().nlargest(15).sort_values()

    fig = px.bar(mean_shap.reset_index(), x=0, y="index", orientation="h",
                 color=0, color_continuous_scale="Oranges",
                 title="Mean |SHAP| — Top 15 Features (XGBoost)",
                 labels={0:"Mean |SHAP|","index":"Feature"})
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

    # Feature interaction
    st.markdown("### 🔁 Overtime × Satisfaction Interaction")
    fig = px.box(df, x="OverTime", y="AttritionProbability", color="OverTime",
                 points="all", color_discrete_map={"Yes":"#ff4d4d","No":"#4fc3f7"},
                 title="Attrition Probability: Overtime vs Non-Overtime")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎛️ What-If Simulator":
    st.markdown("# 🎛️ What-If Scenario Simulator")
    st.markdown("Adjust employee attributes to explore how interventions affect attrition risk.")

    col_inp, col_res = st.columns([1,1])

    with col_inp:
        st.markdown("### 📝 Employee Inputs")
        age         = st.slider("Age", 18, 65, 35)
        monthly_inc = st.slider("Monthly Income ($)", 1000, 20000, 6000, 100)
        job_sat     = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        env_sat     = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        wlb         = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
        job_inv     = st.slider("Job Involvement (1-4)", 1, 4, 3)
        rel_sat     = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
        overtime    = st.selectbox("OverTime", ["No","Yes"])
        yrs_company = st.slider("Years at Company", 0, 40, 5)
        yrs_promo   = st.slider("Years Since Last Promotion", 0, 15, 2)
        dist_home   = st.slider("Distance From Home (km)", 1, 30, 5)
        num_comp    = st.slider("Companies Worked Previously", 0, 10, 2)
        tot_exp     = st.slider("Total Working Years", 0, 40, 10)
        train_times = st.slider("Training Times Last Year", 0, 6, 2)
        stock_opt   = st.slider("Stock Option Level (0-3)", 0, 3, 1)
        pct_hike    = st.slider("% Salary Hike", 11, 25, 15)
        job_level   = st.slider("Job Level (1-5)", 1, 5, 2)

    # Build a synthetic employee row using median values as defaults
    median_row = df.median(numeric_only=True).to_dict()

    employee = {col: median_row.get(col, 0) for col in feature_names}
    # Override with user inputs
    override = {
        "Age": age,
        "MonthlyIncome": monthly_inc,
        "JobSatisfaction": job_sat,
        "EnvironmentSatisfaction": env_sat,
        "WorkLifeBalance": wlb,
        "JobInvolvement": job_inv,
        "RelationshipSatisfaction": rel_sat,
        "OverTime": 1 if overtime=="Yes" else 0,
        "YearsAtCompany": yrs_company,
        "YearsSinceLastPromotion": yrs_promo,
        "DistanceFromHome": dist_home,
        "NumCompaniesWorked": num_comp,
        "TotalWorkingYears": tot_exp,
        "TrainingTimesLastYear": train_times,
        "StockOptionLevel": stock_opt,
        "PercentSalaryHike": pct_hike,
        "JobLevel": job_level,
        # computed features
        "IncomePerExp":    monthly_inc / (tot_exp + 1),
        "PromotionDelay":  int(yrs_promo > 3),
        "EngagementScore": (job_sat+job_inv+env_sat+rel_sat)/4,
        "WorkloadStress":  int(overtime=="Yes" and wlb<=2),
    }
    employee.update(override)

    emp_df      = pd.DataFrame([employee])[feature_names]
    emp_scaled  = pd.DataFrame(scaler.transform(emp_df), columns=feature_names)
    sim_prob    = best_model.predict_proba(emp_scaled)[0, 1]
    sim_risk, sim_icon = assign_risk(sim_prob)
    css_cls = {"High Risk":"risk-high","Medium Risk":"risk-medium","Low Risk":"risk-low"}[sim_risk]

    with col_res:
        st.markdown("### 📊 Predicted Risk")
        st.markdown(f"""<div class="{css_cls}" style="font-size:1.1rem;margin-top:12px">
            {sim_icon} <b>{sim_risk}</b> — Attrition Probability: <b>{sim_prob:.1%}</b>
        </div>""", unsafe_allow_html=True)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=sim_prob*100,
            title={"text": "Attrition Risk (%)"},
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color="#1565c0"),
                steps=[
                    dict(range=[0, 30],  color="#00c853"),
                    dict(range=[30, 60], color="#ffa500"),
                    dict(range=[60, 100],color="#ff4d4d"),
                ],
                threshold=dict(line=dict(color="black",width=3),
                               thickness=0.75, value=sim_prob*100),
            ),
            number=dict(suffix="%", font=dict(size=32)),
        ))
        fig.update_layout(height=280, margin=dict(t=40,b=10,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.markdown("### 💡 Retention Recommendations")
        recs = []
        if job_sat <= 2:   recs.append("⬆️ Improve job satisfaction through role enrichment or project variety")
        if overtime=="Yes": recs.append("⏱️ Reduce overtime workload to improve work-life balance")
        if wlb <= 2:       recs.append("🧘 Offer flexible working arrangements")
        if yrs_promo > 3:  recs.append("🏆 Consider a promotion or title change")
        if monthly_inc < 4000: recs.append("💰 Review compensation against market benchmarks")
        if stock_opt == 0: recs.append("📈 Offer stock options to improve retention incentives")
        if env_sat <= 2:   recs.append("🏢 Address workplace environment concerns")
        if not recs:       recs.append("✅ Employee profile looks healthy — maintain current engagement")
        for rec in recs:
            st.markdown(f"- {rec}")

    # Sensitivity analysis
    st.markdown("---")
    st.markdown("### 📈 Sensitivity: How Each Factor Changes Risk")
    sensitivity = {}
    for feat in ["JobSatisfaction","MonthlyIncome","WorkLifeBalance",
                 "EnvironmentSatisfaction","YearsSinceLastPromotion",
                 "StockOptionLevel","TrainingTimesLastYear"]:
        if feat in feature_names:
            vals_range = np.linspace(
                df[feat].min(), df[feat].max(), 10
            )
            probs = []
            for v in vals_range:
                tmp = emp_df.copy()
                tmp[feat] = v
                tmp_s = pd.DataFrame(scaler.transform(tmp), columns=feature_names)
                probs.append(best_model.predict_proba(tmp_s)[0,1])
            sensitivity[feat] = {"range": vals_range.tolist(), "probs": probs}

    fig = go.Figure()
    for feat, data in sensitivity.items():
        fig.add_trace(go.Scatter(x=data["range"], y=data["probs"],
                                 mode="lines+markers", name=feat))
    fig.update_layout(title="Sensitivity Analysis: Feature Value vs Attrition Risk",
                      xaxis_title="Feature Value", yaxis_title="Attrition Probability",
                      height=420)
    st.plotly_chart(fig, use_container_width=True)


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;font-size:.8rem;padding:8px">
  🛡️ Palo Alto Networks | HR Attrition Intelligence Platform | Built with Streamlit + XGBoost + SHAP
</div>""", unsafe_allow_html=True)