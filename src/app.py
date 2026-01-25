import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

#===============
# CONFIGURATION
# ==============
st.set_page_config(page_title="Project 10 | Big Data Analytics", layout="wide")
st.markdown(
    """<style>.main {background-color: #0e1117;} h1,h3 {color: white;} .caption {color: #888; font-size:14px;}</style>""",
    unsafe_allow_html=True)


#==============
# DATA LOADERS
#==============
def load_kpi():
    try:
        files = glob.glob("data/dashboard_kpi/*.csv")
        if files: return pd.read_csv(files[0])
        return None
    except:
        return None


def load_metrics():
    # Load ML model metrics (F1 Score)
    try:
        files = glob.glob("data/dashboard_metrics/*.csv")
        if files: return pd.read_csv(files[0])
        return None
    except:
        return None


def load_time_data(kind="dep"):
    # Load Heatmap. kind='dep' (Depressed) or 'ctrl' (Healthy)
    try:
        filename = f"data/dashboard_time_{kind}.parquet"
        return pd.read_parquet(filename)
    except:
        return None


def load_scatter_data():
    try:
        return pd.read_parquet("data/dashboard_scatter.parquet")
    except:
        return None


def load_semantic_data():
    try:
        files = glob.glob("data/dashboard_semantic_words/*.csv")
        if files: return pd.read_csv(files[0])
        return None
    except:
        return None


#=========================
# HEATMAP GRAPH FUNCTION
#=========================
def plot_heatmap(df, title, color_scale):
    if df is None: return None
    days = {1: 'Dom', 2: 'Lun', 3: 'Mar', 4: 'Mer', 5: 'Gio', 6: 'Ven', 7: 'Sab'}
    df['day_name'] = df['day'].map(days)
    fig = go.Figure(data=go.Heatmap(
        z=df['count'], x=df['hour'], y=df['day_name'], colorscale=color_scale
    ))
    fig.update_layout(title=title, template="plotly_dark", height=350, margin=dict(l=0, r=0, t=40, b=0))
    return fig


#==========
# UI MAIN
#==========
def main():
    st.sidebar.title("Control Panel")
    st.sidebar.info("Dashboard connected to Spark batch results.")

    st.title("Reddit Depression Analytics")
    st.markdown("**Semantic and behavioral analysis at Big Data scale**")
    st.markdown("---")

    #============
    # DATA LOAD
    #============
    kpi_df = load_kpi()
    metrics_df = load_metrics()
    scatter_df = load_scatter_data()

    if kpi_df is None:
        st.error("MISSING DATA. Run the pipeline..")
        return

    #========================
    # KPI ROW & MODEL METRICS
    #========================
    total = kpi_df['total_posts'][0]
    risk = kpi_df['risk_ratio'][0]
    avg_len = kpi_df['avg_length'][0]

    #=============
    # F1 RECOVERY
    #=============
    f1_score = 0.0
    if metrics_df is not None and not metrics_df.empty:
        f1_score = metrics_df['value'][0]

    st.caption("**Global KPIs:** Aggregated metrics calculated across the entire distributed dataset.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posts Analyzed", f"{total:,.0f}")
    c2.metric("Risk Index", f"{risk:.1%}", delta_color="inverse")
    c3.metric("Medium Length", f"{avg_len:.0f} parole")
    c4.metric("Model F1-Score", f"{f1_score:.2%}", help="Random Forest Model Performance on Test Set")

    st.markdown("---")

    #================================
    # TIME ANALYSIS (Double heatmap)
    #================================
    st.subheader("Circadian Rhythm Analysis (Insomnia)")
    st.caption(
        "**Scientific Objective:** To compare temporal activity patterns between at-risk subjects and a control group to identify sleep disturbances (e.g., nocturnal peaks).")

    col_heat, col_scatter = st.columns([6, 4])

    with col_heat:
        # Load the two dataset
        df_dep = load_time_data("dep")
        df_ctrl = load_time_data("ctrl")

        tab1, tab2 = st.tabs(["🔴 Subjects at Risk", "🟢 Control Group"])

        with tab1:
            if df_dep is not None:
                st.plotly_chart(plot_heatmap(df_dep, "Depressed Users Activity", "Magma"), use_container_width=True)
            else:
                st.warning("Depressed data not available")

        with tab2:
            if df_ctrl is not None:
                st.plotly_chart(plot_heatmap(df_ctrl, "User Activity Control", "Viridis"), use_container_width=True)
            else:
                st.warning("Control data not available")

    with col_scatter:
        st.write("")  # Spacer
        st.write("")
        if scatter_df is not None:
            fig_scat = px.scatter(
                scatter_df, x="avg_len", y="label", color="label", size="avg_len",
                template="plotly_dark", title="Pattern Linguistici (Verbosità)",
                labels={"avg_len": "Parole medie", "label": "Class"}
            )
            st.plotly_chart(fig_scat, use_container_width=True)
            st.caption(
                "**Analysis:** Correlation between average post length (Verbosity/Rumination) and classification.")

    #==================
    # SEMANTIC SESSION
    #==================
    st.markdown("---")
    st.subheader("Semantic Extraction (Word Embeddings)")
    st.caption(
        "**Logic:** Query the Word2Vec model to extract semantically similar concepts to 'Depression' and 'Anxiety' (Cosine similarity > 0.5).")

    sem_df = load_semantic_data()
    if sem_df is not None:
        sem_df = sem_df.sort_values(by="similarity", ascending=True).tail(20)
        fig_sem = px.bar(
            sem_df, x="similarity", y="word", orientation='h',
            template="plotly_dark",
            title="Concetti Latenti Estratti dal Modello",
            color="similarity", color_continuous_scale="RdPu"
        )
        st.plotly_chart(fig_sem, use_container_width=True)
    else:
        st.info("No semantic data found.")


if __name__ == "__main__":
    main()