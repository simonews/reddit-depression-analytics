import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
from sklearn.metrics import roc_curve, auc
from pyvis.network import Network

# ===============
# CONFIGURATION
# ==============
st.set_page_config(page_title="Project 10 | Big Data Analytics", layout="wide")
st.markdown(
    """<style>.main {background-color: #0e1117;} h1,h3 {color: white;} .caption {color: #888; font-size:14px;}</style>""",
    unsafe_allow_html=True)


# ==============
# DATA LOADERS
# ==============
def load_kpi():
    try:
        files = glob.glob("data/dashboard_kpi/*.csv")
        if files: return pd.read_csv(files[0])
        return None
    except:
        return None


def load_metrics():
    try:
        files = glob.glob("data/dashboard_metrics/*.csv")
        if files: return pd.read_csv(files[0])
        return None
    except:
        return None


def load_confusion_matrix():
    try:
        files = glob.glob("data/dashboard_confusion_matrix/*.csv")
        if files: return pd.read_csv(files[0])
        return None
    except:
        return None


def load_roc_data():
    try:
        files = glob.glob("data/dashboard_roc_data/*.csv")
        if files: return pd.read_csv(files[0])
        return None
    except:
        return None


def load_time_data(kind="dep"):
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


def load_semantic_graph():
    try:
        files = glob.glob("data/dashboard_semantic_graph/*.csv")
        if files: return pd.read_csv(files[0])
        return None
    except:
        return None


def load_semantic_words():
    try:
        files = glob.glob("data/dashboard_semantic_words/*.csv")
        if files: return pd.read_csv(files[0])
        return None
    except:
        return None


def load_entropy_data():
    try:
        return pd.read_parquet("data/dashboard_entropy.parquet")
    except:
        return None


# =========================
# GRAPH FUNCTIONS
# =========================
def plot_heatmap(df, title, color_scale):
    if df is None: return None
    days = {1: 'Dom', 2: 'Lun', 3: 'Mar', 4: 'Mer', 5: 'Gio', 6: 'Ven', 7: 'Sab'}
    df['day_name'] = df['day'].map(days)
    fig = go.Figure(data=go.Heatmap(
        z=df['count'], x=df['hour'], y=df['day_name'], colorscale=color_scale
    ))
    fig.update_layout(title=title, template="plotly_dark", height=350, margin=dict(l=0, r=0, t=40, b=0))
    return fig


# ==========
# UI MAIN
# ==========
def main():
    st.sidebar.title("Control Panel")
    st.sidebar.info("Dashboard connected to Spark batch results.")

    st.title("Reddit Depression Analytics")
    st.markdown("**Semantic and behavioral analysis at Big Data scale**")
    st.markdown("---")

    # ============
    # DATA LOAD
    # ============
    kpi_df = load_kpi()
    metrics_df = load_metrics()
    cm_df = load_confusion_matrix()
    roc_df = load_roc_data()
    sem_graph_df = load_semantic_graph()
    sem_words_df = load_semantic_words()
    scatter_df = load_scatter_data()
    entropy_df = load_entropy_data()

    if kpi_df is None:
        st.error("MISSING DATA. Run the pipeline..")
        return

    # ========================
    # KPI ROW
    # ========================
    total = kpi_df['total_posts'][0]
    risk = kpi_df['risk_ratio'][0]
    avg_len = kpi_df['avg_length'][0]

    st.caption("**Global KPIs:** Aggregated metrics calculated across the entire distributed dataset.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Posts Analyzed", f"{total:,.0f}")
    c2.metric("Risk Index", f"{risk:.1%}", delta_color="inverse")
    c3.metric("Medium Length", f"{avg_len:.0f} words")

    st.markdown("---")

    # =============================================
    # 1. MODEL VALIDATION (SPIDER + CONFUSION + ROC)
    # =============================================
    st.subheader("Model Validation & Performance")

    col_radar, col_cm, col_roc = st.columns([1, 1, 1])

    # SPIDER CHART (Radar)
    with col_radar:
        if metrics_df is not None:
            # Prepare data for Radar
            categories = metrics_df['metric_name'].tolist()
            values = metrics_df['value'].tolist()

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Model Perf',
                line_color='#00CC96'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                template="plotly_dark",
                title="Performance Radar",
                height=300,
                margin=dict(t=40, b=20, l=40, r=40)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.warning("Metrics not available.")

    # CONFUSION MATRIX
    with col_cm:
        if cm_df is not None:
            # Pivot table for 2x2 matrix
            cm_pivot = cm_df.pivot_table(index='label', columns='prediction', values='count', fill_value=0)
            # Sort: 0 (Control), 1 (Risk)
            cm_pivot = cm_pivot.sort_index(ascending=True).sort_index(axis=1, ascending=True)

            # Heatmap
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_pivot.values,
                x=["Pred: Healthy", "Pred: Risk"],
                y=["Real: Healthy", "Real: Risk"],
                colorscale='Blues',
                text=cm_pivot.values,
                texttemplate="%{text}",
                textfont={"size": 18}
            ))
            fig_cm.update_layout(
                title="Confusion Matrix",
                template="plotly_dark",
                height=300,
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning("Confusion Matrix data not available.")

    # ROC CURVE
    with col_roc:
        if roc_df is not None:
            try:
                # Calculation AUC and Curve with Scikit-Learn
                y_true = roc_df['label']
                y_score = roc_df['score']

                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)

                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.2f}',
                                             line=dict(color='orange', width=2)))
                fig_roc.add_trace(
                    go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash', color='gray')))

                fig_roc.update_layout(
                    title=f"ROC Curve (AUC={roc_auc:.2f})",
                    xaxis_title="FPR",
                    yaxis_title="TPR",
                    template="plotly_dark",
                    height=300,
                    margin=dict(t=40, b=20, l=20, r=20),
                    legend=dict(y=0.1, x=0.7)
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            except Exception as e:
                st.error(f"Errore ROC: {e}")
        else:
            st.warning("No ROC Data.")

    st.markdown("---")

    # ================================
    # TIME ANALYSIS (Double heatmap)
    # ================================
    st.subheader("Circadian Rhythm Analysis (Insomnia)")
    st.caption(
        "**Scientific Objective:** To compare temporal activity patterns between at-risk subjects and a control group to identify sleep disturbances (e.g., nocturnal peaks) associated with 'Blue Light Insomnia'.")


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

    st.markdown("---")

    # ===================
    # SEMANTIC DISCOVERY
    # ===================
    st.subheader("Semantic Network Discovery (Knowledge Extraction)")
    st.caption(
        "**Insight:** Explore how the Word2Vec model connects clinical concepts. Use the Tabs to switch view between the interactive graph and the ranking list.")

    tab_graph, tab_bar = st.tabs(["Interactive Network", "Top Similar Words"])

    # INTERACTIVE GRAPH
    with tab_graph:
        if sem_graph_df is not None:
            net = Network(height="500px", width="100%", bgcolor="#0E1117", font_color="white")
            sources = sem_graph_df['source'].unique()
            targets = sem_graph_df['target'].unique()

            for s in sources:
                net.add_node(s, label=s, color="#FF4B4B", title="Seed Concept", size=25)
            for t in targets:
                if t not in sources:
                    net.add_node(t, label=t, color="#00CC96", title="Learned Association", size=15)

            for _, row in sem_graph_df.iterrows():
                width = row['similarity'] * 5
                net.add_edge(row['source'], row['target'], value=width, color="#555555")

            net.repulsion(node_distance=100, spring_length=200)
            try:
                path = "temp_graph.html"
                net.save_graph(path)
                with open(path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                components.html(source_code, height=500)
            except Exception as e:
                st.error(f"Error creating graph: {e}")
        else:
            st.info("No semantic graph data available.")

    # BAR CHART
    with tab_bar:
        if sem_words_df is not None:
            sem_words_df = sem_words_df.sort_values(by="similarity", ascending=True)  # Sort
            fig_sem = px.bar(
                sem_words_df, x="similarity", y="word", orientation='h',
                template="plotly_dark",
                title="Top 20 Concepts by Similarity",
                color="similarity", color_continuous_scale="RdPu"
            )
            st.plotly_chart(fig_sem, use_container_width=True)
        else:
            st.info("No semantic list data available.")

    st.markdown("---")

    # ==========================================
    # COGNITIVE ANALYSIS (ENTROPY + VERBOSITY)
    # ==========================================
    st.subheader("Cognitive & Behavioral Analysis")


    st.caption("""
    This section validates two key psychological theories regarding depression:
    1.  **Linguistic Entropy (Left):** Validating *Emotional Inertia*. Lower entropy indicates cognitive rigidity and repetitive language patterns (visible as a "squashed" shape for the Risk group).
    2.  **Verbosity Patterns (Right):** Analyzing the average post length. This tests the hypothesis of *Rumination* (excessive writing) vs *Apathy* (very short interactions).
    """)

    col_violin, col_scatter = st.columns([6, 4])

    # 1. ENTROPY VIOLIN PLOT
    with col_violin:
        if entropy_df is not None:
            # Map labels for better reading
            entropy_df['Condition'] = entropy_df['label'].map({0: 'Healthy (Control)', 1: 'Risk (Depressed)'})

            fig_violin = go.Figure()
            # Violin for Control
            fig_violin.add_trace(go.Violin(
                x=entropy_df['Condition'][entropy_df['label'] == 0],
                y=entropy_df['shannon_entropy'][entropy_df['label'] == 0],
                legendgroup='Healthy', scalegroup='Healthy', name='Healthy',
                line_color='#00CC96', points=False
            ))
            # Violin for Risk
            fig_violin.add_trace(go.Violin(
                x=entropy_df['Condition'][entropy_df['label'] == 1],
                y=entropy_df['shannon_entropy'][entropy_df['label'] == 1],
                legendgroup='Risk', scalegroup='Risk', name='Risk',
                line_color='#FF4B4B', points=False
            ))

            fig_violin.update_traces(box_visible=True, meanline_visible=True)
            fig_violin.update_layout(
                title="Linguistic Entropy Distribution (Cognitive Flexibility)",
                yaxis_title="Shannon Entropy Score",
                template="plotly_dark",
                violinmode='group',
                height=450
            )
            st.plotly_chart(fig_violin, use_container_width=True)
        else:
            st.warning("Entropy data not available.")

    # 2. VERBOSITY SCATTER PLOT
    with col_scatter:
        if scatter_df is not None:
            fig_scat = px.scatter(
                scatter_df, x="avg_len", y="label", color="label",
                size="avg_len", size_max=15,
                template="plotly_dark",
                title="Verbosity vs Classification",
                labels={"avg_len": "Avg Words per Post", "label": "Class"}
            )
            fig_scat.update_layout(height=450)
            st.plotly_chart(fig_scat, use_container_width=True)


if __name__ == "__main__":
    main()