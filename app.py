import streamlit as st
import pandas as pd
import plotly.express as px

import numpy as np

# ============================================================
# AI INFRASTRUCTURE CAPACITY CALCULATIONS
# ============================================================

def calculate_ai_capacity(
    ai_data_tb,
    model_parameters_b=70,
    vram_per_gpu_gb=80,
    precision_bytes=2,
    transfer_window_minutes=120,
    iops_block_kb=64
):

    # --------------------------------------------------------
    # 1. STORAGE
    # --------------------------------------------------------

    # AI pipeline typically creates additional copies:
    #
    # Raw dataset
    # + processed dataset
    # + embeddings / indexes
    #
    # We use 1.5x as a planning assumption.

    ai_storage_tb = ai_data_tb * 1.5


    # --------------------------------------------------------
    # 2. STORAGE IOPS
    # --------------------------------------------------------

        # 2. STORAGE THROUGHPUT AND IOPS

    total_bytes = ai_data_tb * 1_000_000_000_000
    block_bytes = iops_block_kb * 1024
    seconds = transfer_window_minutes * 60

    # Average throughput needed to scan the entire dataset
    required_throughput_bytes_sec = (
        total_bytes / seconds
    )

    # Approximate IOPS for sequential full-dataset scanning
    required_iops = (
        required_throughput_bytes_sec / block_bytes
    )

    # Convert bytes per second to decimal Gbps
    required_throughput_gbps = (
        required_throughput_bytes_sec
        * 8 / 1_000_000_000
    )

    # --------------------------------------------------------
    # 3. NETWORK BANDWIDTH
    # --------------------------------------------------------

    network_gbps = (
        ai_data_tb
        * 8_000
        / transfer_window_minutes
        / 60
    )


    # --------------------------------------------------------
    # 4. GPU VRAM
    # --------------------------------------------------------

    model_memory_gb = (
        model_parameters_b
        * 1_000_000_000
        * precision_bytes
        / (1024 ** 3)
    )


    # Training generally requires additional memory
    # for gradients, optimizer states and activations.

    training_memory_gb = (
        model_memory_gb * 6
    )


    gpu_count = max(
        1,
        int(
            np.ceil(
                training_memory_gb /
                vram_per_gpu_gb
            )
        )
    )


    # --------------------------------------------------------
    # RETURN CAPACITY MODEL
    # --------------------------------------------------------

    return {
        "AI_Data_TB": ai_data_tb,

        "AI_Storage_TB": ai_storage_tb,

        "Required_IOPS": required_iops,
        "Storage_Throughput_Gbps": required_throughput_gbps,

        "Network_Gbps": network_gbps,

        "Model_Memory_GB": model_memory_gb,

        "Training_Memory_GB": training_memory_gb,

        "GPU_Count": gpu_count
    }

# ============================================================
# AI INFRASTRUCTURE RISK ANALYSIS
# ============================================================


def analyze_infrastructure_risk(capacity):

    risks = []

    # GPU risk
    if capacity["GPU_Count"] >= 16:
        risks.append({
            "Area": "GPU",
            "Risk": "High",
            "Reason": "Large GPU footprint required for model training"
        })
    elif capacity["GPU_Count"] >= 8:
        risks.append({
            "Area": "GPU",
            "Risk": "Medium",
            "Reason": "Significant GPU capacity required"
        })
    else:
        risks.append({
            "Area": "GPU",
            "Risk": "Low",
            "Reason": "GPU requirement is relatively modest"
        })

    # Network risk
    if capacity["Network_Gbps"] >= 400:
        risks.append({
            "Area": "Network",
            "Risk": "High",
            "Reason": "High-speed network fabric required"
        })
    elif capacity["Network_Gbps"] >= 200:
        risks.append({
            "Area": "Network",
            "Risk": "Medium",
            "Reason": "High-throughput network connectivity required"
        })
    else:
        risks.append({
            "Area": "Network",
            "Risk": "Low",
            "Reason": "Network requirement is manageable"
        })

    # Storage IOPS risk
    if capacity["Required_IOPS"] >= 500000:
        risks.append({
            "Area": "Storage IOPS",
            "Risk": "High",
            "Reason": "Very high storage I/O demand; validate against target storage capability"
        })
    elif capacity["Required_IOPS"] >= 100000:
        risks.append({
            "Area": "Storage IOPS",
            "Risk": "Medium",
            "Reason": "Elevated storage I/O demand; verify throughput and latency targets"
        })
    else:
        risks.append({
            "Area": "Storage IOPS",
            "Risk": "Low",
            "Reason": "Lower modeled storage I/O demand; validate against workload requirements"
        })

    # Storage capacity risk
    if capacity["AI_Storage_TB"] >= 1000:
        risks.append({
            "Area": "Storage Capacity",
            "Risk": "High",
            "Reason": "Large AI dataset footprint"
        })
    elif capacity["AI_Storage_TB"] >= 500:
        risks.append({
            "Area": "Storage Capacity",
            "Risk": "Medium",
            "Reason": "Significant AI storage footprint"
        })
    else:
        risks.append({
            "Area": "Storage Capacity",
            "Risk": "Low",
            "Reason": "Storage footprint is relatively modest"
        })

    return pd.DataFrame(risks)
def calculate_overall_risk(risk_df):

    high_count = (
        risk_df["Risk"] == "High"
    ).sum()

    medium_count = (
        risk_df["Risk"] == "Medium"
    ).sum()

    if high_count >= 2:
        return "High"

    elif high_count == 1 or medium_count >= 2:
        return "Medium"

    else:
        return "Low"

def identify_primary_bottleneck(risk_df):

    risk_priority = {
        "High": 3,
        "Medium": 2,
        "Low": 1
    }

    bottleneck_priority = {
        "Storage IOPS": 4,
        "Network": 3,
        "GPU": 2,
        "Storage Capacity": 1
    }

    risk_df = risk_df.copy()

    risk_df["Risk_Score"] = (
        risk_df["Risk"].map(risk_priority)
    )

    risk_df["Bottleneck_Priority"] = (
        risk_df["Area"].map(bottleneck_priority)
    )

    risk_df = risk_df.sort_values(
        by=["Risk_Score", "Bottleneck_Priority"],
        ascending=[False, False]
    )

    return risk_df.iloc[0]["Area"]
def generate_infrastructure_recommendation(
    primary_bottleneck,
    overall_risk
):

    recommendations = {

        "GPU": (
            "Evaluate additional GPU capacity or "
            "reduce model/training concurrency."
        ),

        "Network": (
            "Evaluate higher-bandwidth network fabric "
            "such as 400Gbps+ InfiniBand or RoCE."
        ),

        "Storage IOPS": (
            "Evaluate NVMe-based storage and "
            "increase parallel storage throughput."
        ),

        "Storage Capacity": (
            "Increase storage capacity and evaluate "
            "tiering cold data to lower-cost object/archive storage."
        )
    }

    recommendation = recommendations.get(
        primary_bottleneck,
        "Review infrastructure sizing assumptions."
    )

    if overall_risk == "High":
        recommendation = (
            "Priority action: "
            + recommendation
        )

    return recommendation
# ============================================================
# MAINFRAME WORKLOAD SIMULATOR
# ============================================================

def generate_mainframe_workloads(seed=42):

    np.random.seed(seed)

    workloads = [
        {
            "Workload": "VSAM",
            "Data_TB": 180,
            "Daily_Growth_GB": 350,
            "Peak_IOPS": 85000,
            "Throughput_GBps": 4.5,
            "Batch_Window_Min": 70,
            "SLA_Min": 90,
            "Access_Frequency": "High",
            "Primary_Use": "Transactional"
        },
        {
            "Workload": "Db2",
            "Data_TB": 240,
            "Daily_Growth_GB": 500,
            "Peak_IOPS": 125000,
            "Throughput_GBps": 6.2,
            "Batch_Window_Min": 95,
            "SLA_Min": 120,
            "Access_Frequency": "High",
            "Primary_Use": "Transactional / Analytical"
        },
        {
            "Workload": "SMF",
            "Data_TB": 110,
            "Daily_Growth_GB": 750,
            "Peak_IOPS": 45000,
            "Throughput_GBps": 8.5,
            "Batch_Window_Min": 140,
            "SLA_Min": 180,
            "Access_Frequency": "Medium",
            "Primary_Use": "Performance / Operations"
        },
        {
            "Workload": "Batch",
            "Data_TB": 75,
            "Daily_Growth_GB": 280,
            "Peak_IOPS": 65000,
            "Throughput_GBps": 3.8,
            "Batch_Window_Min": 115,
            "SLA_Min": 120,
            "Access_Frequency": "Medium",
            "Primary_Use": "Batch Processing"
        },
        {
            "Workload": "Application Logs",
            "Data_TB": 95,
            "Daily_Growth_GB": 900,
            "Peak_IOPS": 30000,
            "Throughput_GBps": 5.5,
            "Batch_Window_Min": 180,
            "SLA_Min": 240,
            "Access_Frequency": "Low",
            "Primary_Use": "Historical / AI"
        }
    ]

    df = pd.DataFrame(workloads)

    # --------------------------------------------------------
    # SLA STATUS
    # --------------------------------------------------------

    df["SLA_Status"] = np.where(
        df["Batch_Window_Min"] <= df["SLA_Min"],
        "Within SLA",
        "At Risk"
    )

    # --------------------------------------------------------
    # DATA GROWTH
    # --------------------------------------------------------

    df["Annual_Growth_TB"] = (
        df["Daily_Growth_GB"] * 365 / 1024
    )

    df["Projected_1Y_TB"] = (
        df["Data_TB"] + df["Annual_Growth_TB"]
    )

    # --------------------------------------------------------
    # AI WORKLOAD CLASSIFICATION
    # --------------------------------------------------------

    def classify_ai_workload(row):

        workload = row["Workload"]

        if workload in ["SMF", "Application Logs"]:
            return "AI Analytics"

        elif workload in ["Db2", "VSAM"]:
            return "RAG / Knowledge Retrieval"

        elif workload == "Batch":
            return "Model Training"

        else:
            return "Operational / Not AI Prioritized"

    df["AI_Workload_Type"] = df.apply(
        classify_ai_workload,
        axis=1
    )

    # --------------------------------------------------------
    # AI CANDIDATE PERCENTAGE
    # --------------------------------------------------------

    ai_percentage = {
        "VSAM": 0.35,
        "Db2": 0.45,
        "SMF": 0.70,
        "Batch": 0.25,
        "Application Logs": 0.60
    }

    df["AI_Candidate_Pct"] = df["Workload"].map(
        ai_percentage
    )

    # --------------------------------------------------------
    # AI CANDIDATE DATA VOLUME
    # --------------------------------------------------------

    df["AI_Candidate_TB"] = (
        df["Data_TB"] *
        df["AI_Candidate_Pct"]
    )

    # IMPORTANT:
    # return must remain INSIDE the function

    return df

# Generate synthetic mainframe workload data

mainframe_df = generate_mainframe_workloads()

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Enterprise AI Capacity Co-Pilot",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# EXECUTIVE APP NAVIGATION
# ============================================================

st.sidebar.title("AI Infrastructure Co-Pilot")

app_page = st.sidebar.radio(
    "Navigate to",
    [
        "Executive Overview",
        "Capacity & Economics",
        "Workload Intelligence",
        "AI Architecture",
        "Decision Support"
    ],
    key="main_app_navigation"
)

# ============================================================
# HEADER
# ============================================================

st.title("⚡ Enterprise Hybrid AI & Mainframe Capacity Co-Pilot")

st.markdown(
    """
    ### Executive Infrastructure Capacity Dashboard

    Extending **mainframe storage and capacity engineering**
    disciplines into modern **AI infrastructure planning**.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Infrastructure Assumptions")

dataset_tb = st.sidebar.slider(
    "Total Dataset Size (TB)",
    min_value=10,
    max_value=5000,
    value=500,
    step=10
)

st.sidebar.subheader("Storage Tier Distribution")

hot_pct = st.sidebar.slider(
    "Hot Data (%)",
    min_value=0,
    max_value=100,
    value=15
)

warm_pct = st.sidebar.slider(
    "Warm Data (%)",
    min_value=0,
    max_value=100,
    value=45
)

cold_pct = st.sidebar.slider(
    "Cold Data (%)",
    min_value=0,
    max_value=100,
    value=40
)


# ============================================================
# VALIDATE STORAGE PERCENTAGES
# ============================================================

total_pct = hot_pct + warm_pct + cold_pct

if total_pct != 100:

    st.error(
        f"Hot + Warm + Cold currently equals {total_pct}%. "
        "Please make the total exactly 100%."
    )

    st.stop()


# ============================================================
# STORAGE COST ASSUMPTIONS
# ============================================================

st.sidebar.subheader("Storage Cost ($ / TB / Month)")

hot_cost = st.sidebar.number_input(
    "Hot / NVMe",
    min_value=0.0,
    value=90.0,
    step=5.0
)

warm_cost = st.sidebar.number_input(
    "Warm / Object Storage",
    min_value=0.0,
    value=25.0,
    step=5.0
)

cold_cost = st.sidebar.number_input(
    "Cold / Archive",
    min_value=0.0,
    value=5.0,
    step=1.0
)

st.sidebar.header("AI Infrastructure Assumptions")

model_parameters_b = st.sidebar.selectbox(
    "AI Model Size (B parameters)",
    [7, 13, 34, 70, 175],
    index=3
)

vram_per_gpu_gb = st.sidebar.selectbox(
    "GPU VRAM (GB)",
    [40, 80, 96, 141],
    index=1
)

precision_bytes = st.sidebar.selectbox(
    "Model Precision",
    [2, 1],
    index=0,
    format_func=lambda x: "FP16 / BF16" if x == 2 else "INT8"
)

transfer_window_minutes = st.sidebar.slider(
    "Data Transfer Window (minutes)",
    min_value=30,
    max_value=480,
    value=120,
    step=30
)

iops_block_kb = st.sidebar.selectbox(
    "Storage I/O Block Size (KB)",
    [4, 16, 64, 256],
    index=2
)
# ============================================================
# CALCULATE STORAGE CAPACITY
# ============================================================

hot_tb = dataset_tb * hot_pct / 100
warm_tb = dataset_tb * warm_pct / 100
cold_tb = dataset_tb * cold_pct / 100


# ============================================================
# CALCULATE MONTHLY COST
# ============================================================

hot_monthly_cost = hot_tb * hot_cost
warm_monthly_cost = warm_tb * warm_cost
cold_monthly_cost = cold_tb * cold_cost

total_monthly_cost = (
    hot_monthly_cost
    + warm_monthly_cost
    + cold_monthly_cost
)

total_annual_cost = total_monthly_cost * 12


# ============================================================
# ALL-HOT COMPARISON
# ============================================================

all_hot_monthly_cost = dataset_tb * hot_cost

all_hot_annual_cost = all_hot_monthly_cost * 12

annual_savings = (
    all_hot_annual_cost
    - total_annual_cost
)


# ============================================================
# EXECUTIVE KPIs
# ============================================================
# ============================================================
# MAINFRAME WORKLOAD VIEW
# ============================================================

st.header("Mainframe Workload Intelligence")

st.markdown(
    """
    Synthetic workload telemetry representing a typical
    enterprise mainframe environment.
    """
)

st.dataframe(
    mainframe_df,
    use_container_width=True,
    hide_index=True
)
# ============================================================
# MAINFRAME PERFORMANCE ANALYSIS
# ============================================================

st.subheader("Mainframe Performance Profile")

col1, col2 = st.columns(2)


with col1:

    fig = px.bar(
        mainframe_df,
        x="Workload",
        y="Peak_IOPS",
        title="Peak IOPS by Mainframe Workload",
        text_auto=".2s"
    )

    fig.update_layout(
        yaxis_title="Peak IOPS",
        xaxis_title="Workload"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    fig = px.bar(
        mainframe_df,
        x="Workload",
        y="Throughput_GBps",
        title="Peak Throughput by Mainframe Workload",
        text_auto=".2f"
    )

    fig.update_layout(
        yaxis_title="Throughput (GB/s)",
        xaxis_title="Workload"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================================
# SLA RISK
# ============================================================

st.subheader("Batch Window & SLA Risk")

fig = px.bar(
    mainframe_df,
    x="Workload",
    y=[
        "Batch_Window_Min",
        "SLA_Min"
    ],
    barmode="group",
    title="Batch Processing Time vs SLA"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
# ============================================================
# CAPACITY RISK SUMMARY
# ============================================================

st.subheader("Infrastructure Risk Summary")

risk_df = mainframe_df[
    [
        "Workload",
        "Data_TB",
        "Daily_Growth_GB",
        "Peak_IOPS",
        "Throughput_GBps",
        "Batch_Window_Min",
        "SLA_Min",
        "SLA_Status"
    ]
].copy()

st.dataframe(
    risk_df,
    use_container_width=True,
    hide_index=True
)
# ============================================================
# AI WORKLOAD INTELLIGENCE
# ============================================================

st.header("AI Workload Intelligence")

st.markdown(
    """
    Identifies which enterprise mainframe workloads are potential
    candidates for AI processing and estimates the amount of data
    that may enter the AI pipeline.
    """
)

# ------------------------------------------------------------
# AI DATA SUMMARY
# ------------------------------------------------------------

total_mainframe_tb = mainframe_df["Data_TB"].sum()

total_ai_candidate_tb = (
    mainframe_df["AI_Candidate_TB"].sum()
)

# Scale AI-candidate data to the selected dataset size.
# Assumption: the AI-candidate percentage stays constant.

dataset_scale_factor = (
    dataset_tb / total_mainframe_tb
)

scaled_ai_candidate_tb = (
    total_ai_candidate_tb * dataset_scale_factor
)

# ============================================================
# AI INFRASTRUCTURE CAPACITY MODEL
# ============================================================

ai_capacity = calculate_ai_capacity(
    ai_data_tb=scaled_ai_candidate_tb,
    model_parameters_b=model_parameters_b,
    vram_per_gpu_gb=vram_per_gpu_gb,
    precision_bytes=precision_bytes,
    transfer_window_minutes=transfer_window_minutes,
    iops_block_kb=iops_block_kb
)
infrastructure_risk_df = analyze_infrastructure_risk(
    ai_capacity
)
overall_risk = calculate_overall_risk(
    infrastructure_risk_df
)
primary_bottleneck = identify_primary_bottleneck(
    infrastructure_risk_df
)
recommendation = generate_infrastructure_recommendation(
    primary_bottleneck,
    overall_risk
)

st.subheader("Executive AI Infrastructure Capacity Summary")

st.markdown(
    f"""
    **Executive Assessment:** The modeled AI infrastructure has an
    overall risk level of **{overall_risk}**.

    **Primary Bottleneck:** {primary_bottleneck}

    **Recommended Action:** {recommendation}
    """
)

st.caption(
    "Planning estimate based on synthetic workload data and "
    "deterministic sizing assumptions. Validate against actual "
    "workload telemetry, vendor specifications, and performance targets "
    "before making production infrastructure decisions."
)


st.markdown("### Key Capacity Requirements")

col1, col2, col3 = st.columns(3)

col1.metric(
    "AI Dataset",
    f'{ai_capacity["AI_Data_TB"]:,.1f} TB'
)

col2.metric(
    "Estimated AI Storage",
    f'{ai_capacity["AI_Storage_TB"]:,.1f} TB'
)

col3.metric(
    "Estimated GPU Count",
    f'{ai_capacity["GPU_Count"]:,}'
)

col4, col5, col6 = st.columns(3)

col4.metric(
    "Required Storage IOPS",
    f'{ai_capacity["Required_IOPS"]:,.0f}'
)

col5.metric(
    "Storage Throughput",
    f'{ai_capacity["Storage_Throughput_Gbps"]:,.1f} Gbps'
)

col6.metric(
    "Network Bandwidth",
    f'{ai_capacity["Network_Gbps"]:,.1f} Gbps'
)


with st.expander("Sizing Assumptions & Limitations"):

    st.markdown(
        """
        **Data assumptions**
        - AI-eligible data is estimated using a constant percentage
          of the selected total dataset size.
        - AI storage is estimated at 1.5 times the AI dataset size.

        **Performance assumptions**
        - Storage throughput and network bandwidth are theoretical
          averages based on the selected transfer window.
        - Required IOPS is derived from modeled throughput and the
          selected I/O block size; it is not measured workload IOPS.
        - GPU count uses estimated training memory divided by the
          selected GPU VRAM, rounded up to a whole GPU.

        **Validation required before production use**
        - Validate storage throughput, IOPS, and latency against
          actual workload measurements and vendor specifications.
        - Validate network bandwidth against the target architecture
          and protocol overhead.
        - Validate GPU requirements against model architecture,
          training strategy, parallelism, and framework overhead.

        **Classification**
        This application is a capacity-planning prototype using
        synthetic workload data and deterministic rules. Its outputs
        are not production sizing guarantees.
        """
    )

st.subheader("Executive Infrastructure Assessment")

risk_col, bottleneck_col = st.columns(2)

risk_col.metric(
    "Overall AI Infrastructure Risk",
    overall_risk
)

bottleneck_col.metric(
    "Primary Bottleneck",
    primary_bottleneck
)
st.info(
    f"**Recommended Action:** {recommendation}"
)
st.subheader("Infrastructure Bottleneck & Risk Analysis")
st.dataframe(
    infrastructure_risk_df,
    use_container_width=True,
    hide_index=True
)
st.header("AI Infrastructure Capacity Model")

col1, col2, col3, col4 = st.columns(4)
st.metric(
    "Required Storage Throughput",
    f"{ai_capacity['Storage_Throughput_Gbps']:.1f} Gbps"
)
col1.metric(
    "AI Data",
    f"{ai_capacity['AI_Data_TB']:.1f} TB"
)

col2.metric(
    "AI Storage Required",
    f"{ai_capacity['AI_Storage_TB']:.1f} TB"
)

col3.metric(
    "Required IOPS",
    f"{ai_capacity['Required_IOPS']:,.0f}"
)

col4.metric(
    "Network Bandwidth",
    f"{ai_capacity['Network_Gbps']:.1f} Gbps"
)

col5, col6, col7 = st.columns(3)

col5.metric(
    "Model Memory",
    f"{ai_capacity['Model_Memory_GB']:.1f} GB"
)

col6.metric(
    "Training Memory",
    f"{ai_capacity['Training_Memory_GB']:.1f} GB"
)

col7.metric(
    "Estimated GPUs",
    f"{ai_capacity['GPU_Count']}"
)
ai_candidate_pct = (
    total_ai_candidate_tb /
    total_mainframe_tb *
    100
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Mainframe Data",
        f"{total_mainframe_tb:,.0f} TB"
    )

with col2:

    st.metric(
        "AI Candidate Data",
        f"{total_ai_candidate_tb:,.0f} TB"
    )

with col3:

    st.metric(
        "AI Candidate %",
        f"{ai_candidate_pct:.1f}%"
    )


# ------------------------------------------------------------
# AI WORKLOAD DISTRIBUTION
# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    ai_summary = (
        mainframe_df
        .groupby("AI_Workload_Type")["AI_Candidate_TB"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        ai_summary,
        names="AI_Workload_Type",
        values="AI_Candidate_TB",
        hole=0.45,
        title="AI Candidate Data by Workload Type"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    fig = px.bar(
        mainframe_df,
        x="Workload",
        y="AI_Candidate_TB",
        color="AI_Workload_Type",
        title="AI Candidate Data by Mainframe Workload",
        text_auto=".1f"
    )

    fig.update_layout(
        yaxis_title="AI Candidate Data (TB)",
        xaxis_title="Mainframe Workload"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ------------------------------------------------------------
# AI CANDIDATE DETAIL
# ------------------------------------------------------------

st.subheader("AI Data Pipeline Candidates")

ai_candidate_df = mainframe_df[
    [
        "Workload",
        "Data_TB",
        "AI_Workload_Type",
        "AI_Candidate_Pct",
        "AI_Candidate_TB"
    ]
].copy()

ai_candidate_df["AI_Candidate_Pct"] = (
    ai_candidate_df["AI_Candidate_Pct"] * 100
)

st.dataframe(
    ai_candidate_df,
    use_container_width=True,
    hide_index=True
)

st.header("Executive Storage View")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Dataset",
        f"{dataset_tb:,.0f} TB"
    )

with col2:

    st.metric(
        "Monthly Storage Cost",
        f"${total_monthly_cost:,.0f}"
    )

with col3:

    st.metric(
        "Annual Storage Cost",
        f"${total_annual_cost:,.0f}"
    )

with col4:

    st.metric(
        "Annual Savings vs All-Hot",
        f"${annual_savings:,.0f}"
    )


# ============================================================
# STORAGE DATAFRAME
# ============================================================

storage_df = pd.DataFrame(
    {
        "Storage Tier": [
            "Hot / NVMe",
            "Warm / Object",
            "Cold / Archive"
        ],

        "Capacity (TB)": [
            hot_tb,
            warm_tb,
            cold_tb
        ],

        "Cost / TB / Month": [
            hot_cost,
            warm_cost,
            cold_cost
        ],

        "Monthly Cost": [
            hot_monthly_cost,
            warm_monthly_cost,
            cold_monthly_cost
        ]
    }
)


# ============================================================
# VISUALIZATIONS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    fig = px.pie(
        storage_df,
        names="Storage Tier",
        values="Capacity (TB)",
        hole=0.45,
        title="Enterprise Data Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    fig = px.bar(
        storage_df,
        x="Storage Tier",
        y="Monthly Cost",
        title="Monthly Storage Cost by Tier",
        text_auto=".2s"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# DETAIL TABLE
# ============================================================

st.subheader("Storage Tier Economics")

st.dataframe(
    storage_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# EXECUTIVE INSIGHT
# ============================================================

st.subheader("Executive Insight")

if annual_savings > 0:

    st.success(
        f"""
        The hybrid storage strategy reduces estimated annual storage
        cost by **${annual_savings:,.0f}** compared with keeping the
        entire {dataset_tb:,.0f} TB dataset in the Hot/NVMe tier.

        This demonstrates the same lifecycle optimization principle
        used in enterprise mainframe storage: keep performance-sensitive
        data on expensive high-performance storage while moving less
        frequently accessed data to lower-cost tiers.
        """
    )

else:

    st.warning(
        "The current tier configuration does not produce savings "
        "relative to the all-hot scenario."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Enterprise Hybrid AI & Mainframe Storage/Capacity Co-Pilot "
    "| Synthetic planning model"
)


# ============================================================
# STEP 8 - AI DATA PIPELINE ARCHITECTURE
# ============================================================

st.divider()

st.header("AI Data Pipeline Architecture")

st.caption(
    "Plan the movement of enterprise mainframe data into AI-ready "
    "data products and downstream AI infrastructure."
)

st.info(
    "Prototype only: this section models an illustrative architecture. "
    "It does not connect to live mainframe, AWS, or cloud data services."
)

# ------------------------------------------------------------
# 8.1 - MAINFRAME DATA SOURCES
# ------------------------------------------------------------

pipeline_sources = pd.DataFrame([
    {
        "Source": "VSAM",
        "Data Type": "Operational records",
        "Potential AI Use": "Operational pattern analysis",
        "Default Ingestion": "Batch extract"
    },
    {
        "Source": "Db2",
        "Data Type": "Structured business data",
        "Potential AI Use": "Predictive analytics",
        "Default Ingestion": "Batch or CDC"
    },
    {
        "Source": "SMF",
        "Data Type": "System and performance records",
        "Potential AI Use": "Capacity and anomaly analysis",
        "Default Ingestion": "Batch or streaming"
    },
    {
        "Source": "Batch",
        "Data Type": "Job execution records",
        "Potential AI Use": "Job performance analysis",
        "Default Ingestion": "Batch extract"
    },
    {
        "Source": "Application Logs",
        "Data Type": "Events and messages",
        "Potential AI Use": "Incident analysis and retrieval",
        "Default Ingestion": "Batch or streaming"
    }
])

st.subheader("1. Mainframe Data Sources")

st.dataframe(
    pipeline_sources,
    use_container_width=True,
    hide_index=True
)

# ------------------------------------------------------------
# 8.2 - VISUAL PIPELINE ARCHITECTURE
# ------------------------------------------------------------

st.subheader("2. End-to-End Pipeline Architecture")

pipeline_layers = [
    {
        "name": "Mainframe Sources",
        "detail": "VSAM | Db2 | SMF | Batch | Application Logs"
    },
    {
        "name": "Ingestion",
        "detail": "Batch extraction | CDC | Streaming"
    },
    {
        "name": "Raw Data Layer",
        "detail": "Source-preserved data and ingestion metadata"
    },
    {
        "name": "Transformation",
        "detail": "Parsing | Normalization | Schema mapping"
    },
    {
        "name": "Data Quality & Governance",
        "detail": "Validation | Access controls | Sensitive-data handling"
    },
    {
        "name": "AI-Ready Data",
        "detail": "Curated datasets | Documents | Features"
    },
    {
        "name": "AI Consumption",
        "detail": "Analytics | Model training | Inference | Retrieval"
    }
]

for index, layer in enumerate(pipeline_layers):
    with st.container(border=True):
        st.markdown(f"**{index + 1}. {layer['name']}**")
        st.caption(layer["detail"])

    if index < len(pipeline_layers) - 1:
        st.markdown(
            "<div style='text-align:center; font-size:24px;'>↓</div>",
            unsafe_allow_html=True
        )

# ------------------------------------------------------------
# 8.3 - INTERACTIVE INGESTION PLANNING
# ------------------------------------------------------------

st.subheader("3. Ingestion Planning")

source_options = pipeline_sources["Source"].tolist()

selected_source = st.selectbox(
    "Select a mainframe source",
    source_options,
    key="step8_selected_source"
)

ingestion_options = [
    "Batch extraction",
    "Change Data Capture (CDC)",
    "Streaming"
]

selected_ingestion = st.selectbox(
    "Select an ingestion method",
    ingestion_options,
    key="step8_selected_ingestion"
)

ingestion_guidance = {
    "Batch extraction": (
        "Suitable for scheduled extracts and periodic processing. "
        "Define batch windows, restartability, reconciliation, and "
        "source-system impact controls."
    ),
    "Change Data Capture (CDC)": (
        "Suitable when supported source changes must be captured "
        "incrementally. Validate source support, log access, ordering, "
        "recovery, and replication latency."
    ),
    "Streaming": (
        "Suitable for event-driven use cases that need low-latency "
        "delivery. Validate event support, throughput, ordering, "
        "backpressure, and replay behavior."
    )
}

selected_source_row = pipeline_sources[
    pipeline_sources["Source"] == selected_source
].iloc[0]

st.markdown("**Selected source profile**")
st.write(
    f"Data type: {selected_source_row['Data Type']}"
)
st.write(
    f"Potential AI use: {selected_source_row['Potential AI Use']}"
)

st.info(ingestion_guidance[selected_ingestion])

# ------------------------------------------------------------
# 8.4 - TRANSFORMATION, QUALITY, AND DESTINATION PLANNING
# ------------------------------------------------------------

st.subheader("4. Transformation and Data Quality")

st.markdown(
    "Select the controls that should be included in the proposed pipeline."
)

quality_schema = st.checkbox(
    "Schema and format validation",
    value=True,
    key="step8_schema_validation"
)

quality_completeness = st.checkbox(
    "Completeness and record-count reconciliation",
    value=True,
    key="step8_completeness"
)

quality_sensitive = st.checkbox(
    "Sensitive-data identification and access controls",
    value=True,
    key="step8_sensitive_data"
)

quality_lineage = st.checkbox(
    "Data lineage and ingestion metadata",
    value=True,
    key="step8_lineage"
)

destination_options = [
    "Object storage / data lake",
    "Curated analytical tables",
    "Document store for retrieval",
    "Feature store",
    "Multiple destinations"
]

selected_destination = st.selectbox(
    "Select the intended AI-ready destination",
    destination_options,
    key="step8_destination"
)

st.markdown("**Proposed pipeline configuration**")

configuration = {
    "Source": selected_source,
    "Ingestion Method": selected_ingestion,
    "Destination": selected_destination,
    "Schema Validation": quality_schema,
    "Completeness Reconciliation": quality_completeness,
    "Sensitive-Data Controls": quality_sensitive,
    "Lineage and Metadata": quality_lineage
}

st.dataframe(
    pd.DataFrame(
        [
            {"Setting": name, "Selected Value": value}
            for name, value in configuration.items()
        ]
    ),
    use_container_width=True,
    hide_index=True
)

# ------------------------------------------------------------
# 8.5 - CONNECTION TO EXISTING CAPACITY PLANNING
# ------------------------------------------------------------

st.subheader("5. Pipeline-to-Infrastructure Considerations")

st.markdown(
    "Pipeline design should be checked against the capacity estimates "
    "from Step 7. The following metrics are shown only when the existing "
    "capacity result is available."
)

if "ai_capacity" in globals():

    cap_col1, cap_col2, cap_col3 = st.columns(3)

    cap_col1.metric(
        "AI Dataset",
        f'{ai_capacity["AI_Data_TB"]:,.1f} TB'
    )

    cap_col2.metric(
        "Storage Throughput",
        f'{ai_capacity["Storage_Throughput_Gbps"]:,.1f} Gbps'
    )

    cap_col3.metric(
        "Network Bandwidth",
        f'{ai_capacity["Network_Gbps"]:,.1f} Gbps'
    )

    st.caption(
        "These are theoretical planning estimates from Step 7, not "
        "measured pipeline throughput or a guarantee of transfer time."
    )

else:
    st.warning(
        "The existing ai_capacity result was not found in the current "
        "application scope. Pipeline planning remains available, but "
        "capacity metrics are not displayed in this section."
    )

st.markdown("**Architecture validation checklist**")

st.checkbox(
    "Confirm source-system access and extraction constraints",
    key="step8_check_source"
)

st.checkbox(
    "Define transfer window, throughput, and latency objectives",
    key="step8_check_performance"
)

st.checkbox(
    "Define data quality, governance, and security requirements",
    key="step8_check_governance"
)

st.checkbox(
    "Validate destination compatibility with the intended AI workload",
    key="step8_check_destination"
)

st.warning(
    "Production readiness requires real source integration, operational "
    "monitoring, failure recovery, security testing, and measured "
    "performance validation. Selecting controls here records a planning "
    "configuration; it does not execute pipeline operations."
)


# ============================================================
# STEP 9 - INTERACTIVE ARCHITECTURE VISUALIZATION
# ============================================================

st.divider()

st.header("Interactive AI Infrastructure Architecture")

st.caption(
    "Explore how mainframe data flows through the AI data pipeline "
    "and connects to infrastructure capacity requirements."
)

st.info(
    "This is an interactive architecture model. It does not represent "
    "live connections, deployed services, or real-time telemetry."
)

# ------------------------------------------------------------
# 9.1 - DEFINE ARCHITECTURE COMPONENTS
# ------------------------------------------------------------

architecture_components = {
    "Mainframe Sources": {
        "layer": "Data Sources",
        "description": (
            "Enterprise data originating from VSAM, Db2, SMF, "
            "batch processing, and application logs."
        ),
        "dependency": (
            "Source access, extraction windows, record formats, "
            "and mainframe processing constraints."
        )
    },
    "Ingestion": {
        "layer": "Data Movement",
        "description": (
            "Batch extraction, change data capture, or streaming "
            "moves supported source data into the target platform."
        ),
        "dependency": (
            "Transfer throughput, source impact, connectivity, "
            "recovery, and delivery latency."
        )
    },
    "Raw Data": {
        "layer": "Data Storage",
        "description": (
            "Preserves ingested data and source metadata for "
            "reconciliation, replay, and downstream processing."
        ),
        "dependency": (
            "Landing-zone storage capacity, write throughput, "
            "retention, and access controls."
        )
    },
    "Transformation": {
        "layer": "Data Processing",
        "description": (
            "Parses, normalizes, enriches, and transforms source "
            "records into consistent data structures."
        ),
        "dependency": (
            "CPU or compute capacity, memory, processing time, "
            "schema handling, and data volume."
        )
    },
    "Data Quality & Governance": {
        "layer": "Controls",
        "description": (
            "Validates data completeness and quality, tracks lineage, "
            "and applies security and sensitive-data controls."
        ),
        "dependency": (
            "Validation rules, identity and access management, "
            "auditability, and governance policies."
        )
    },
    "AI-Ready Data": {
        "layer": "Curated Data",
        "description": (
            "Produces curated datasets, documents, or features "
            "suitable for approved AI use cases."
        ),
        "dependency": (
            "Curated storage, metadata, retrieval preparation, "
            "and data freshness requirements."
        )
    },
    "AI Infrastructure": {
        "layer": "Compute and Network",
        "description": (
            "Provides storage, network connectivity, GPU compute, "
            "and memory for model training or inference."
        ),
        "dependency": (
            "Storage IOPS and throughput, network bandwidth, "
            "GPU memory, GPU count, and workload concurrency."
        )
    }
}

architecture_order = list(architecture_components.keys())

# ------------------------------------------------------------
# 9.2 - DRAW THE ARCHITECTURE FLOW
# ------------------------------------------------------------

st.subheader("1. End-to-End Architecture")

st.markdown(
    "Select a component below the diagram to inspect its role "
    "and infrastructure dependencies."
)

# First row
row1 = st.columns(3)

with row1[0]:
    with st.container(border=True):
        st.markdown("**1. Mainframe Sources**")
        st.caption("VSAM · Db2 · SMF · Batch · Logs")

with row1[1]:
    with st.container(border=True):
        st.markdown("**2. Ingestion**")
        st.caption("Batch · CDC · Streaming")

with row1[2]:
    with st.container(border=True):
        st.markdown("**3. Raw Data**")
        st.caption("Landing zone · Source metadata")

st.markdown(
    "<div style='text-align:center; font-size:24px;'>"
    "↓ Data preparation and controls ↓"
    "</div>",
    unsafe_allow_html=True
)

# Second row
row2 = st.columns(2)

with row2[0]:
    with st.container(border=True):
        st.markdown("**4. Transformation**")
        st.caption("Parsing · Normalization · Enrichment")

with row2[1]:
    with st.container(border=True):
        st.markdown("**5. Data Quality & Governance**")
        st.caption("Validation · Security · Lineage")

st.markdown(
    "<div style='text-align:center; font-size:24px;'>"
    "↓ Curated AI data ↓"
    "</div>",
    unsafe_allow_html=True
)

# Third row
row3 = st.columns(2)

with row3[0]:
    with st.container(border=True):
        st.markdown("**6. AI-Ready Data**")
        st.caption("Curated datasets · Documents · Features")

with row3[1]:
    with st.container(border=True):
        st.markdown("**7. AI Infrastructure**")
        st.caption("Storage · Network · GPU compute")

# ------------------------------------------------------------
# 9.3 - INTERACTIVE COMPONENT EXPLORER
# ------------------------------------------------------------

st.subheader("2. Explore Architecture Components")

selected_component = st.selectbox(
    "Choose a component",
    architecture_order,
    key="step9_selected_component"
)

component_details = architecture_components[selected_component]

st.markdown(f"### {selected_component}")

st.write(
    f"**Architecture layer:** {component_details['layer']}"
)

st.write(
    f"**Purpose:** {component_details['description']}"
)

st.write(
    f"**Infrastructure dependencies:** "
    f"{component_details['dependency']}"
)

# ------------------------------------------------------------
# 9.4 - PIPELINE-TO-INFRASTRUCTURE DEPENDENCY VIEW
# ------------------------------------------------------------

st.subheader("3. Pipeline and Infrastructure Dependencies")

dependency_rows = [
    {
        "Pipeline Concern": "Data transfer",
        "Infrastructure Dependency": "Network bandwidth",
        "Planning Question": (
            "Can the data be transferred within the required window?"
        )
    },
    {
        "Pipeline Concern": "Data landing and retention",
        "Infrastructure Dependency": "Storage capacity and throughput",
        "Planning Question": (
            "Can storage handle the incoming data volume and write rate?"
        )
    },
    {
        "Pipeline Concern": "Transformation and scans",
        "Infrastructure Dependency": "Compute and storage IOPS",
        "Planning Question": (
            "Can processing meet the required completion time?"
        )
    },
    {
        "Pipeline Concern": "Model training",
        "Infrastructure Dependency": "GPU memory and GPU capacity",
        "Planning Question": (
            "Can the selected model and training strategy fit the available resources?"
        )
    },
    {
        "Pipeline Concern": "AI inference and retrieval",
        "Infrastructure Dependency": "Network, memory, and data access",
        "Planning Question": (
            "Can data and model responses meet the target latency?"
        )
    }
]

st.dataframe(
    pd.DataFrame(dependency_rows),
    use_container_width=True,
    hide_index=True
)

# ------------------------------------------------------------
# 9.5 - EXISTING CAPACITY MODEL INTEGRATION
# ------------------------------------------------------------

st.subheader("4. Capacity Planning Reference")

if "ai_capacity" in globals():

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "AI Dataset",
        f'{ai_capacity["AI_Data_TB"]:,.1f} TB'
    )

    metric2.metric(
        "Storage Throughput",
        f'{ai_capacity["Storage_Throughput_Gbps"]:,.1f} Gbps'
    )

    metric3.metric(
        "Network Bandwidth",
        f'{ai_capacity["Network_Gbps"]:,.1f} Gbps'
    )

    metric4, metric5, metric6 = st.columns(3)

    metric4.metric(
        "Required Storage IOPS",
        f'{ai_capacity["Required_IOPS"]:,.0f}'
    )

    metric5.metric(
        "Estimated GPU Count",
        f'{ai_capacity["GPU_Count"]:,}'
    )

    metric6.metric(
        "Estimated AI Storage",
        f'{ai_capacity["AI_Storage_TB"]:,.1f} TB'
    )

else:
    st.warning(
        "The capacity model is not available in this section's "
        "current scope. The architecture visualization remains available."
    )

st.caption(
    "Capacity values are theoretical estimates from the existing model. "
    "They are not measurements of actual pipeline performance."
)

# ------------------------------------------------------------
# 9.6 - ARCHITECTURE REVIEW CHECKLIST
# ------------------------------------------------------------

st.subheader("5. Architecture Review Checklist")

st.checkbox(
    "Source access and data movement method have been identified",
    key="step9_source_review"
)

st.checkbox(
    "Network and storage requirements have been considered",
    key="step9_storage_network_review"
)

st.checkbox(
    "Data quality, security, and governance requirements have been considered",
    key="step9_governance_review"
)

st.checkbox(
    "AI workload requirements have been mapped to infrastructure",
    key="step9_ai_review"
)

st.warning(
    "The checklist records selections in the current app session. "
    "It does not certify architecture readiness or persist an approval record."
)


# ============================================================
# STEP 10 - AI DECISION-SUPPORT CO-PILOT
# ============================================================

st.divider()

st.header("AI Infrastructure Decision-Support Co-Pilot")

st.caption(
    "Explore infrastructure concerns, review modeled evidence, "
    "and identify practical validation and mitigation actions."
)

st.info(
    "This prototype uses deterministic decision rules and modeled "
    "capacity estimates. It does not use an LLM or live infrastructure telemetry."
)

# ------------------------------------------------------------
# 10.1 - USER INPUTS
# ------------------------------------------------------------

decision_area = st.selectbox(
    "What infrastructure concern do you want to investigate?",
    [
        "Automatic: Primary Bottleneck",
        "Storage IOPS",
        "Network Bandwidth",
        "GPU Capacity and Memory",
        "Storage Capacity",
        "Data Pipeline Performance"
    ],
    key="step10_decision_area"
)

decision_priority = st.selectbox(
    "Business priority",
    [
        "Meet a data processing window",
        "Control infrastructure cost",
        "Reduce performance risk",
        "Prepare for AI workload growth"
    ],
    key="step10_business_priority"
)

# ------------------------------------------------------------
# 10.2 - EVIDENCE AND RECOMMENDATION RULES
# ------------------------------------------------------------

decision_rules = {
    "Storage IOPS": {
        "concern": "Storage I/O demand may exceed the target platform's capability.",
        "evidence": (
            "The modeled IOPS estimate is derived from dataset transfer "
            "throughput and the selected I/O block size."
        ),
        "actions": [
            "Measure workload IOPS, read/write mix, and latency requirements.",
            "Compare measured demand with the target storage platform's limits.",
            "Evaluate parallelism, caching, and suitable high-performance storage.",
            "Repeat the sizing exercise using measured workload characteristics."
        ]
    },
    "Network Bandwidth": {
        "concern": "The transfer window may require substantial network bandwidth.",
        "evidence": (
            "The modeled bandwidth estimate depends on AI dataset size "
            "and the selected transfer window."
        ),
        "actions": [
            "Confirm the required transfer window and effective data volume.",
            "Measure available end-to-end bandwidth and protocol overhead.",
            "Check for network contention and concurrent data transfers.",
            "Evaluate transfer scheduling or higher-bandwidth connectivity if needed."
        ]
    },
    "GPU Capacity and Memory": {
        "concern": "The selected model may require significant GPU memory and compute capacity.",
        "evidence": (
            "GPU count is estimated from model memory, a training-memory "
            "multiplier, and selected GPU VRAM."
        ),
        "actions": [
            "Confirm model architecture, precision, and training strategy.",
            "Validate memory requirements against the selected framework.",
            "Account for parallelism, activations, optimizer state, and runtime overhead.",
            "Benchmark representative training or inference workloads before procurement."
        ]
    },
    "Storage Capacity": {
        "concern": "The AI dataset and its additional storage requirements may increase capacity needs.",
        "evidence": (
            "The model estimates AI storage at 1.5 times the AI dataset size."
        ),
        "actions": [
            "Confirm the retained dataset size and growth rate.",
            "Include replicas, checkpoints, intermediate data, and retention requirements.",
            "Separate hot, warm, and archival data where appropriate.",
            "Validate usable capacity after redundancy and platform overhead."
        ]
    },
    "Data Pipeline Performance": {
        "concern": "The pipeline may not deliver AI-ready data within the required time.",
        "evidence": (
            "The architecture describes pipeline stages, but this prototype "
            "does not measure actual pipeline execution."
        ),
        "actions": [
            "Define end-to-end freshness and completion-time objectives.",
            "Measure ingestion, transformation, and data-quality stage durations.",
            "Identify retries, bottlenecks, source-system constraints, and data-quality failures.",
            "Run a representative end-to-end test with realistic data volumes."
        ]
    }
}

# ------------------------------------------------------------
# 10.3 - SELECT THE INVESTIGATION AREA
# ------------------------------------------------------------

if decision_area == "Automatic: Primary Bottleneck":

    if "primary_bottleneck" in globals():
        investigation_area = primary_bottleneck
    else:
        investigation_area = "Data Pipeline Performance"

else:
    investigation_area = decision_area

# ------------------------------------------------------------
# 10.4 - DISPLAY THE DECISION BRIEF
# ------------------------------------------------------------

st.subheader("Decision Brief")

st.write(f"**Investigation area:** {investigation_area}")
st.write(f"**Business priority:** {decision_priority}")

if investigation_area in decision_rules:

    selected_rule = decision_rules[investigation_area]

    st.markdown("### Assessment")
    st.write(selected_rule["concern"])

    st.markdown("### Supporting evidence")
    st.write(selected_rule["evidence"])

    # Show relevant modeled metrics when available.
    if "ai_capacity" in globals():

        if investigation_area == "Storage IOPS":
            st.metric(
                "Modeled Required IOPS",
                f'{ai_capacity["Required_IOPS"]:,.0f}'
            )

        elif investigation_area == "Network Bandwidth":
            st.metric(
                "Modeled Network Bandwidth",
                f'{ai_capacity["Network_Gbps"]:,.1f} Gbps'
            )

        elif investigation_area == "GPU Capacity and Memory":

            gpu_col1, gpu_col2 = st.columns(2)

            gpu_col1.metric(
                "Estimated GPU Count",
                f'{ai_capacity["GPU_Count"]:,}'
            )

            gpu_col2.metric(
                "Estimated Training Memory",
                f'{ai_capacity["Training_Memory_GB"]:,.1f} GB'
            )

        elif investigation_area == "Storage Capacity":
            st.metric(
                "Estimated AI Storage",
                f'{ai_capacity["AI_Storage_TB"]:,.1f} TB'
            )

    st.markdown("### Recommended next actions")

    for action in selected_rule["actions"]:
        st.markdown(f"- {action}")

    # Business-priority guidance
    priority_guidance = {
        "Meet a data processing window": (
            "Prioritize end-to-end timing measurements and identify "
            "which pipeline or infrastructure stage consumes the most time."
        ),
        "Control infrastructure cost": (
            "Compare workload demand with provisioned capacity, utilization, "
            "retention requirements, and tiering opportunities."
        ),
        "Reduce performance risk": (
            "Validate performance limits, establish service objectives, "
            "and test representative peak-load conditions."
        ),
        "Prepare for AI workload growth": (
            "Model multiple growth scenarios and validate scaling limits "
            "for storage, network, memory, and compute."
        )
    }

    st.markdown("### Priority-specific guidance")
    st.write(priority_guidance[decision_priority])

else:
    st.warning(
        "No recommendation rule is configured for this investigation area."
    )

# ------------------------------------------------------------
# 10.5 - EXISTING OVERALL RISK CONTEXT
# ------------------------------------------------------------

st.subheader("Overall Infrastructure Context")

if "overall_risk" in globals():
    st.metric(
        "Modeled Overall Infrastructure Risk",
        overall_risk
    )

if "primary_bottleneck" in globals():
    st.write(f"**Current primary bottleneck:** {primary_bottleneck}")

if "recommendation" in globals():
    st.write(f"**Current model recommendation:** {recommendation}")

st.caption(
    "Recommendations are rule-based planning guidance, not automated "
    "operational commands. Validate assumptions with actual telemetry, "
    "vendor specifications, security requirements, and workload testing."
)


# ============================================================
# STEP 11 - SENIOR LEADERSHIP TEAM EXECUTIVE DASHBOARD
# ============================================================

st.divider()

st.header("Executive Dashboard | AI Infrastructure Readiness")

st.caption(
    "Leadership view of modeled AI capacity, infrastructure risk, "
    "primary bottleneck, and recommended next actions."
)

st.info(
    "This dashboard summarizes the existing planning model. "
    "It is not a production readiness certification or live telemetry view."
)

# ------------------------------------------------------------
# 11.1 - EXECUTIVE ASSESSMENT
# ------------------------------------------------------------

if "overall_risk" in globals():
    dashboard_overall_risk = overall_risk
else:
    dashboard_overall_risk = "Not available"

if "primary_bottleneck" in globals():
    dashboard_bottleneck = primary_bottleneck
else:
    dashboard_bottleneck = "Not available"

if "recommendation" in globals():
    dashboard_recommendation = recommendation
else:
    dashboard_recommendation = (
        "Review capacity assumptions and infrastructure risk results."
    )

st.subheader("1. Executive Assessment")

risk_col, bottleneck_col = st.columns(2)

risk_col.metric(
    "Overall Infrastructure Risk",
    dashboard_overall_risk
)

bottleneck_col.metric(
    "Primary Bottleneck",
    dashboard_bottleneck
)

st.markdown("**Recommended Priority Action**")
st.write(dashboard_recommendation)

# ------------------------------------------------------------
# 11.2 - CAPACITY SCORECARD
# ------------------------------------------------------------

st.subheader("2. AI Infrastructure Capacity Scorecard")

if "ai_capacity" in globals():

    capacity_col1, capacity_col2, capacity_col3 = st.columns(3)

    capacity_col1.metric(
        "AI Dataset",
        f'{ai_capacity["AI_Data_TB"]:,.1f} TB'
    )

    capacity_col2.metric(
        "Estimated AI Storage",
        f'{ai_capacity["AI_Storage_TB"]:,.1f} TB'
    )

    capacity_col3.metric(
        "Estimated GPU Count",
        f'{ai_capacity["GPU_Count"]:,}'
    )

    capacity_col4, capacity_col5, capacity_col6 = st.columns(3)

    capacity_col4.metric(
        "Required Storage IOPS",
        f'{ai_capacity["Required_IOPS"]:,.0f}'
    )

    capacity_col5.metric(
        "Storage Throughput",
        f'{ai_capacity["Storage_Throughput_Gbps"]:,.1f} Gbps'
    )

    capacity_col6.metric(
        "Network Bandwidth",
        f'{ai_capacity["Network_Gbps"]:,.1f} Gbps'
    )

else:
    st.warning(
        "Capacity results are not available in this section. "
        "Review the scope of the existing ai_capacity variable."
    )

# ------------------------------------------------------------
# 11.3 - RISK BREAKDOWN
# ------------------------------------------------------------

st.subheader("3. Infrastructure Risk Breakdown")

if "infrastructure_risk_df" in globals():

    st.dataframe(
        infrastructure_risk_df,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning(
        "The infrastructure risk table is not available in this section. "
        "Review the scope of infrastructure_risk_df."
    )

# ------------------------------------------------------------
# 11.4 - LEADERSHIP DECISION GUIDE
# ------------------------------------------------------------

st.subheader("4. Leadership Decision Guide")

decision_guidance = {
    "Storage IOPS": (
        "Validate storage I/O demand, latency, and parallel throughput "
        "against the target storage platform."
    ),
    "Network": (
        "Validate effective transfer bandwidth, network contention, "
        "and the required data movement window."
    ),
    "GPU": (
        "Validate GPU memory, compute requirements, model configuration, "
        "and training or inference concurrency."
    ),
    "Storage Capacity": (
        "Validate usable capacity, data retention, replication, "
        "growth assumptions, and storage tiering."
    )
}

if dashboard_bottleneck in decision_guidance:
    st.write(decision_guidance[dashboard_bottleneck])
else:
    st.write(
        "Review the primary bottleneck and validate the relevant "
        "infrastructure assumptions before committing investment."
    )

st.markdown("**Before approving an infrastructure design:**")

st.markdown(
    "- Validate the model against actual workload telemetry.\n"
    "- Confirm storage, network, and GPU specifications with vendors.\n"
    "- Test representative workloads and expected peak demand.\n"
    "- Confirm security, governance, availability, and recovery requirements."
)

# ------------------------------------------------------------
# 11.5 - EXECUTIVE CAVEATS
# ------------------------------------------------------------

with st.expander("Executive Assumptions and Limitations"):

    st.markdown(
        """
        - Workload data is synthetic and used for demonstration.
        - AI-eligible data is estimated using assumed source percentages.
        - AI storage uses a fixed multiplier.
        - IOPS and throughput are theoretical estimates, not measured
          production workload characteristics.
        - GPU requirements use a simplified training-memory heuristic.
        - Risk classifications use illustrative thresholds.
        - Recommendations are deterministic rules, not LLM-generated
          advice or automated infrastructure actions.

        **Executive interpretation:** Use this dashboard to structure
        capacity planning and identify questions requiring validation.
        Do not use it alone to approve production infrastructure
        procurement or certify production readiness.
        """
    )

