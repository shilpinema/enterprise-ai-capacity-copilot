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
