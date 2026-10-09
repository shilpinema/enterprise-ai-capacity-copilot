import streamlit as st
import pandas as pd
import plotly.express as px

import numpy as np


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

    # Calculate whether the workload is currently
    # completing inside its batch SLA.

    df["SLA_Status"] = np.where(
        df["Batch_Window_Min"] <= df["SLA_Min"],
        "Within SLA",
        "At Risk"
    )

    # Estimate annual data growth.

    df["Annual_Growth_TB"] = (
        df["Daily_Growth_GB"] * 365 / 1024
    )

    # Projected data after one year.

    df["Projected_1Y_TB"] = (
        df["Data_TB"] + df["Annual_Growth_TB"]
    )

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
