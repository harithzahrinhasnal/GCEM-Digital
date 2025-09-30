import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Credit Spread Trading Dashboard", page_icon="📊", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .stMetric > div > div > div > div {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 0.375rem;
    }
    .spread-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load and prepare spread data
# -----------------------------
@st.cache_data
def load_spread_data():
    data = {
        "Trade No": [1,2,3,4,5,6,7,8,9,10,11,12],
        "Trade Date": ["21/07/2025","21/07/2025","21/07/2025","21/07/2025","28/07/2025","28/07/2025","29/07/2025","29/07/2025","06/08/2025","06/08/2025","13/08/2025","13/08/2025"],
        "Underlying": ["MCL","MCL","MCL","MCL","MCL","MCL","MCL","MCL","MCL","MCL","MCL","MCL"],
        "Tenure": ["CLU25","CLU25","CLV25","CLV25","CLU25","CLU25","CLU25","CLU25","CLX25","CLX25","CLX25","CLX25"],
        "Expiration Date": ["15/08/2025","15/08/2025","17/09/2025","17/09/2025","15/08/2025","15/08/2025","15/08/2025","15/08/2025","16/10/2025","16/10/2025","16/10/2025","16/10/2025"],
        "Days to Expiry": [0,0,30,30,0,0,0,0,59,59,59,59],
        "Expired?": ["Expired","Expired","Active","Active","Expired","Expired","Expired","Expired","Active","Active","Active","Active"],
        "Option Type": ["Put","Put","Put","Put","Call","Call","Call","Call","Put","Put","Put","Put"],
        "Direction": ["Sell","Buy","Sell","Buy","Sell","Buy","Sell","Buy","Sell","Buy","Sell","Buy"],
        "Strike Price": [64,60,64,60,69,72,70,72.5,64,59,62,59],
        "Entry Price": [142,52,277,139,110,55,105,60,400,197,326,202],
        "Entry Price/bbl": [1.42,0.52,2.77,1.39,1.1,0.55,1.05,0.6,4,1.97,3.26,2.02],
        "Settlement Price": [61.98,61.98,62.41,62.41,61.98,61.98,61.98,61.98,61.88,61.88,61.88,61.88],
        "Commission": [2.04,2.04,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.02,1.02],
        "Payoff at Settlement": [88.96,-53.04,274.39,-140.02,108.98,-56.02,103.98,-61.02,396.86,-198.02,324.86,-203.02]
    }
    df = pd.DataFrame(data)
    
    # Convert dates
    for col in ["Trade Date", "Expiration Date"]:
        df[col] = pd.to_datetime(df[col], dayfirst=True)
    
    # Identify spread pairs and calculate spread metrics
    spreads = []
    processed_trades = set()
    
    for i, row in df.iterrows():
        if i in processed_trades:
            continue
            
        # Find matching leg for spread
        matching_legs = df[
            (df['Trade Date'] == row['Trade Date']) & 
            (df['Tenure'] == row['Tenure']) & 
            (df['Option Type'] == row['Option Type']) &
            (df['Direction'] != row['Direction']) &
            (df.index != i) &
            (~df.index.isin(processed_trades))
        ]
        
        if not matching_legs.empty:
            leg2 = matching_legs.iloc[0]
            
            # Determine spread type
            if row['Direction'] == 'Sell':
                short_leg = row
                long_leg = leg2
            else:
                short_leg = leg2
                long_leg = row
            
            if short_leg['Option Type'] == 'Put':
                if short_leg['Strike Price'] > long_leg['Strike Price']:
                    spread_type = 'Put Credit Spread'
                else:
                    spread_type = 'Put Debit Spread'
            else:  # Call
                if short_leg['Strike Price'] < long_leg['Strike Price']:
                    spread_type = 'Call Credit Spread'
                else:
                    spread_type = 'Call Debit Spread'
            
            # Calculate spread metrics
            net_credit = short_leg['Entry Price'] - long_leg['Entry Price']
            max_risk = abs(short_leg['Strike Price'] - long_leg['Strike Price']) * 100 - net_credit
            net_pnl = (short_leg['Payoff at Settlement'] + long_leg['Payoff at Settlement']) - (short_leg['Commission'] + long_leg['Commission'])
            
            spread_data = {
                'Spread_ID': len(spreads) + 1,
                'Trade_Date': row['Trade Date'],
                'Expiration_Date': row['Expiration Date'],
                'Tenure': row['Tenure'],
                'Spread_Type': spread_type,
                'Short_Strike': short_leg['Strike Price'],
                'Long_Strike': long_leg['Strike Price'],
                'Spread_Width': abs(short_leg['Strike Price'] - long_leg['Strike Price']),
                'Net_Credit': net_credit,
                'Max_Risk': max_risk,
                'Settlement_Price': row['Settlement Price'],
                'Net_PnL': net_pnl,
                'Total_Commission': short_leg['Commission'] + long_leg['Commission'],
                'Days_to_Expiry': row['Days to Expiry'],
                'Expired': row['Expired?'],
                'ROI': (net_pnl / max_risk) * 100 if max_risk > 0 else 0,
                'Credit_Captured': (net_credit / net_credit) * 100 if net_credit > 0 else 0,
                'Max_Profit': net_credit,
                'Profit_Margin': (net_pnl / net_credit) * 100 if net_credit != 0 else 0
            }
            
            spreads.append(spread_data)
            processed_trades.add(i)
            processed_trades.add(leg2.name)
    
    return pd.DataFrame(spreads)

spread_df = load_spread_data()

# -----------------------------
# Sidebar Filters for Spreads
# -----------------------------
st.sidebar.header("🔧 Spread Filters & Controls")

# Date range filter
if not spread_df.empty:
    date_range = st.sidebar.date_input(
        "Trade Date Range",
        value=(spread_df["Trade_Date"].min().date(), spread_df["Trade_Date"].max().date()),
        min_value=spread_df["Trade_Date"].min().date(),
        max_value=spread_df["Trade_Date"].max().date()
    )

    # Spread type filter
    spread_types = st.sidebar.multiselect(
        "Filter by Spread Type", 
        spread_df["Spread_Type"].unique(), 
        default=spread_df["Spread_Type"].unique()
    )

    # Tenure filter
    tenures = st.sidebar.multiselect(
        "Filter by Tenure", 
        spread_df["Tenure"].unique(), 
        default=spread_df["Tenure"].unique()
    )

    # Status filter
    status_filter = st.sidebar.multiselect(
        "Filter by Status", 
        spread_df["Expired"].unique(), 
        default=spread_df["Expired"].unique()
    )

    # Spread width filter
    width_range = st.sidebar.slider(
        "Spread Width Range",
        min_value=float(spread_df["Spread_Width"].min()),
        max_value=float(spread_df["Spread_Width"].max()),
        value=(float(spread_df["Spread_Width"].min()), float(spread_df["Spread_Width"].max()))
    )

    # Apply filters
    filtered_spread_df = spread_df[
        (spread_df["Trade_Date"].dt.date >= date_range[0]) & 
        (spread_df["Trade_Date"].dt.date <= date_range[1]) &
        (spread_df["Spread_Type"].isin(spread_types)) &
        (spread_df["Tenure"].isin(tenures)) &
        (spread_df["Expired"].isin(status_filter)) &
        (spread_df["Spread_Width"] >= width_range[0]) &
        (spread_df["Spread_Width"] <= width_range[1])
    ]
else:
    filtered_spread_df = spread_df

# -----------------------------
# Header and Credit Spread KPIs
# -----------------------------
st.title("📊 Credit Spread Trading Dashboard")
st.markdown("### Zero-Capital Spread Trading Performance Analysis")

# Spread trading info box
with st.container():
    st.markdown("""
    <div class="spread-info">
    <h4>🎯 Credit Spread Strategy Overview</h4>
    <p><strong>No Capital Required:</strong> You're collecting premium upfront by selling spreads, with defined max risk per trade. 
    This dashboard focuses on credit collection, risk management, and probability of profit analysis.</p>
    </div>
    """, unsafe_allow_html=True)

if not filtered_spread_df.empty:
    # Enhanced KPI Section for Credit Spreads
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total_pnl = filtered_spread_df['Net_PnL'].sum()
        total_credit = filtered_spread_df['Net_Credit'].sum()
        st.metric(
            "Net P&L", 
            f"${total_pnl:,.2f}",
            delta=f"${total_credit:,.2f} credit collected",
            delta_color="normal"
        )

    with col2:
        win_rate = (filtered_spread_df['Net_PnL'] > 0).mean() * 100
        profitable_spreads = len(filtered_spread_df[filtered_spread_df['Net_PnL'] > 0])
        st.metric(
            "Win Rate", 
            f"{win_rate:.1f}%", 
            delta=f"{profitable_spreads}/{len(filtered_spread_df)} profitable"
        )

    with col3:
        avg_roi = filtered_spread_df['ROI'].mean()
        max_roi = filtered_spread_df['ROI'].max()
        st.metric(
            "Avg ROI", 
            f"{avg_roi:.1f}%", 
            delta=f"Max: {max_roi:.1f}%"
        )

    with col4:
        total_risk = filtered_spread_df['Max_Risk'].sum()
        avg_risk = filtered_spread_df['Max_Risk'].mean()
        st.metric(
            "Total Risk", 
            f"${total_risk:,.2f}", 
            delta=f"Avg: ${avg_risk:.2f}"
        )

    with col5:
        active_spreads = (filtered_spread_df['Expired'] == 'Active').sum()
        expired_spreads = (filtered_spread_df['Expired'] == 'Expired').sum()
        st.metric(
            "Active/Expired", 
            f"{active_spreads}/{expired_spreads}", 
            delta=f"Total: {len(filtered_spread_df)}"
        )

    # -----------------------------
    # Credit Spread Specific Visualizations
    # -----------------------------

    # Row 1: P&L and Credit Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Cumulative P&L vs Credit Collected")
        
        # Create dual-axis chart
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        
        cumulative_pnl = filtered_spread_df.groupby("Trade_Date")["Net_PnL"].sum().cumsum().reset_index()
        cumulative_credit = filtered_spread_df.groupby("Trade_Date")["Net_Credit"].sum().cumsum().reset_index()
        
        fig_dual.add_trace(
            go.Scatter(x=cumulative_pnl["Trade_Date"], y=cumulative_pnl["Net_PnL"],
                      name="Cumulative P&L", line=dict(color='green', width=3)),
            secondary_y=False
        )
        
        fig_dual.add_trace(
            go.Scatter(x=cumulative_credit["Trade_Date"], y=cumulative_credit["Net_Credit"],
                      name="Cumulative Credit", line=dict(color='blue', width=2, dash='dash')),
            secondary_y=True
        )
        
        fig_dual.update_yaxes(title_text="Cumulative P&L ($)", secondary_y=False)
        fig_dual.update_yaxes(title_text="Cumulative Credit ($)", secondary_y=True)
        fig_dual.update_xaxes(title_text="Date")
        fig_dual.update_layout(title="Portfolio Performance vs Credit Collection")
        
        st.plotly_chart(fig_dual, use_container_width=True)

    with col2:
        st.subheader("🎯 Spread Performance Matrix")
        
        # ROI vs Max Risk scatter
        fig_scatter = px.scatter(
            filtered_spread_df,
            x="Max_Risk",
            y="ROI",
            color="Spread_Type",
            size="Net_Credit",
            hover_data=['Spread_Width', 'Days_to_Expiry', 'Net_PnL'],
            title="ROI vs Risk (Bubble size = Credit collected)"
        )
        
        # Add break-even line at ROI = 0
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="red", 
                             annotation_text="Break-even")
        
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Row 2: Spread Strategy Analysis
    st.subheader("🎯 Spread Strategy Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Spread type performance
        spread_perf = filtered_spread_df.groupby('Spread_Type').agg({
            'Net_PnL': 'sum',
            'Net_Credit': 'sum',
            'Max_Risk': 'sum',
            'Spread_ID': 'count'
        }).reset_index()
        spread_perf['Profit_Margin'] = (spread_perf['Net_PnL'] / spread_perf['Net_Credit']) * 100

        fig_margin = px.bar(
            spread_perf,
            x='Spread_Type',
            y='Profit_Margin',
            title="Profit Margin by Spread Type (%)",
            color='Profit_Margin',
            color_continuous_scale='RdYlGn'
        )
        fig_margin.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_margin, use_container_width=True)

    with col2:
        # Credit collection efficiency
        fig_credit_eff = px.bar(
            spread_perf,
            x='Spread_Type',
            y='Net_Credit',
            title="Total Credit Collected by Strategy",
            color='Net_Credit',
            color_continuous_scale='blues'
        )
        fig_credit_eff.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_credit_eff, use_container_width=True)

    with col3:
        # Risk-adjusted returns
        spread_perf['Risk_Adj_Return'] = spread_perf['Net_PnL'] / spread_perf['Max_Risk']
        
        fig_risk_adj = px.bar(
            spread_perf,
            x='Spread_Type',
            y='Risk_Adj_Return',
            title="Risk-Adjusted Returns",
            color='Risk_Adj_Return',
            color_continuous_scale='viridis'
        )
        fig_risk_adj.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_risk_adj, use_container_width=True)

    # Row 3: Risk Management Analysis
    st.subheader("⚠️ Risk Management & Probability Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Spread width vs profitability
        fig_width = px.box(
            filtered_spread_df,
            x='Spread_Width',
            y='Net_PnL',
            title="P&L Distribution by Spread Width",
            color='Spread_Type'
        )
        fig_width.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_width, use_container_width=True)

    with col2:
        # Days to expiry analysis
        fig_dte = px.scatter(
            filtered_spread_df,
            x='Days_to_Expiry',
            y='Net_PnL',
            color='Spread_Type',
            size='Net_Credit',
            title="P&L vs Days to Expiry at Trade Entry",
            hover_data=['Spread_Width', 'Settlement_Price']
        )
        fig_dte.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_dte, use_container_width=True)

    # Row 4: Credit Spread Analytics
    st.subheader("📊 Advanced Spread Analytics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Credit Collection Metrics**")
        credit_metrics = pd.DataFrame({
            'Metric': [
                'Total Credit Collected',
                'Average Credit per Spread',
                'Credit Retention Rate',
                'Best Credit Collection',
                'Credit Collection Efficiency'
            ],
            'Value': [
                f"${filtered_spread_df['Net_Credit'].sum():.2f}",
                f"${filtered_spread_df['Net_Credit'].mean():.2f}",
                f"{((filtered_spread_df['Net_PnL'] / filtered_spread_df['Net_Credit']).mean() * 100):.1f}%",
                f"${filtered_spread_df['Net_Credit'].max():.2f}",
                f"{(filtered_spread_df['Net_Credit'].sum() / filtered_spread_df['Max_Risk'].sum() * 100):.1f}%"
            ]
        })
        st.dataframe(credit_metrics, use_container_width=True)

    with col2:
        st.write("**Risk Management Metrics**")
        risk_metrics = pd.DataFrame({
            'Metric': [
                'Total Risk Capital',
                'Average Risk per Spread',
                'Max Single Risk',
                'Risk Utilization',
                'Win/Risk Ratio'
            ],
            'Value': [
                f"${filtered_spread_df['Max_Risk'].sum():.2f}",
                f"${filtered_spread_df['Max_Risk'].mean():.2f}",
                f"${filtered_spread_df['Max_Risk'].max():.2f}",
                f"{(filtered_spread_df[filtered_spread_df['Expired'] == 'Active']['Max_Risk'].sum() / filtered_spread_df['Max_Risk'].sum() * 100):.1f}%",
                f"{(filtered_spread_df[filtered_spread_df['Net_PnL'] > 0]['Net_PnL'].sum() / filtered_spread_df['Max_Risk'].sum()):.3f}"
            ]
        })
        st.dataframe(risk_metrics, use_container_width=True)

    with col3:
        st.write("**Profitability Analysis**")
        profit_metrics = pd.DataFrame({
            'Metric': [
                'Probability of Profit',
                'Average Winner',
                'Average Loser',
                'Profit Factor',
                'Expectancy per Trade'
            ],
            'Value': [
                f"{((filtered_spread_df['Net_PnL'] > 0).sum() / len(filtered_spread_df) * 100):.1f}%",
                f"${filtered_spread_df[filtered_spread_df['Net_PnL'] > 0]['Net_PnL'].mean():.2f}",
                f"${filtered_spread_df[filtered_spread_df['Net_PnL'] < 0]['Net_PnL'].mean():.2f}",
                f"{abs(filtered_spread_df[filtered_spread_df['Net_PnL'] > 0]['Net_PnL'].sum() / filtered_spread_df[filtered_spread_df['Net_PnL'] < 0]['Net_PnL'].sum()):.2f}" if filtered_spread_df[filtered_spread_df['Net_PnL'] < 0]['Net_PnL'].sum() != 0 else "∞",
                f"${filtered_spread_df['Net_PnL'].mean():.2f}"
            ]
        })
        st.dataframe(profit_metrics, use_container_width=True)

    # -----------------------------
    # Enhanced Spread Data Table
    # -----------------------------
    st.subheader("📑 Detailed Spread Analysis")

    # Add spread-specific sorting options
    sort_column = st.selectbox(
        "Sort by:", 
        options=['Net_PnL', 'ROI', 'Net_Credit', 'Max_Risk', 'Spread_Width', 'Trade_Date'], 
        index=0
    )
    sort_order = st.radio("Order:", ["Descending", "Ascending"], horizontal=True)

    sorted_spread_df = filtered_spread_df.sort_values(
        by=sort_column, 
        ascending=(sort_order == "Ascending")
    )

    # Format and display spread dataframe
    display_df = sorted_spread_df.copy()
    
    # Format currency columns
    for col in ['Net_Credit', 'Max_Risk', 'Net_PnL', 'Max_Profit']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
    
    # Format percentage columns
    for col in ['ROI', 'Profit_Margin']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%")

    st.dataframe(display_df, use_container_width=True)

    # Export functionality
    if st.button("📥 Export Spread Data to CSV"):
        csv = sorted_spread_df.to_csv(index=False)
        st.download_button(
            label="Download Spread Analysis CSV",
            data=csv,
            file_name=f"credit_spreads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    st.warning("No spread data found. Please check your trade pairing logic.")

# Footer
st.markdown("---")
st.markdown("**Credit Spread Dashboard** - Designed for zero-capital spread trading strategies")
st.markdown("*Metrics calculated based on spread pairs. ROI = Net P&L / Max Risk per spread.*")
st.markdown("**Dashboard last updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))