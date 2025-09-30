import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configure page
st.set_page_config(page_title="Oil Hedging Simulator", layout="wide", initial_sidebar_state="expanded")
st.title("🛢️ Oil Hedging Simulator Dashboard")

# Initialize session state
if "production_df" not in st.session_state:
    calendar_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    st.session_state["production_df"] = pd.DataFrame({
        "Month": calendar_months,
        "Volume": [100_000] * 12,
        "Hedge Ratio (%)": [50] * 12
    })

if "hedge_strategies" not in st.session_state:
    st.session_state["hedge_strategies"] = []

# Helper functions
def calculate_hedge_payoff(hedge_type, price, strike=None, floor=None, cap=None, premium=0):
    """Calculate hedge payoff based on instrument type"""
    if hedge_type == "Swap":
        return strike - price
    elif hedge_type == "Put":
        return max(strike - price, 0) - premium
    elif hedge_type == "Call":
        return max(price - strike, 0) - premium
    elif hedge_type == "Collar":
        put_payoff = max(floor - price, 0)
        call_payoff = max(price - cap, 0)
        return put_payoff - call_payoff - premium
    elif hedge_type == "3-Way Collar":
        put_payoff = max(floor - price, 0)
        call_payoff = max(price - cap, 0)
        short_put_payoff = max(price - strike, 0)  # Short put at strike
        return put_payoff - call_payoff - short_put_payoff - premium
    return 0

def get_strategy_months(start_month, end_month):
    """Get list of months between start and end"""
    calendar_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    start_idx = calendar_months.index(start_month)
    end_idx = calendar_months.index(end_month)
    
    if end_idx >= start_idx:
        return calendar_months[start_idx:end_idx + 1]
    else:  # Wraps around year
        return calendar_months[start_idx:] + calendar_months[:end_idx + 1]

# Sidebar for global settings
with st.sidebar:
    st.header("🎛️ Global Settings")

    # Global market parameters
    current_oil_price = st.slider("Current Oil Price ($/bbl)", 30.0, 150.0, 75.0, 1.0)
    volatility = st.slider("Implied Volatility (%)", 10.0, 100.0, 25.0, 1.0)
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 3.0, 0.1)

    st.divider()
    st.header("📈 Quick Analytics")

    # Check if entitlement_volume and hedge_ratio exist
    if "entitlement_volume" in st.session_state and "hedge_ratio" in st.session_state:
        # Pull data from session state
        entitlement_volume = st.session_state["entitlement_volume"]
        hedge_ratio = st.session_state["hedge_ratio"]

        annual_hedging_volume = entitlement_volume * hedge_ratio / 100
        annual_exposed_volume = entitlement_volume - annual_hedging_volume


        st.metric("Entitlement Volume", f"{entitlement_volume:,.0f} bbl")
        st.metric("Hedge Ratio", f"{hedge_ratio:.1f}%")
        st.metric("Annual Hedged Volume", f"{annual_hedging_volume:,.0f} bbl")

    else:
        st.info("Adjust values in the Production tab to see quick analytics.")


# Main dashboard tabs
tab_prod, tab_hedge, tab_results, tab_stress, tab_analytics = st.tabs([
    "🏭 Production Data", 
    "🛡️ Hedging Strategy", 
    "📊 Results", 
    "⚠️ Stress Testing",
    "📈 Analytics"
])

# Production Data Tab
with tab_prod:
    st.markdown("## 🛢️ Production Hedging Overview")
    st.markdown("Set your entitlement volume and hedge ratio to visualize monthly hedged vs. exposed volumes.")

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("### ⚙️ Inputs")

        entitlement_volume = st.number_input(
            "Entitlement Volume (Annual, bbl)", 
            value=602_250_000.00, 
            step=1_000_000.0, 
            min_value=0.0, 
            format="%.2f"
        )

        hedge_ratio = st.slider("Hedge Ratio (%)", 0, 100, value=20)

        # Calculations
        annual_hedging_volume = entitlement_volume * hedge_ratio / 100
        monthly_hedging_volume = annual_hedging_volume / 12
        monthly_exposed_volume = (entitlement_volume - annual_hedging_volume) / 12

        st.markdown("### 📊 Summary")
        st.metric("Annual Hedged Volume", f"{annual_hedging_volume:,.2f} bbl")
        st.metric("Monthly Hedged Volume", f"{monthly_hedging_volume:,.2f} bbl")
        st.metric("Monthly Exposed Volume", f"{monthly_exposed_volume:,.2f} bbl")

    with right_col:
        # Data preparation
        calendar_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        production_df = pd.DataFrame({
            "Month": calendar_months,
            "Hedged Volume": [monthly_hedging_volume] * 12,
            "Exposed Volume": [monthly_exposed_volume] * 12
        })

        st.markdown("### 📉 Hedging Distribution Chart")

        fig = px.bar(
            production_df,
            x="Month",
            y=["Hedged Volume", "Exposed Volume"],
            title="Monthly Hedged vs. Exposed Volume",
            labels={"value": "Volume (bbl)", "variable": "Volume Type"},
            barmode="stack",
            color_discrete_map={
                "Hedged Volume": "#1f77b4", 
                "Exposed Volume": "#ff7f0e"
            }
        )
        fig.update_layout( legend_title_text="", height=400)

        st.plotly_chart(fig, use_container_width=True)
    
    # Store values in session state when user clicks "Save & Refresh Sidebar"
    if st.button("💾 Save & Refresh Sidebar"):
        st.session_state["entitlement_volume"] = entitlement_volume
        st.session_state["hedge_ratio"] = hedge_ratio
        st.rerun()  # Refresh the app to trigger sidebar update




# Hedging Strategy Tab
with tab_hedge:
    st.markdown("## 🛡️ Hedging Strategies")
    st.markdown("Configure option strategies to manage downside risk and optimize upside potential.")

    # Strategy Entry
    with st.expander("➕ Add New Strategy", expanded=True):
        strategy_type = st.selectbox("Select Strategy Type", ["Put", "Put Spread", "Collar"], key="strategy_select")

        with st.form(f"form_{strategy_type}"):
            col1, col2 = st.columns(2)

            if strategy_type == "Put":
                with col1:
                    strike_price = st.number_input("Strike Price ($/bbl)", min_value=0.0, value=70.0, step=1.0)
                with col2:
                    premium = st.number_input("Premium ($/bbl)", min_value=0.0, value=3.0, step=0.5)

            elif strategy_type == "Put Spread":
                with col1:
                    lower_strike = st.number_input("Long Put Strike ($/bbl)", min_value=0.0, value=65.0, step=1.0)
                with col2:
                    upper_strike = st.number_input("Short Put Strike ($/bbl)", min_value=0.0, value=55.0, step=1.0)
                premium = st.number_input("Net Premium ($/bbl)", min_value=0.0, value=2.0, step=0.5)

            elif strategy_type == "Collar":
                with col1:
                    put_strike = st.number_input("Put Strike ($/bbl)", min_value=0.0, value=65.0, step=1.0)
                with col2:
                    call_strike = st.number_input("Call Strike ($/bbl)", min_value=0.0, value=85.0, step=1.0)
                premium = st.number_input("Net Premium ($/bbl)", min_value=-10.0, value=0.0, step=0.5)

            submitted = st.form_submit_button("✅ Add Strategy")

            if submitted:
                if "strategies" not in st.session_state:
                    st.session_state["strategies"] = []

                strategy_data = {
                    "type": strategy_type,
                    "premium": premium
                }
                if strategy_type == "Put":
                    strategy_data.update({"strike": strike_price})
                elif strategy_type == "Put Spread":
                    strategy_data.update({"long_put": lower_strike, "short_put": upper_strike})
                elif strategy_type == "Collar":
                    strategy_data.update({"put": put_strike, "call": call_strike})

                st.session_state["strategies"].append(strategy_data)
                st.success(f"{strategy_type} strategy added successfully!")
                st.rerun()

    # Strategy Display and Payoff Visualization
    if "strategies" in st.session_state and st.session_state["strategies"]:
        st.markdown("### 📋 Current Strategies")
        import numpy as np
        import plotly.graph_objects as go

        for i, strat in enumerate(st.session_state["strategies"]):
            with st.expander(f"Strategy #{i + 1}: {strat['type']}"):
                st.json(strat)

                prices = np.linspace(30, 150, 200)
                payoff = np.zeros_like(prices)

                if strat["type"] == "Put":
                    strike = strat.get("strike")
                    premium = strat.get("premium", 0)
                    if strike is not None:
                        payoff = np.maximum(strike - prices, 0) - premium

                elif strat["type"] == "Put Spread":
                    long_put = strat.get("long_put")
                    short_put = strat.get("short_put")
                    premium = strat.get("premium", 0)
                    if long_put is not None and short_put is not None:
                        payoff = np.maximum(long_put - prices, 0) - np.maximum(short_put - prices, 0) - premium

                elif strat["type"] == "Collar":
                    put = strat.get("put")
                    call = strat.get("call")
                    premium = strat.get("premium", 0)
                    if put is not None and call is not None:
                        payoff = np.maximum(put - prices, 0) - np.maximum(prices - call, 0) - premium

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=prices, y=payoff, mode='lines', name='Payoff'))
                fig.update_layout(
                    title=f"Payoff Diagram - {strat['type']}",
                    xaxis_title="Oil Price ($/bbl)",
                    yaxis_title="Profit / Loss ($/bbl)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

        # Button to clear all strategies
        if st.button("🗑️ Clear All Strategies"):
            st.session_state.pop("strategies")
            st.rerun()

    # Optional: Reset button to clear old malformed data
    if st.button("🧹 Reset All Strategy Data (Force Clear)"):
        st.session_state.pop("strategies", None)
        st.rerun()

# Results Tab
with tab_results:
    st.header("📊 Monthly Results Analysis")
    
    if not st.session_state["hedge_strategies"]:
        st.warning("⚠️ Please define at least one hedging strategy to see results.")
    else:
        # Price scenario input
        col1, col2, col3 = st.columns(3)
        with col1:
            scenario_price = st.slider("Market Price Scenario ($/bbl)", 30.0, 150.0, current_oil_price, 1.0)
        with col2:
            show_unhedged = st.checkbox("Show Unhedged Comparison", value=True)
        with col3:
            show_details = st.checkbox("Show Strategy Details", value=False)

        # Calculate monthly results
        results_data = []
        for _, month_data in st.session_state["production_df"].iterrows():
            month = month_data["Month"]
            volume = month_data["Volume"]
            
            month_result = {
                "Month": month,
                "Production Volume": volume,
                "Hedged Volume": 0,
                "Unhedged Volume": volume,
                "Hedged Revenue": 0,
                "Unhedged Revenue": volume * scenario_price,
                "Total Revenue": volume * scenario_price,
                "Hedge P&L": 0
            }
            
            # Apply applicable strategies
            for strategy in st.session_state["hedge_strategies"]:
                strategy_months = get_strategy_months(strategy["Start Month"], strategy["End Month"])
                
                if month in strategy_months:
                    # Calculate hedge volume (percentage of production)
                    hedge_volume = volume * (month_data["Hedge Ratio (%)"] / 100)
                    
                    # Calculate hedge payoff
                    payoff = calculate_hedge_payoff(
                        strategy["Type"],
                        scenario_price,
                        strike=strategy.get("Strike"),
                        floor=strategy.get("Floor"),
                        cap=strategy.get("Cap"),
                        premium=strategy.get("Premium", 0)
                    )
                    
                    hedge_pnl = hedge_volume * payoff
                    
                    month_result["Hedged Volume"] += hedge_volume
                    month_result["Hedge P&L"] += hedge_pnl
            
            # Update unhedged volume and total revenue
            month_result["Unhedged Volume"] = max(0, volume - month_result["Hedged Volume"])
            month_result["Hedged Revenue"] = month_result["Hedged Volume"] * scenario_price
            month_result["Total Revenue"] = month_result["Hedged Revenue"] + month_result["Unhedged Revenue"] + month_result["Hedge P&L"]
            
            results_data.append(month_result)
        
        results_df = pd.DataFrame(results_data)
        
        # Display results table
        st.markdown("### 📋 Monthly Results Summary")
        display_df = results_df.copy()
        
        # Format currency columns
        currency_cols = ["Hedged Revenue", "Unhedged Revenue", "Total Revenue", "Hedge P&L"]
        for col in currency_cols:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
        
        # Format volume columns
        volume_cols = ["Production Volume", "Hedged Volume", "Unhedged Volume"]
        for col in volume_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Visualizations
        st.markdown("### 📈 Visual Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Volume breakdown
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                name="Hedged Volume",
                x=results_df["Month"],
                y=results_df["Hedged Volume"],
                marker_color="lightblue"
            ))
            fig_volume.add_trace(go.Bar(
                name="Unhedged Volume",
                x=results_df["Month"],
                y=results_df["Unhedged Volume"],
                marker_color="lightcoral"
            ))
            fig_volume.update_layout(
                title="Monthly Volume Breakdown",
                xaxis_title="Month",
                yaxis_title="Volume (bbl)",
                barmode="stack"
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with col2:
            # Revenue comparison
            fig_revenue = go.Figure()
            fig_revenue.add_trace(go.Scatter(
                name="Total Revenue",
                x=results_df["Month"],
                y=results_df["Total Revenue"],
                mode="lines+markers",
                line=dict(color="green", width=3)
            ))
            if show_unhedged:
                unhedged_revenue = results_df["Production Volume"] * scenario_price
                fig_revenue.add_trace(go.Scatter(
                    name="Unhedged Revenue",
                    x=results_df["Month"],
                    y=unhedged_revenue,
                    mode="lines+markers",
                    line=dict(color="red", dash="dash")
                ))
            fig_revenue.update_layout(
                title="Monthly Revenue Comparison",
                xaxis_title="Month",
                yaxis_title="Revenue ($)"
            )
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Hedge P&L chart
        fig_pnl = px.bar(
            results_df,
            x="Month",
            y="Hedge P&L",
            title="Monthly Hedge P&L",
            color="Hedge P&L",
            color_continuous_scale=["red", "white", "green"]
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
        
        # Summary metrics
        st.markdown("### 📊 Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        total_revenue = results_df["Total Revenue"].sum()
        total_hedge_pnl = results_df["Hedge P&L"].sum()
        total_production = results_df["Production Volume"].sum()
        unhedged_total = total_production * scenario_price
        
        with col1:
            st.metric("Total Revenue", f"${total_revenue:,.0f}")
        with col2:
            st.metric("Total Hedge P&L", f"${total_hedge_pnl:,.0f}")
        with col3:
            st.metric("Unhedged Revenue", f"${unhedged_total:,.0f}")
        with col4:
            improvement = total_revenue - unhedged_total
            st.metric("Improvement vs Unhedged", f"${improvement:,.0f}")

# Stress Testing Tab
with tab_stress:
    st.header("⚠️ Stress Testing")
    
    if not st.session_state["hedge_strategies"]:
        st.warning("⚠️ Please define at least one hedging strategy to perform stress testing.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Test Parameters")
            price_min = st.number_input("Min Price ($/bbl)", value=30.0, step=5.0)
            price_max = st.number_input("Max Price ($/bbl)", value=120.0, step=5.0)
            price_steps = st.slider("Price Steps", 10, 100, 50)
        
        # Generate price scenarios
        price_range = np.linspace(price_min, price_max, price_steps)
        stress_results = []
        
        for test_price in price_range:
            total_revenue = 0
            total_hedge_pnl = 0
            total_volume = 0
            
            for _, month_data in st.session_state["production_df"].iterrows():
                month = month_data["Month"]
                volume = month_data["Volume"]
                total_volume += volume
                
                # Base revenue
                month_revenue = volume * test_price
                month_hedge_pnl = 0
                
                # Apply strategies
                for strategy in st.session_state["hedge_strategies"]:
                    strategy_months = get_strategy_months(strategy["Start Month"], strategy["End Month"])
                    
                    if month in strategy_months:
                        hedge_volume = volume * (month_data["Hedge Ratio (%)"] / 100)
                        payoff = calculate_hedge_payoff(
                            strategy["Type"],
                            test_price,
                            strike=strategy.get("Strike"),
                            floor=strategy.get("Floor"),
                            cap=strategy.get("Cap"),
                            premium=strategy.get("Premium", 0)
                        )
                        month_hedge_pnl += hedge_volume * payoff
                
                total_revenue += month_revenue + month_hedge_pnl
                total_hedge_pnl += month_hedge_pnl
            
            stress_results.append({
                "Price": test_price,
                "Total Revenue": total_revenue,
                "Hedge P&L": total_hedge_pnl,
                "Unhedged Revenue": total_volume * test_price,
                "Revenue per Barrel": total_revenue / total_volume if total_volume > 0 else 0
            })
        
        stress_df = pd.DataFrame(stress_results)
        
        with col2:
            # Stress test chart
            fig_stress = go.Figure()
            fig_stress.add_trace(go.Scatter(
                name="Hedged Portfolio",
                x=stress_df["Price"],
                y=stress_df["Total Revenue"],
                mode="lines",
                line=dict(color="blue", width=3)
            ))
            fig_stress.add_trace(go.Scatter(
                name="Unhedged",
                x=stress_df["Price"],
                y=stress_df["Unhedged Revenue"],
                mode="lines",
                line=dict(color="red", dash="dash", width=2)
            ))
            fig_stress.update_layout(
                title="Revenue vs Oil Price",
                xaxis_title="Oil Price ($/bbl)",
                yaxis_title="Total Annual Revenue ($)"
            )
            st.plotly_chart(fig_stress, use_container_width=True)
        
        # Additional stress test metrics
        st.markdown("### 📊 Stress Test Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hedge effectiveness
            fig_hedge_pnl = px.line(
                stress_df,
                x="Price",
                y="Hedge P&L",
                title="Hedge P&L vs Oil Price"
            )
            fig_hedge_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_hedge_pnl, use_container_width=True)
        
        with col2:
            # Revenue per barrel
            fig_per_barrel = px.line(
                stress_df,
                x="Price",
                y="Revenue per Barrel",
                title="Effective Price per Barrel"
            )
            # Add line showing unhedged price
            fig_per_barrel.add_trace(go.Scatter(
                x=stress_df["Price"],
                y=stress_df["Price"],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Unhedged Price"
            ))
            st.plotly_chart(fig_per_barrel, use_container_width=True)
        
        # Risk metrics
        st.markdown("### 🎯 Risk Metrics")
        
        # Calculate key metrics
        current_idx = np.argmin(np.abs(stress_df["Price"] - current_oil_price))
        downside_protection = stress_df.iloc[0]["Total Revenue"] - stress_df.iloc[0]["Unhedged Revenue"]
        upside_participation = (stress_df.iloc[-1]["Total Revenue"] - stress_df.iloc[-1]["Unhedged Revenue"]) / stress_df.iloc[-1]["Unhedged Revenue"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Scenario Revenue", f"${stress_df.iloc[current_idx]['Total Revenue']:,.0f}")
        with col2:
            st.metric("Downside Protection", f"${downside_protection:,.0f}")
        with col3:
            st.metric("Upside Participation", f"{upside_participation:.1%}")
        with col4:
            max_hedge_benefit = stress_df["Hedge P&L"].max()
            st.metric("Max Hedge Benefit", f"${max_hedge_benefit:,.0f}")

# Analytics Tab
with tab_analytics:
    st.header("📈 Portfolio Analytics")
    
    if not st.session_state["hedge_strategies"]:
        st.warning("⚠️ Please define at least one hedging strategy to see analytics.")
    else:
        # Portfolio composition
        st.markdown("### 🏗️ Portfolio Composition")
        
        # Calculate strategy exposure by month
        exposure_data = []
        for _, month_data in st.session_state["production_df"].iterrows():
            month = month_data["Month"]
            volume = month_data["Volume"]
            hedge_ratio = month_data["Hedge Ratio (%)"] / 100
            
            for strategy in st.session_state["hedge_strategies"]:
                strategy_months = get_strategy_months(strategy["Start Month"], strategy["End Month"])
                if month in strategy_months:
                    exposure_data.append({
                        "Month": month,
                        "Strategy": strategy["Name"],
                        "Type": strategy["Type"],
                        "Volume": volume * hedge_ratio,
                        "Percentage": hedge_ratio * 100
                    })
        
        if exposure_data:
            exposure_df = pd.DataFrame(exposure_data)
            
            # Strategy type breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                type_summary = exposure_df.groupby("Type")["Volume"].sum().reset_index()
                fig_pie = px.pie(
                    type_summary,
                    values="Volume",
                    names="Type",
                    title="Hedge Volume by Instrument Type"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Monthly exposure timeline
                monthly_exposure = exposure_df.groupby(["Month", "Type"])["Volume"].sum().reset_index()
                fig_timeline = px.bar(
                    monthly_exposure,
                    x="Month",
                    y="Volume",
                    color="Type",
                    title="Monthly Hedge Exposure by Type"
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Strategy performance metrics
        st.markdown("### 📊 Strategy Performance Metrics")
        
        performance_data = []
        price_scenarios = [50, 60, 70, 80, 90, 100]
        
        for strategy in st.session_state["hedge_strategies"]:
            strategy_performance = {"Strategy": strategy["Name"], "Type": strategy["Type"]}
            
            for price in price_scenarios:
                payoff = calculate_hedge_payoff(
                    strategy["Type"],
                    price,
                    strike=strategy.get("Strike"),
                    floor=strategy.get("Floor"),
                    cap=strategy.get("Cap"),
                    premium=strategy.get("Premium", 0)
                )
                strategy_performance[f"${price}"] = payoff
            
            performance_data.append(strategy_performance)
        
        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            
            # Format the performance table
            display_performance = performance_df.copy()
            price_cols = [col for col in display_performance.columns if col.startswith('$')]
            for col in price_cols:
                display_performance[col] = display_performance[col].apply(lambda x: f"${x:.2f}")
            
            st.dataframe(display_performance, use_container_width=True, hide_index=True)
            
        
        # Risk analysis
        st.markdown("### ⚖️ Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Portfolio Characteristics")
            
            # Calculate portfolio metrics
            total_production = st.session_state["production_df"]["Volume"].sum()
            total_hedged = 0
            
            for _, month_data in st.session_state["production_df"].iterrows():
                month = month_data["Month"]
                volume = month_data["Volume"]
                
                for strategy in st.session_state["hedge_strategies"]:
                    strategy_months = get_strategy_months(strategy["Start Month"], strategy["End Month"])
                    if month in strategy_months:
                        total_hedged += volume * (month_data["Hedge Ratio (%)"] / 100)
            
            hedge_coverage = (total_hedged / total_production) * 100 if total_production > 0 else 0
            
            st.metric("Total Production", f"{total_production:,.0f} bbl")
            st.metric("Total Hedged Volume", f"{total_hedged:,.0f} bbl")
            st.metric("Hedge Coverage", f"{hedge_coverage:.1f}%")
            
            # Strategy diversification
            strategy_types = [s["Type"] for s in st.session_state["hedge_strategies"]]
            unique_types = len(set(strategy_types))
            st.metric("Strategy Diversification", f"{unique_types} instrument types")
        
        with col2:
            st.markdown("#### Price Sensitivity Analysis")
            
            # Calculate price sensitivity
            base_price = current_oil_price
            price_shock = 10  # $10 price change
            
            base_revenue = 0
            shocked_revenue_up = 0
            shocked_revenue_down = 0
            
            for _, month_data in st.session_state["production_df"].iterrows():
                month = month_data["Month"]
                volume = month_data["Volume"]
                
                # Base case
                month_base = volume * base_price
                month_up = volume * (base_price + price_shock)
                month_down = volume * (base_price - price_shock)
                
                # Add hedge effects
                for strategy in st.session_state["hedge_strategies"]:
                    strategy_months = get_strategy_months(strategy["Start Month"], strategy["End Month"])
                    if month in strategy_months:
                        hedge_volume = volume * (month_data["Hedge Ratio (%)"] / 100)
                        
                        base_payoff = calculate_hedge_payoff(
                            strategy["Type"], base_price,
                            strike=strategy.get("Strike"),
                            floor=strategy.get("Floor"),
                            cap=strategy.get("Cap"),
                            premium=strategy.get("Premium", 0)
                        )
                        
                        up_payoff = calculate_hedge_payoff(
                            strategy["Type"], base_price + price_shock,
                            strike=strategy.get("Strike"),
                            floor=strategy.get("Floor"),
                            cap=strategy.get("Cap"),
                            premium=strategy.get("Premium", 0)
                        )
                        
                        down_payoff = calculate_hedge_payoff(
                            strategy["Type"], base_price - price_shock,
                            strike=strategy.get("Strike"),
                            floor=strategy.get("Floor"),
                            cap=strategy.get("Cap"),
                            premium=strategy.get("Premium", 0)
                        )
                        
                        month_base += hedge_volume * base_payoff
                        month_up += hedge_volume * up_payoff
                        month_down += hedge_volume * down_payoff
                
                base_revenue += month_base
                shocked_revenue_up += month_up
                shocked_revenue_down += month_down
            
            upside_change = shocked_revenue_up - base_revenue
            downside_change = shocked_revenue_down - base_revenue
            
            st.metric("Base Case Revenue", f"${base_revenue:,.0f}")
            st.metric(f"+${price_shock} Price Impact", f"${upside_change:,.0f}")
            st.metric(f"-${price_shock} Price Impact", f"${downside_change:,.0f}")
            
            # Calculate hedge effectiveness
            unhedged_up = total_production * (base_price + price_shock)
            unhedged_down = total_production * (base_price - price_shock)
            unhedged_base = total_production * base_price
            
            upside_protection = 1 - (upside_change / (unhedged_up - unhedged_base)) if (unhedged_up - unhedged_base) != 0 else 0
            downside_protection = 1 - (abs(downside_change) / abs(unhedged_down - unhedged_base)) if (unhedged_down - unhedged_base) != 0 else 0
            
            st.metric("Upside Participation", f"{(1-upside_protection)*100:.1f}%")
            st.metric("Downside Protection", f"{downside_protection*100:.1f}%")
        
        # Monte Carlo simulation section
        st.markdown("### 🎲 Monte Carlo Simulation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Simulation Parameters")
            num_simulations = st.slider("Number of Simulations", 100, 2000, 1000, 100)
            volatility_annual = volatility / 100
            time_horizon = 1.0  # 1 year
            
            if st.button("🚀 Run Simulation", type="primary"):
                # Run Monte Carlo simulation
                np.random.seed(42)  # For reproducible results
                
                simulation_results = []
                
                for sim in range(num_simulations):
                    # Generate random price path (geometric Brownian motion)
                    dt = time_horizon / 12  # Monthly steps
                    price_path = [current_oil_price]
                    
                    for month in range(12):
                        drift = (risk_free_rate / 100 - 0.5 * volatility_annual**2) * dt
                        shock = volatility_annual * np.sqrt(dt) * np.random.normal()
                        next_price = price_path[-1] * np.exp(drift + shock)
                        price_path.append(next_price)
                    
                    # Calculate revenue for this price path
                    total_revenue = 0
                    
                    for month_idx, month_data in st.session_state["production_df"].iterrows():
                        month = month_data["Month"]
                        volume = month_data["Volume"]
                        price = price_path[month_idx + 1]  # +1 because path starts with current price
                        
                        month_revenue = volume * price
                        
                        # Add hedge effects
                        for strategy in st.session_state["hedge_strategies"]:
                            strategy_months = get_strategy_months(strategy["Start Month"], strategy["End Month"])
                            if month in strategy_months:
                                hedge_volume = volume * (month_data["Hedge Ratio (%)"] / 100)
                                payoff = calculate_hedge_payoff(
                                    strategy["Type"], price,
                                    strike=strategy.get("Strike"),
                                    floor=strategy.get("Floor"),
                                    cap=strategy.get("Cap"),
                                    premium=strategy.get("Premium", 0)
                                )
                                month_revenue += hedge_volume * payoff
                        
                        total_revenue += month_revenue
                    
                    simulation_results.append({
                        "Simulation": sim + 1,
                        "Final_Price": price_path[-1],
                        "Total_Revenue": total_revenue,
                        "Avg_Price": np.mean(price_path[1:])
                    })
                
                sim_df = pd.DataFrame(simulation_results)
                st.session_state["simulation_results"] = sim_df
        
        with col2:
            if "simulation_results" in st.session_state:
                sim_df = st.session_state["simulation_results"]
                
                # Revenue distribution
                fig_dist = px.histogram(
                    sim_df,
                    x="Total_Revenue",
                    nbins=50,
                    title="Revenue Distribution (Monte Carlo)",
                    labels={"Total_Revenue": "Total Revenue ($)"}
                )
                fig_dist.add_vline(
                    x=sim_df["Total_Revenue"].mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Mean"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Statistics
                st.markdown("#### Simulation Statistics")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Mean Revenue", f"${sim_df['Total_Revenue'].mean():,.0f}")
                    st.metric("Std Deviation", f"${sim_df['Total_Revenue'].std():,.0f}")
                
                with col_b:
                    st.metric("5th Percentile", f"${sim_df['Total_Revenue'].quantile(0.05):,.0f}")
                    st.metric("95th Percentile", f"${sim_df['Total_Revenue'].quantile(0.95):,.0f}")
                
                with col_c:
                    st.metric("Value at Risk (5%)", f"${sim_df['Total_Revenue'].mean() - sim_df['Total_Revenue'].quantile(0.05):,.0f}")
                    st.metric("Max Revenue", f"${sim_df['Total_Revenue'].max():,.0f}")
            else:
                st.info("Click 'Run Simulation' to see Monte Carlo results")


            for col in price_cols:
                display_performance[col] = display_performance[col].apply(
                    lambda x: f"${float(str(x).replace('$', '').replace(',', '')):.2f}"
                    if pd.notnull(x) else x
                )
            
            st.dataframe(display_performance, use_container_width=True, hide_index=True)
            
        
        # Risk analysis
        st.markdown("### ⚖️ Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Portfolio Characteristics")
            
            # Calculate portfolio metrics
            total_production = st.session_state["production_df"]["Volume"].sum()
            total_hedged = 0
            
            for _, month_data in st.session_state["production_df"].iterrows():
                month = month_data["Month"]
                volume = month_data["Volume"]
                
                for strategy in st.session_state["hedge_strategies"]:
                    strategy_months = get_strategy_months(strategy["Start Month"], strategy["End Month"])
                    if month in strategy_months:
                        total_hedged += volume * (month_data["Hedge Ratio (%)"] / 100)
            
            hedge_coverage = (total_hedged / total_production) * 100 if total_production > 0 else 0
            
            st.metric("Total Production", f"{total_production:,.0f} bbl")
            st.metric("Total Hedged Volume", f"{total_hedged:,.0f} bbl")
            st.metric("Hedge Coverage", f"{hedge_coverage:.1f}%")
            
            # Strategy diversification
            strategy_types = [s["Type"] for s in st.session_state["hedge_strategies"]]
            unique_types = len(set(strategy_types))
            st.metric("Strategy Diversification", f"{unique_types} instrument types")
        
        with col2:
            st.markdown("#### Price Sensitivity Analysis")
            
            # Calculate price sensitivity
            base_price = current_oil_price
            price_shock = 10  # $10 price change
            
            base_revenue = 0
            shocked_revenue_up = 0
            shocked_revenue_down = 0
            
            for _, month_data in st.session_state["production_df"].iterrows():
                month = month_data["Month"]
                volume = month_data["Volume"]
                
                # Base case
                month_base = volume * base_price
                month_up = volume * (base_price + price_shock)
                month_down = volume * (base_price - price_shock)
                
                # Add hedge effects
                for strategy in st.session_state["hedge_strategies"]:
                    strategy_months = get_strategy_months(strategy["Start Month"], strategy["End Month"])
                    if month in strategy_months:
                        hedge_volume = volume * (month_data["Hedge Ratio (%)"] / 100)
                        
                        base_payoff = calculate_hedge_payoff(
                            strategy["Type"], base_price,
                            strike=strategy.get("Strike"),
                            floor=strategy.get("Floor"),
                            cap=strategy.get("Cap"),
                            premium=strategy.get("Premium", 0)
                        )
                        
                        up_payoff = calculate_hedge_payoff(
                            strategy["Type"], base_price + price_shock,
                            strike=strategy.get("Strike"),
                            floor=strategy.get("Floor"),
                            cap=strategy.get("Cap"),
                            premium=strategy.get("Premium", 0)
                        )
                        
                        down_payoff = calculate_hedge_payoff(
                            strategy["Type"], base_price - price_shock,
                            strike=strategy.get("Strike"),
                            floor=strategy.get("Floor"),
                            cap=strategy.get("Cap"),
                            premium=strategy.get("Premium", 0)
                        )
                        
                        month_base += hedge_volume * base_payoff
                        month_up += hedge_volume * up_payoff
                        month_down += hedge_volume * down_payoff
                
                base_revenue += month_base
                shocked_revenue_up += month_up
                shocked_revenue_down += month_down
            
            upside_change = shocked_revenue_up - base_revenue
            downside_change = shocked_revenue_down - base_revenue
            
            st.metric("Base Case Revenue", f"${base_revenue:,.0f}")
            st.metric(f"+${price_shock} Price Impact", f"${upside_change:,.0f}")
            st.metric(f"-${price_shock} Price Impact", f"${downside_change:,.0f}")
            
            # Calculate hedge effectiveness
            unhedged_up = total_production * (base_price + price_shock)
            unhedged_down = total_production * (base_price - price_shock)
            unhedged_base = total_production * base_price
            
            upside_protection = 1 - (upside_change / (unhedged_up - unhedged_base)) if (unhedged_up - unhedged_base) != 0 else 0
            downside_protection = 1 - (abs(downside_change) / abs(unhedged_down - unhedged_base)) if (unhedged_down - unhedged_base) != 0 else 0
            
            st.metric("Upside Participation", f"{(1-upside_protection)*100:.1f}%")
            st.metric("Downside Protection", f"{downside_protection*100:.1f}%")
        
        # Monte Carlo simulation section
        st.markdown("### 🎲 Monte Carlo Simulation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Simulation Parameters")
            num_simulations = st.slider("Number of Simulations", 100, 2000, 1000, 100,key ="mc_simulations")
            volatility_annual = volatility / 100
            time_horizon = 1.0  # 1 year
            
            if st.button("🚀 Run Simulation", type="primary",key='sim_run'):
                # Run Monte Carlo simulation
                np.random.seed(42)  # For reproducible results
                
                simulation_results = []
                
                for sim in range(num_simulations):
                    # Generate random price path (geometric Brownian motion)
                    dt = time_horizon / 12  # Monthly steps
                    price_path = [current_oil_price]
                    
                    for month in range(12):
                        drift = (risk_free_rate / 100 - 0.5 * volatility_annual**2) * dt
                        shock = volatility_annual * np.sqrt(dt) * np.random.normal()
                        next_price = price_path[-1] * np.exp(drift + shock)
                        price_path.append(next_price)
                    
                    # Calculate revenue for this price path
                    total_revenue = 0
                    
                    for month_idx, month_data in st.session_state["production_df"].iterrows():
                        month = month_data["Month"]
                        volume = month_data["Volume"]
                        price = price_path[month_idx + 1]  # +1 because path starts with current price
                        
                        month_revenue = volume * price
                        
                        # Add hedge effects
                        for strategy in st.session_state["hedge_strategies"]:
                            strategy_months = get_strategy_months(strategy["Start Month"], strategy["End Month"])
                            if month in strategy_months:
                                hedge_volume = volume * (month_data["Hedge Ratio (%)"] / 100)
                                payoff = calculate_hedge_payoff(
                                    strategy["Type"], price,
                                    strike=strategy.get("Strike"),
                                    floor=strategy.get("Floor"),
                                    cap=strategy.get("Cap"),
                                    premium=strategy.get("Premium", 0)
                                )
                                month_revenue += hedge_volume * payoff
                        
                        total_revenue += month_revenue
                    
                    simulation_results.append({
                        "Simulation": sim + 1,
                        "Final_Price": price_path[-1],
                        "Total_Revenue": total_revenue,
                        "Avg_Price": np.mean(price_path[1:])
                    })
                
                sim_df = pd.DataFrame(simulation_results)
                st.session_state["simulation_results"] = sim_df
        
        with col2:
            if "simulation_results" in st.session_state:
                sim_df = st.session_state["simulation_results"]
                
                # Revenue distribution
                fig_dist = px.histogram(
                    sim_df,
                    x="Total_Revenue",
                    nbins=50,
                    title="Revenue Distribution (Monte Carlo)",
                    labels={"Total_Revenue": "Total Revenue ($)"}
                )
                fig_dist.add_vline(
                    x=sim_df["Total_Revenue"].mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Mean"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Statistics
                st.markdown("#### Simulation Statistics")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Mean Revenue", f"${sim_df['Total_Revenue'].mean():,.0f}")
                    st.metric("Std Deviation", f"${sim_df['Total_Revenue'].std():,.0f}")
                
                with col_b:
                    st.metric("5th Percentile", f"${sim_df['Total_Revenue'].quantile(0.05):,.0f}")
                    st.metric("95th Percentile", f"${sim_df['Total_Revenue'].quantile(0.95):,.0f}")
                
                with col_c:
                    st.metric("Value at Risk (5%)", f"${sim_df['Total_Revenue'].mean() - sim_df['Total_Revenue'].quantile(0.05):,.0f}")
                    st.metric("Max Revenue", f"${sim_df['Total_Revenue'].max():,.0f}")
            else:
                st.info("Click 'Run Simulation' to see Monte Carlo results")