import streamlit as st
import pandas as pd
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# MongoDB connection
# -------------------------------
MONGO_URI = "mongodb+srv://harithzahrin_db_user:VH2COsd5t6n84ckO@crudelabs.28ae9lf.mongodb.net/?retryWrites=true&w=majority&appName=CrudeLabs"
client = MongoClient(MONGO_URI)
db = client["OptionTrades_db"]
collection = db["OptionTrades"]
tenure_collection = db["Tenures"]

st.set_page_config(page_title="Option Trades Manager", layout="wide")

st.title("📊 Option Trades Manager")

# -------------------------------
# Sidebar Navigation
# -------------------------------
menu = st.sidebar.radio(
    "Navigation", 
    ["➕ Add Trade", "📂 View Trades", "✏️ Edit/Delete Trades", "📅 Manage Tenures", "📈 Analysis"]
)

# -------------------------------
# Add Trade
# -------------------------------
if menu == "➕ Add Trade":
    st.subheader("➕ Add a New Trade")

    with st.form("trade_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            trade_no = st.number_input("Trade No", min_value=1, step=1)
            trade_date = st.date_input("Trade Date")
            underlying = st.text_input("Underlying")
            
            # Load available tenures from Tenures collection
            tenures = list(tenure_collection.find({}))
            tenure_options = [t["Tenure"] for t in tenures]
            tenure = st.selectbox("Tenure", tenure_options if tenure_options else ["None"])

        with col2:
            option_type = st.selectbox("Option Type", ["Call", "Put"])
            direction = st.selectbox("Direction", ["Buy", "Sell"])
            strike_price = st.number_input("Strike Price", step=0.01)
            entry_price = st.number_input("Entry Price", step=0.01)

        with col3:
            closing_price = st.number_input("Closing Price", step=0.01, value=0.0)
            close_date = st.date_input("Close Date")

        submitted = st.form_submit_button("Save Trade")

        if submitted:
            trade_doc = {
                "Trade No": trade_no,
                "Trade Date": str(trade_date),
                "Underlying": underlying,
                "Tenure": tenure,
                "Option Type": option_type,
                "Direction": direction,
                "Strike Price": strike_price,
                "Entry Price": entry_price,
                "Closing Price": closing_price if closing_price > 0 else None,
                "Close Date": str(close_date) if closing_price > 0 else None,
            }

            collection.insert_one(trade_doc)
            st.success(f"✅ Trade {trade_no} added successfully!")

# -------------------------------
# View Trades
# -------------------------------
elif menu == "📂 View Trades":
    st.subheader("📂 All Trades")

    trades = list(collection.find({}, {"_id": 0}))
    tenures = list(tenure_collection.find({}))
    tenure_map = {t["Tenure"]: t for t in tenures}

    if trades:
        df = pd.DataFrame(trades)

        # Merge with tenure data
        df["Expiration Date"] = df["Tenure"].map(lambda t: tenure_map.get(t, {}).get("Expiration Date"))
        df["Underlying Price"] = df["Tenure"].map(lambda t: tenure_map.get(t, {}).get("Underlying Price"))

        # Convert expiration date
        df["Expiration Date"] = pd.to_datetime(df["Expiration Date"], errors="coerce").dt.date

        today = datetime.today().date()
        df["Days to Expiry"] = df["Expiration Date"].apply(
            lambda d: max((d - today).days, 0) if pd.notna(d) else None
        )
        df["Expired?"] = df["Days to Expiry"].apply(
            lambda x: "Yes" if x == 0 else "No"
        )

        # Filtering
        col1, col2 = st.columns(2)
        with col1:
            tenure_filter = st.selectbox("Filter by Tenure", ["All"] + sorted(df["Tenure"].dropna().unique().tolist()))
        with col2:
            option_filter = st.selectbox("Filter by Option Type", ["All"] + sorted(df["Option Type"].dropna().unique().tolist()))

        if tenure_filter != "All":
            df = df[df["Tenure"] == tenure_filter]
        if option_filter != "All":
            df = df[df["Option Type"] == option_filter]

        st.dataframe(
            df[
                [
                    "Trade No", "Trade Date", "Underlying", "Tenure",
                    "Underlying Price", "Option Type", "Direction", "Strike Price",
                    "Entry Price", "Closing Price", "Close Date",
                    "Expiration Date", "Days to Expiry", "Expired?"
                ]
            ],hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No trades found in the database.")





# -------------------------------
# Edit/Delete Trades
# -------------------------------
elif menu == "✏️ Edit/Delete Trades":
    st.subheader("✏️ Edit or Delete Trades")

    trades = list(collection.find({}))
    if not trades:
        st.info("No trades available to edit or delete.")
    else:
        df = pd.DataFrame(trades)

        # Selection list shows: Trade No, Option Type, Direction, Tenure
        selected_index = st.selectbox(
            "Choose a trade",
            df.index,
            format_func=lambda x: f"Trade {df.loc[x, 'Trade No']} | {df.loc[x, 'Option Type']} | {df.loc[x, 'Direction']} | {df.loc[x, 'Tenure']}"
        )

        selected_trade = df.loc[selected_index]
        trade_id = selected_trade["_id"]

        # Edit form: only Closing Price + Close Date
        with st.form("edit_form"):
            st.write(f"Editing Trade {selected_trade['Trade No']}")

            closing_price = st.number_input(
                "Closing Price",
                step=0.01,
                value=float(selected_trade.get("Closing Price") or 0)
            )
            close_date = st.date_input(
                "Close Date",
                value=pd.to_datetime(selected_trade.get("Close Date", pd.Timestamp.today()))
            )

            update_btn = st.form_submit_button("Update Trade")
            if update_btn:
                collection.update_one(
                    {"_id": ObjectId(trade_id)},
                    {"$set": {
                        "Closing Price": closing_price if closing_price > 0 else None,
                        "Close Date": str(close_date) if closing_price > 0 else None
                    }}
                )
                st.success("✅ Trade updated successfully!")

        # Delete button
        if st.button("🗑️ Delete This Trade"):
            collection.delete_one({"_id": ObjectId(trade_id)})
            st.warning(f"Trade {selected_trade['Trade No']} deleted.")
            st.experimental_rerun()

# -------------------------------
# Manage Tenures
# -------------------------------
elif menu == "📅 Manage Tenures":
    st.subheader("📅 Manage Tenures")

    # Show existing tenures
    tenures = list(tenure_collection.find({}))
    if tenures:
        df = pd.DataFrame(tenures)
        today = datetime.today().date()

        # Convert expiration date
        df["Expiration Date"] = pd.to_datetime(df["Expiration Date"], errors="coerce").dt.date

        # Clamp days to expiry at 0
        df["Days to Expiry"] = df["Expiration Date"].apply(
            lambda d: max((d - today).days, 0) if pd.notna(d) else None
        )
        df["Expired?"] = df["Days to Expiry"].apply(
            lambda x: "Yes" if x == 0 else "No"
        )

        # Ensure underlying price column exists
        if "Underlying Price" not in df.columns:
            df["Underlying Price"] = None

        st.dataframe(
            df[["Tenure", "Expiration Date", "Underlying Price", "Days to Expiry", "Expired?"]],
            use_container_width=True
        )
    else:
        st.info("No tenures found in the database.")

    st.markdown("---")

    # Add new tenure
    st.subheader("➕ Add a New Tenure")
    with st.form("tenure_form"):
        tenure_code = st.text_input("Tenure Code (e.g., CLU25)")
        expiration_date = st.date_input("Expiration Date")
        underlying_price = st.number_input("Underlying Price", min_value=0.0, step=0.01, format="%.2f")
        add_btn = st.form_submit_button("Save Tenure")

        if add_btn:
            tenure_collection.insert_one({
                "Tenure": tenure_code,
                "Expiration Date": str(expiration_date),
                "Underlying Price": underlying_price
            })
            st.success(f"✅ Tenure {tenure_code} added successfully!")

    st.markdown("---")

    # Edit or Delete tenure
    st.subheader("⚙ Manage Existing Tenure")
    tenures = list(tenure_collection.find({}))
    if tenures:
        tenure_df = pd.DataFrame(tenures)
        selected_idx = st.selectbox(
            "Choose a tenure",
            tenure_df.index,
            format_func=lambda x: tenure_df.loc[x, "Tenure"]
        )

        selected_tenure = tenure_df.loc[selected_idx]
        tenure_id = selected_tenure["_id"]

        with st.form("edit_delete_tenure_form"):
            st.write(f"**Editing Tenure:** {selected_tenure['Tenure']}")

            new_expiration_date = st.date_input(
                "Expiration Date",
                value=pd.to_datetime(selected_tenure["Expiration Date"], errors="coerce").date()
            )
            new_underlying_price = st.number_input(
                "Underlying Price",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                value=float(selected_tenure.get("Underlying Price") or 0.0)
            )

            col1, col2 = st.columns(2)
            with col1:
                update_btn = st.form_submit_button("Update Tenure")
            with col2:
                delete_btn = st.form_submit_button("Delete Tenure", type="secondary")

            if update_btn:
                tenure_collection.update_one(
                    {"_id": ObjectId(tenure_id)},
                    {"$set": {
                        "Expiration Date": str(new_expiration_date),
                        "Underlying Price": new_underlying_price
                    }}
                )
                st.success("✅ Tenure updated successfully!")

            if delete_btn:
                tenure_collection.delete_one({"_id": ObjectId(tenure_id)})
                st.success(f"❌ Tenure {selected_tenure['Tenure']} deleted successfully!")
    else:
        st.info("No tenures available.")

# -------------------------------
# Analysis Tab
# -------------------------------
elif menu == "📈 Analysis":
    st.subheader("📈 Trade Analysis")
    
    trades = list(collection.find({}, {"_id": 0}))
    
    if not trades:
        st.info("No trades available for analysis.")
    else:
        df = pd.DataFrame(trades)

        st.markdown("### Strike Price by Tenure (with Underlying Price points)")

        trades = list(collection.find({}, {"_id": 0}))
        tenures = list(tenure_collection.find({}))

        # Map tenure → data
        tenure_map = {t["Tenure"]: t for t in tenures}

        # Add Underlying Price to each trade
        for trade in trades:
            tenure_info = tenure_map.get(trade.get("Tenure"))
            trade["Underlying Price"] = tenure_info.get("Underlying Price") if tenure_info else None

        df = pd.DataFrame(trades)

        st.markdown("### Strike Price by Tenure (with Underlying Price points)")

        # Main scatter for Strike Price
        fig_strike = px.scatter(
            df,
            x="Tenure",
            y="Strike Price",
            color="Direction",
            color_discrete_map={"Buy": "#00CC96", "Sell": "#EF553B"},
            labels={
                "Strike Price": "Strike Price",
                "Tenure": "Tenure",
                "Underlying Price": "Underlying Price"
            },
            title="Strike Price by Tenure (Buy vs Sell)",
            hover_data=["Underlying", "Option Type", "Entry Price", "Trade Date", "Underlying Price"]
        )

        # Add Underlying Price as extra points
        fig_strike.add_scatter(
            x=df["Tenure"],
            y=df["Underlying Price"],
            mode="markers",
            marker=dict(symbol="x", size=12, color="blue"),
            name="Underlying Price"
        )

        #fig_strike.update_layout(height=500)
        # Set y-axis increments to $1
        fig_strike.update_layout(
            height=500,
            yaxis=dict(dtick=1)
        )
        st.plotly_chart(fig_strike, use_container_width=True)


        # Strike Price by Tenure - Heatmap
        st.markdown("### Strike Price by Tenure (Heatmap)")

        fig_heatmap = px.density_heatmap(
            df,
            x="Tenure",
            y="Strike Price",
            z=None,  # counts points by default
            nbinsx=20,  # adjust for resolution
            nbinsy=20,
            color_continuous_scale="Viridis",
            labels={"Strike Price": "Strike Price", "Tenure": "Tenure"},
            title="Heatmap of Strike Price by Tenure"
        )

        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
       
        