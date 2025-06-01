import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Gold Price Prediction Dashboard", layout="wide")

st.title("Gold Price Prediction Dashboard")

PREDICTIONS_PATH = "Artifacts/batch_preds.csv"

# Load data
@st.cache_data
def load_data(pred_path):
    pred_df = pd.read_csv(pred_path, parse_dates=["dates"])
    return pred_df

if not os.path.exists(PREDICTIONS_PATH):
    st.error("Prediction data file not found. Please run the batch prediction pipeline first.")
else:
    pred_df = load_data(PREDICTIONS_PATH)

    # Show only the latest week's predictions
    latest_week = pred_df["dates"].max() - pd.Timedelta(days=12)
    week_df = pred_df[pred_df["dates"] >= latest_week].sort_values("dates")

    st.subheader("Batch Predictions Table (Latest Week)")
    st.dataframe(week_df[["dates", "predictions"]], use_container_width=True)

    st.subheader("Predicted Gold Prices (Latest Week)")
    st.line_chart(
        week_df.set_index("dates")[["predictions"]],
        use_container_width=True
    )

    st.markdown("---")
    st.caption("Gold Price Prediction System | Streamlit Dashboard")

    st.info(" Fetch Batch Predictions for other days.")