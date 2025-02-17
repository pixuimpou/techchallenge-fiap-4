import streamlit as st
import pickle
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go

FORECAST_LIMIT = 15


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


st.set_page_config(page_title="FIAP - Tech Challenge 4", layout="wide")


@st.cache_resource
def read_df():
    return pd.read_excel(
        "./data/raw/rbrted.xls",
        sheet_name="Data 1",
        skiprows=2,
    ).rename(
        columns={
            "Date": "ds",
            "Europe Brent Spot Price FOB (Dollars per Barrel)": "y",
        }
    )


model = load_model()
historical_data = read_df()

last_date = historical_data["ds"].max().to_pydatetime().date()

historical_data = historical_data.loc[
    historical_data["ds"].dt.date > last_date - timedelta(days=30)
]


st.header(":red[Previsão do Preço do Petróleo Brent]", divider="red")

st.info(
    f"A data de previsão foi limitada a {FORECAST_LIMIT} dias após a data final da base de dados. Atualmente a data final da base é: {last_date.isoformat()}"
)
with st.container():
    col, _ = st.columns([0.25, 0.75])
    with col:

        min_value = last_date + timedelta(days=1)
        max_value = last_date + timedelta(days=FORECAST_LIMIT)

        input_date = st.date_input(
            "**Insira a data para previsão**",
            min_value=min_value,
            max_value=max_value,
            value=min_value,
        )
        h = (input_date - last_date).days
        button_clicked = st.button("OK", use_container_width=True)

if button_clicked:
    with st.spinner("Realizando a previsão..."):
        prediction = (
            model.predict(h=h)
            .reset_index(drop=True)
            .rename(columns={"SeasonalNaive": "y"})
            .drop(columns="unique_id")
        )

        trace1 = go.Scatter(
            x=historical_data["ds"],
            y=historical_data["y"],
            mode="lines",
            name="Dados Históricos",
        )
        trace2 = go.Scatter(
            x=prediction["ds"],
            y=prediction["y"],
            mode="lines",
            name="Previsão",
            line_color="red",
        )

        layout = go.Layout(
            xaxis={"title": "Data"},
            yaxis={"title": "Preço do Barril de Petróleo (US$)"},
        )

        fig = go.Figure(data=[trace1, trace2], layout=layout)

        st.plotly_chart(fig)

        df_result = prediction.rename(
            columns={"ds": "Data", "y": "Preço Previsto (US$)"}
        )

        df_result["Data"] = df_result["Data"].dt.strftime("%Y-%m-%d")

        st.write(
            df_result.rename(columns={"ds": "Data", "y": "Preço Previsto"}).set_index(
                "Data"
            )
        )
