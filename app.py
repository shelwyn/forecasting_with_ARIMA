import pandas as pd
import statsmodels.api as sm
import warnings
from dateutil.relativedelta import relativedelta
import streamlit as st

class SalesPredictor:
    warnings.filterwarnings("ignore")
    st.set_page_config(layout="wide")
    def __init__(self):
        self.sales_df=pd.read_csv('sales.csv')
        self.products = list(set(self.sales_df['Product'].tolist()))
        self.sellers = list(set(self.sales_df['Seller'].tolist()))
        self.sales_df['Date'] = pd.to_datetime(self.sales_df['Date'])

    def forecast_arima(self, product, seller):
        subset = self.sales_df[(self.sales_df['Product'] == product) & (self.sales_df['Seller'] == seller)]
        time_series = pd.Series(subset['Sales'].values, index=subset['Date'])
        last_month = time_series.index[len(time_series) - 1]
        last_month = last_month + relativedelta(months=1)
        model = sm.tsa.ARIMA(time_series, order=(2, 1, 1))
        model_fit = model.fit()

        forecast_steps = 3
        forecast = model_fit.forecast(steps=forecast_steps)
        dates = pd.date_range(start=last_month, periods=forecast_steps, freq='M')

        date_pred = [date.strftime('%b %y') for date in dates]
        seller_pred = [seller] * forecast_steps
        product_pred = [product] * forecast_steps

        return date_pred, seller_pred, product_pred, forecast.tolist()

    def run(self):

        col_table, col_graph = st.columns(2)
        with st.sidebar:
            st.title("Sales Forecasting")
            selected_product = st.selectbox("Select a product", self.products)
            selected_seller = st.selectbox("Select a seller", self.sellers)
            if st.button("Predict"):
                date_pred, seller_pred, product_pred, forecast_pred = self.forecast_arima(selected_product,
                                                                                          selected_seller)
                predicted_dataframe = pd.DataFrame({
                    "Date": date_pred,
                    "Product": product_pred,
                    "Seller": seller_pred,
                    "Forecast": forecast_pred
                })
                col_graph.line_chart(
                    predicted_dataframe,
                    use_container_width=True,
                    height=200,
                    x='Forecast',
                    y='Date'
                )
                col_table.write(predicted_dataframe)

if __name__ == "__main__":
    predictor = SalesPredictor()
    predictor.run()