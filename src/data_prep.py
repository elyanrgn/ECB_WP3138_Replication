import pandas as pd
import numpy as np

# gdp processing
gdp = pd.read_csv("data\\raw\\GDP (1).csv", index_col=0, parse_dates=True)
gdp_growth = 400 * np.log(gdp / gdp.shift(1))
gdp_growth = gdp_growth.round(2)
inflation_exp = pd.read_csv("data\\raw\\EXPINF1YR.csv", index_col=0, parse_dates=True)
inflation_exp_quaterly = inflation_exp.resample("QS").mean()
# inflation_exp_quaterly_change = inflation_exp_quaterly.pct_change() * 100
inflation_exp_quaterly = inflation_exp_quaterly.round(2)
price_consumer_index = pd.read_csv("data\\raw\\amCharts.csv")
price_consumer_index.columns = ["TIME_PERIOD", "PCE"]
price_consumer_index["date"] = pd.to_datetime(
    price_consumer_index["TIME_PERIOD"], format="%YM%m"
)
price_consumer_index = price_consumer_index.set_index("date")[["PCE"]].dropna()
growth_factors = 1 + price_consumer_index["PCE"] / 100
growth_factors_q = growth_factors.resample("QS").prod()
price_consumer_index_quarterly_change = (growth_factors_q - 1) * 100 * 4
price_consumer_index_quarterly_change = price_consumer_index_quarterly_change.round(2)

crude_oil_prices = pd.read_csv("data\\raw\\WTISPLC.csv", index_col=0, parse_dates=True)
CRB_index = pd.read_csv(
    "data\\raw\\Thomson Reuters_CoreCommodity CRB Total Return Historical Data (1).csv",
    index_col=0,
    parse_dates=True,
)
CRB_index.index = pd.to_datetime(CRB_index.index)
CRB_index.sort_index(inplace=True)
CRB_index_price = CRB_index["Price"]
first_date = CRB_index.index.min()
crude_oil_prices = crude_oil_prices[crude_oil_prices.index < first_date]
crude_oil_prices.index = pd.to_datetime(crude_oil_prices.index)
crude_oil_prices.rename(columns={"WTISPLC": "Price"}, inplace=True)
full_crb_index = pd.concat([crude_oil_prices, CRB_index_price], axis=0)
full_crb_index_quarterly = full_crb_index.resample("QS").mean()
full_crb_index_quarterly_change = full_crb_index_quarterly.pct_change() * 100
full_crb_index_quarterly_change = full_crb_index_quarterly_change.round(2)
tb3m = pd.read_csv("data\\raw\\TB3MS.csv", index_col=0, parse_dates=True)
tb3m_quarterly = tb3m.resample("QS").last()
tb3m_quarterly = tb3m_quarterly.round(2)
input_data = pd.concat(
    [
        gdp_growth,
        inflation_exp_quaterly,
        price_consumer_index_quarterly_change,
        full_crb_index_quarterly_change,
        tb3m_quarterly,
    ],
    axis=1,
)
input_data.columns = [
    "GDP Growth (%)",
    "Expected Inflation (%)",
    "PCE Inflation (%)",
    "CRB Index Change (%)",
    "3-Month Treasury Bill Change (%)",
]
input_data = input_data[1:]  # Remove the first row with NaN values due to pct_change
input_data.to_csv("data\\processed\\ecb_estimation_prepared.csv", index=True)
gdp_prior = pd.read_csv("data\\raw\\GDP prior.csv", index_col=0, parse_dates=True)
gdp_prior_growth = gdp_prior.pct_change() * 100
gdp_prior_growth = gdp_prior_growth.round(2)

price_consumer_index_prior = pd.read_csv("data\\raw\\pce_prior.csv")
price_consumer_index_prior.columns = ["TIME_PERIOD", "PCE"]
price_consumer_index_prior["date"] = pd.to_datetime(
    price_consumer_index_prior["TIME_PERIOD"], format="%YM%m"
)
price_consumer_index_prior = price_consumer_index_prior.set_index("date")[
    ["PCE"]
].dropna()
growth_factors = 1 + price_consumer_index_prior["PCE"] / 100
growth_factors_q = growth_factors.resample("QS").prod()
price_consumer_index_quarterly_change_prior = (growth_factors_q - 1) * 100 * 4
price_consumer_index_quarterly_change_prior = (
    price_consumer_index_quarterly_change_prior.round(2)
)

t3bms_prior = pd.read_csv("data\\raw\\t3bms_prior.csv", index_col=0, parse_dates=True)
t3bms_prior = t3bms_prior.resample("QS").last()
t3bms_prior = t3bms_prior.round(2)

CBR_index_price_prior = pd.read_csv(
    "data\\raw\\CBR_index_prior.csv", index_col=0, parse_dates=True
)
CBR_index_price_prior = CBR_index_price_prior.resample("QS").mean()
CBR_index_price_prior_change = CBR_index_price_prior.pct_change() * 100
CBR_index_price_prior_change = CBR_index_price_prior_change.round(2)

input_data_prior = pd.concat(
    [
        gdp_prior_growth,
        price_consumer_index_quarterly_change_prior,
        CBR_index_price_prior_change,
        t3bms_prior,
    ],
    axis=1,
)
input_data_prior.columns = [
    "GDP Growth (%)",
    "PCE Inflation (%)",
    "CBR Index Price Change (%)",
    "3-Month Treasury Bill Change (%)",
]
input_data_prior = input_data_prior[
    1:
]  # Remove the first row with NaN values due to pct_change

input_data_prior.to_csv("data\\processed\\ecb_prior_prepared.csv", index=True)
