# RentalPropertyInsights
This repository is a solution to the C1DC by Kasey Suszko

Contents Include:
Airbnb_Data_Exploration.ipynb : offers exploratory insights into the Airbnb data so the rental property can set up their property to optimize rental potential

Data_Processing_Zillow.ipynb : cleans the 'Zip_Zhvi_2bedroom.csv' file, uses ARIMA forecasting and CAGR, and prepares file to merge with Airbnb data. It adds many variables for ARIMA forecasts over 10+ years. Was meant to be used in further analysis, but was unable to due to time constraints

Data_Processing_Airbnb_Combination_ARIMA.ipynb : cleans the 'listings.csv' data, adds new variables such as a sentiment score from the summaries, annual rent based on Airbnb price, and profit/loss predictions based on selling the property in 2020, 2025, and 2030 (with use of ARIMA forecasts), has visualizations that are the primary data insights (2 interactive bar charts and an interactive map)

ARIMA.py : includes most of the functions for Data_Processing_Zillow.ipynb . Functions were added to this file to enhance readability in the notebook file.

DataCleaning.py : includes most of the functions for Data_Processing_Airbnb_Combination_ARIMA.ipynb . Functions were added to this file to enhance readability in the notebook file.

cleaned_geodata.json : geojson data for NYC boroughs found online

neighbourhoods.geojson : geojson data for NYC from Airbnb website

zillow_cleaned.csv : output of Data_Processing_Zillow.ipynb used in Data_Processing_Airbnb_Combination_ARIMA.ipynb

# OBJECTIVE
Identifying Zip Codes for 2-bedroom sets in New York City that would generate the most profit on short term rentals

Background:

You are consulting for a real estate company that has a niche in purchasing properties to rent out short-term as part of their business model specifically within New York City. The real estate company has already concluded that two bedroom properties are the most profitable; however, they do not know which zip codes are the best to invest in.

The objective is to identify the zip codes would generate the most profit on short term rentals within New York City.

Data: Publicly available data sets from Zillow and AirBnB

# ASSUMPTIONS
1. The investor will pay for the property in cash 
2. The time value of money discount rate is 0% 
3. All properties and all square feet within each locale can be assumed to be homogeneous 
4. The occupancy rate is assumed to be 75%
5. The company charges the same rent on Airbnb every night. This allows the Airbnb price to be used as a rent estimate
6. There is no disruption to the general economic environment. The values estimated for 2020 are based on historical data, not known values.
7. The rental agency is interested in selling the properties in the future
8. All two bedrooms are assumed to have the same square footage and the same median property value as other 2-bedroom properties in their zipcode
