import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pycaret.regression import *
from vega_datasets import data as veg_data
st.set_page_config(layout='wide')
alt.data_transformers.disable_max_rows()

# Loading dataset
def load_data():
    data = pd.read_csv('US_Accidents_Dec21.csv')
    return data


# Cleaning data
def clean_data(df):
    # Convert Start_Time and End_Time to datetimes
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

    # Extract year, month, day, hour and weekday
    df['Year'] = df['Start_Time'].dt.year
    df['Month'] = df['Start_Time'].dt.strftime('%b')
    df['Day'] = df['Start_Time'].dt.day
    df['Hour'] = df['Start_Time'].dt.hour
    df['Weekday'] = df['Start_Time'].dt.strftime('%a')

    # Extract the amount of time in the unit of minutes for each accident, round to the nearest integer
    df['Duration'] = round((df['End_Time'] - df['Start_Time']) / np.timedelta64(1, 'm'))

    # Drop the rows where duration is negative
    df = df[df['Duration'] >= 0]

    # Handle missing values
    # For categorical variables like 'City', 'Zipcode', 'Timezone' etc., we'll fill missing values with the mode (most frequently occurring       value)
    for col in ['City', 'Zipcode', 'Timezone', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',
                'Astronomical_Twilight']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # For numerical weather-related variables, we'll fill missing values with the median
    for col in ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
                'Precipitation(in)']:
        df[col].fillna(df[col].median(), inplace=True)

    # For categorical weather-related variables, fill with the mode
    for col in ['Weather_Condition', 'Wind_Direction']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # For POI variables, we'll assume missing values mean the POI was absent
    for col in ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
                'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']:
        df[col].fillna(False, inplace=True)

    # Remove rows where 'Lat' or 'Lng' is missing
    df = df.dropna(subset=['Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng'])

    return df


def pred_dataset(df):
    grouped_multiple = df.groupby(['State', 'City', 'Street', 'Weather_Condition', 'Day_Time']).agg(
        {'Temperature(F)': ['mean'], 'Wind_Chill(F)': ['mean'], 'Pressure(in)': ['mean'], 'Wind_Speed(mph)': ['mean']})
    grouped_multiple.columns = ['Temp_mean', 'Wind_mean', 'Pressure_mean', 'Wind_speed_mean']
    grouped_multiple = grouped_multiple.reset_index()
    grouped_multiple = grouped_multiple.astype({'Street': str})
    grouped_multiple['Street'] = grouped_multiple['Street'].apply(lambda x: x.strip())
    return grouped_multiple

def get_prediction_data(df,Street,City,State):
    results = df.loc[(df['State'] == State) & (df['City'] == City) & (
                df['Street'] == Street)]
    return results

def get_2021_dataset(df):
    cleaned_data_2021 = df.loc[df['Year'] == 2021]
    cleaned_data_2021 = cleaned_data_2021.reset_index(drop=True)
    return cleaned_data_2021

def get_2017_dataset(df):
    cleaned_data_2017 = df.loc[df['Year'] == 2017]
    cleaned_data_2017 = cleaned_data_2017.reset_index(drop=True)
    return cleaned_data_2017

def top_states_accident(df):
    temp_df = df.groupby(['State']).agg({'Number':sum})
    states = temp_df.apply(lambda x: x.sort_values(ascending=False).head(10))
    states = states.reset_index()
    df2 = df[df['State'].isin(list(states['State']))]
    return df2

def predict_model_2017(test_data):
    rf_saved = load_model('final_rf_model_2017')
    predictions = predict_model(rf_saved, data = test_data)
    return predictions['prediction_label']

def predict_model_2021(test_data):
    rf_saved = load_model('final_rf_model_2021')
    predictions = predict_model(rf_saved, data = test_data)
    return predictions['prediction_label']

def predict_model_both(test_data):
    rf_saved = load_model('final_rf_model_both_years')
    predictions = predict_model(rf_saved, data = test_data)
    return predictions['prediction_label']

def plot_points(data):
    points = alt.Chart(data).mark_circle().encode(
        longitude='Start_Lng:Q',
        latitude='Start_Lat:Q',
        size=alt.value(10),
        tooltip='Weather_Condition'
    )
    return points

def main():
    st.title("US Accidents Analysis")

    data = load_data()
    cleaned_data = clean_data(data)
    temp_cleanned_data = cleaned_data[
        ['Severity', 'Weather_Condition', 'State', 'City', 'Street', 'Temperature(F)', 'Wind_Chill(F)',
         'Pressure(in)', 'Wind_Speed(mph)', 'Year', 'Hour']]
    temp_cleanned_data = temp_cleanned_data.reset_index(drop=True)
    time_conditions = [(temp_cleanned_data['Hour'] >= 6) & (temp_cleanned_data['Hour'] < 12),
                       (temp_cleanned_data['Hour'] >= 12) & (temp_cleanned_data['Hour'] < 18),
                       (temp_cleanned_data['Hour'] >= 18) & (temp_cleanned_data['Hour'] <= 23),
                       (temp_cleanned_data['Hour'] >= 0) & (temp_cleanned_data['Hour'] < 6)]
    time_categories = ['Morning', 'Afternoon', 'Evening', 'Night']
    temp_cleanned_data['Day_Time'] = np.select(time_conditions, time_categories)
    
    
    # Create tabs at the top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Data"):
            st.session_state.selected_tab = "Data"
    with col2:
        if st.button("Charts"):
            st.session_state.selected_tab = "Charts"
    with col3:
        if st.button("Choropleth"):
            st.session_state.selected_tab = "Choropleth"
    with col4:
        if st.button("Prediction"):
            st.session_state.selected_tab = "Prediction"

    if not "selected_tab" in st.session_state:
        st.session_state.selected_tab = "Data"

    # Display content based on the selected tab
    if st.session_state.selected_tab == "Data":
        st.header("Raw Data and Cleaned Data")
        radio_selection = st.radio("Choose data to display:", ("Raw Data", "Cleaned Data"))

        if radio_selection == "Raw Data":
            st.write(data)
        elif radio_selection == "Cleaned Data":
            cleaned_df = clean_data(data)
            st.write(cleaned_df)

    elif st.session_state.selected_tab == "Charts":
        # charting code 

        # Ensure 'Start_Time' is in datetime format
        cleaned_data['Start_Time'] = pd.to_datetime(cleaned_data['Start_Time'])

        # Extract 'Hour', 'Month', and 'Weekday' from 'Start_Time'
        cleaned_data['Hour'] = cleaned_data['Start_Time'].dt.hour
        cleaned_data['Month'] = cleaned_data['Start_Time'].dt.month
        cleaned_data['Weekday'] = cleaned_data['Start_Time'].dt.weekday

        st.title(" ")
        st.title("Bar Charts")
        # Bar charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))



        # Bar Chart 1
        sns.countplot(ax=axes[0, 0], x='State', data=cleaned_data,
                      order=cleaned_data['State'].value_counts().iloc[:5].index)
        axes[0, 0].set_title('Top 5 States with Most Accidents')

        # Bar Chart 2
        sns.countplot(ax=axes[0, 1], x='Hour', data=cleaned_data)
        axes[0, 1].set_title('Accidents by Hour of the Day')


        # Bar Chart 3
        sns.countplot(ax=axes[1, 0], x='Month', data=cleaned_data)
        axes[1, 0].set_title('Accidents by Month')

        # Bar Chart 4
        sns.countplot(ax=axes[1, 1], x='Weekday', data=cleaned_data)
        axes[1, 1].set_title('Accidents by Day of the Week')

        plt.tight_layout()

        # Display bar charts in Streamlit
        st.pyplot(fig)
        
        
        st.title(" ")
        st.title("Pie Charts")
        # Pie charts
        fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))

        # Pie Chart 1
        severity_counts = cleaned_data['Severity'].value_counts()
        axes2[0, 0].pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%')
        axes2[0, 0].set_title('Accidents by Severity')

        # Pie Chart 2
        weather_condition_counts = cleaned_data['Weather_Condition'].value_counts().iloc[:5]
        axes2[0, 1].pie(weather_condition_counts, labels=weather_condition_counts.index, autopct='%1.1f%%')
        axes2[0, 1].set_title('Top 5 Weather Conditions in Accidents')


        # Pie Chart 3
        side_counts = cleaned_data['Side'].value_counts()
        axes2[1, 0].pie(side_counts, labels=side_counts.index, autopct='%1.1f%%')
        axes2[1, 0].set_title('Accidents by Side of the Road')

        # Pie Chart 4
        sunrise_sunset_counts = cleaned_data['Sunrise_Sunset'].value_counts()
        axes2[1, 1].pie(sunrise_sunset_counts, labels=sunrise_sunset_counts.index, autopct='%1.1f%%')
        axes2[1, 1].set_title('Accidents by Day and Night')

        

        plt.tight_layout()

        # Display pie charts in Streamlit
        st.pyplot(fig2)

#Choropleth

    elif st.session_state.selected_tab == "Choropleth":
        # choropleth code here
        
        

        
        states = alt.topo_feature(veg_data.us_10m.url, 'states')
        data_2021 = get_2021_dataset(cleaned_data)
        data_2017 = get_2017_dataset(cleaned_data)
        background = alt.Chart(states).mark_geoshape(
            fill='white',
            stroke='black'
        ).project('albersUsa').properties(
            width=500,
            height=300
        )

        points_2021 = alt.Chart(data_2021).mark_circle().encode(
            longitude='Start_Lng:Q',

            latitude='Start_Lat:Q',
            size=alt.value(10),
            tooltip='Weather_Condition'
        )
        points_2017 = alt.Chart(data_2017).mark_circle().encode(
            longitude='Start_Lng:Q',
            latitude='Start_Lat:Q',
            size=alt.value(10),
            tooltip='Weather_Condition'
        )
        
        top_state_2017 = top_states_accident(data_2017)
        top_states_points_2017 = plot_points(top_state_2017)
        top_state_2021 = top_states_accident(data_2021)
        top_states_points_2021 = plot_points(top_state_2021)
        
        st.markdown("<h2 style='text-align: center;'>2017 vs 2021</h2>", unsafe_allow_html=True)
        
        
        st.markdown("<h5 style='text-align: center;'>Accident occured areas</h5>", unsafe_allow_html=True)
        cols = st.columns(2)
        st.markdown("<h5 style='text-align: center;'>Top 10 accident occured areas</h5>", unsafe_allow_html=True)
        cols1 = st.columns(2)

        cols[0].write('2017')
        cols[1].write('2021')
        cols[0].write(background+points_2017)
        cols[1].write(background+points_2021)

        cols1[0].write('2017')
        cols1[1].write('2021')
        cols1[0].write(background+top_states_points_2017)
        cols1[1].write(background + top_states_points_2021)
      

        st.markdown("<h2 style='text-align: center;'>Analysis of Entire Data</h2>", unsafe_allow_html=True)
      

        # Choropleth map 1: Number of accidents by state
        accidents_by_state = cleaned_data['State'].value_counts().reset_index()
        choropleth_map1 = px.choropleth(accidents_by_state, locations='index', color='State', locationmode='USA-states',
                                        scope='usa', title='Number of Accidents by State')

        # Choropleth map 2: Average severity of accidents by state
        avg_severity_by_state = cleaned_data.groupby('State')['Severity'].mean().reset_index()
        choropleth_map2 = px.choropleth(avg_severity_by_state, locations='State', color='Severity',
                                        locationmode='USA-states', scope='usa',
                                        title='Average Severity of Accidents by State')

        # Choropleth map 3: Average temperature during accidents by state
        avg_temperature_by_state = cleaned_data.groupby('State')['Temperature(F)'].mean().reset_index()
        choropleth_map3 = px.choropleth(avg_temperature_by_state, locations='State', color='Temperature(F)',
                                        locationmode='USA-states', scope='usa',
                                        title='Average Temperature during Accidents by State')

        # Choropleth map 4: Average visibility during accidents by state
        avg_visibility_by_state = cleaned_data.groupby('State')['Visibility(mi)'].mean().reset_index()
        choropleth_map4 = px.choropleth(avg_visibility_by_state, locations='State', color='Visibility(mi)',
                                        locationmode='USA-states', scope='usa',
                                        title='Average Visibility during Accidents by State')

        # Choropleth map 5: Average wind speed during accidents by state
        avg_wind_speed_by_state = cleaned_data.groupby('State')['Wind_Speed(mph)'].mean().reset_index()
        choropleth_map5 = px.choropleth(avg_wind_speed_by_state, locations='State', color='Wind_Speed(mph)',
                                        locationmode='USA-states', scope='usa',
                                        title='Average Wind Speed during Accidents by State')
        
         
        

        # Choropleth map 7: Average humidity during accidents by state
        avg_humidity_by_state = cleaned_data.groupby('State')['Humidity(%)'].mean().reset_index()
        choropleth_map7 = px.choropleth(avg_humidity_by_state, locations='State', color='Humidity(%)',
                                        locationmode='USA-states', scope='usa',
                                        title='Average Humidity during Accidents by State')

        # Choropleth map 8: Average precipitation during accidents by state
        avg_precipitation_by_state = cleaned_data.groupby('State')['Precipitation(in)'].mean().reset_index()
        choropleth_map8 = px.choropleth(avg_precipitation_by_state, locations='State', color='Precipitation(in)',
                                        locationmode='USA-states', scope='usa',
                                        title='Average Precipitation during Accidents by State')

        # Choropleth map 9: Proportion of accidents occurring during daylight by state
        accidents_daylight = cleaned_data[cleaned_data['Sunrise_Sunset'] == 'Day'].groupby('State').size().reset_index(
            name='Daylight_Accidents')
        total_accidents_by_state = cleaned_data.groupby('State').size().reset_index(name='Total_Accidents')
        accidents_daylight_proportion = accidents_daylight.merge(total_accidents_by_state, on='State')
        accidents_daylight_proportion['Proportion'] = accidents_daylight_proportion['Daylight_Accidents'] / \
                                                      accidents_daylight_proportion['Total_Accidents']
        choropleth_map9 = px.choropleth(accidents_daylight_proportion, locations='State', color='Proportion',
                                        locationmode='USA-states', scope='usa',
                                        title='Proportion of Accidents Occurring during Daylight by State')

        # Choropleth map 10: Proportion of accidents occurring near junctions by state
        accidents_junctions = cleaned_data[cleaned_data['Junction'] == True].groupby('State').size().reset_index(
            name='Junction_Accidents')
        accidents_junctions_proportion = accidents_junctions.merge(total_accidents_by_state, on='State')
        accidents_junctions_proportion['Proportion'] = accidents_junctions_proportion['Junction_Accidents'] / \
                                                       accidents_junctions_proportion['Total_Accidents']
        choropleth_map10 = px.choropleth(accidents_junctions_proportion, locations='State', color='Proportion',
                                         locationmode='USA-states', scope='usa',
                                         title='Proportion of Accidents Occurring near Junctions by State')
        
        
        
        st.plotly_chart(choropleth_map1)
        st.plotly_chart(choropleth_map2)
        st.plotly_chart(choropleth_map3)
        st.plotly_chart(choropleth_map4)
        st.plotly_chart(choropleth_map5)
        st.plotly_chart(choropleth_map7)
        st.plotly_chart(choropleth_map8)
        st.plotly_chart(choropleth_map9)
        st.plotly_chart(choropleth_map10)
        
        
    elif st.session_state.selected_tab == "Prediction":
        st.header("Prediction")
        cols = st.columns(6)
        pred_data  = pred_dataset(temp_cleanned_data)
        State_list = list(pred_data.State.unique())
        weather_list = list(pred_data.Weather_Condition.unique())
        day_list = list(pred_data.Day_Time.unique())
        state = cols[0].selectbox('Select a State',State_list,index=0)
        City_df = pred_data.loc[(pred_data['State'] == state)]
        City_list = list(City_df.City.unique())
        city = cols[1].selectbox('Select City',City_list,index=0)
        Street_df = pred_data.loc[(pred_data['State'] == state)&(pred_data['City'] == city)]
        Street_list = list(Street_df.Street.unique())
        street = cols[2].selectbox('Select Street',Street_list,index=0)
        weather = cols[3].selectbox('Select Weather',weather_list,index=0)
        day = cols[4].selectbox('Select day time', day_list,index=0)

        select = cols[5].radio('Select Data set',('YEAR 2017','YEAR 2021','2017 & 2021'))
        year = ''
        Sevearity = ''
        if select == 'YEAR 2017':
            year = '2017'

        if select == 'YEAR 2021':
            year = '2021'

        if select == '2017 & 2021':
            year = '2017&2021'
        
        st.markdown("<div style='color: red;'>*Prediction scale is 1 - 4 </div>", unsafe_allow_html=True)

        start_predict = st.button('Predict')
        if start_predict:
            test_data = get_prediction_data(pred_data, street, city, state)
            test_data = test_data.reset_index(drop = True)
            temp = test_data.iloc[0]['Temp_mean']
            test = pd.DataFrame({'State': [state], 'City': [city],
                                 'Street': [street],
                                 'Weather_Condition': [weather],
                                 'Temperature(F)': [test_data.iloc[0]['Temp_mean']],
                                 'Wind_Chill(F)': [test_data.iloc[0]['Wind_mean']],
                                 'Pressure(in)': [test_data.iloc[0]['Pressure_mean']],
                                 'Wind_Speed(mph)': [test_data.iloc[0]['Wind_speed_mean']],
                                 'Day_Time': [day]
                                 })
            if year == '2017':
                Sevearity = predict_model_2017(test)[0]

            if year == '2021':
                Sevearity = predict_model_2021(test)[0]

            if year == '2017&2021':
                Sevearity = predict_model_2017(test)[0]

            if Sevearity >=1 and Sevearity <2:
                st.write('Predicted Severity based on '+year+' data set = ' + str(Sevearity))
                st.write('Low risk of accidents')
            if Sevearity >=2 and Sevearity <3:
                st.write('Predicted Severity based on '+year+' data set = ' + str(Sevearity))
                st.write('Medium risk of accidents')
            if Sevearity >=3 and Sevearity <4:
                st.write('Predicted Severity based on '+year+' data set = ' + str(Sevearity))
                st.write('High risk of accidents')
            



        else:
            st.write("NO predictions yet")

if __name__ == "__main__":
    main()
