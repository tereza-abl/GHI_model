import numpy as np
import pandas as pd
import xarray as xr
import plotly.express as px
import datetime
import streamlit as st


def RMSE(observed, predicted):
    ''' Root mean square error metrics.'''
    return ((predicted - observed) ** 2).mean() ** 0.5


def MAE(observed, predicted):
    ''' Mean absolute error metrics.'''
    return (abs(predicted - observed)).mean()


def figure_2D(da_model, datetime_idx, lead_time_idx):
    fig = px.imshow(
        da_model.loc[datetime_idx, lead_time_idx], 
        aspect='equal', 
        zmin=0, 
        zmax=1200
        )
    fig.update_coloraxes(colorbar_orientation='h')
    fig.update_layout(
        width=500,
        coloraxis_colorbar=dict(
        title="GHI [W/m2]",
        ))
    return fig


def main_app(DS):

    st.set_page_config(layout="wide")
    st.header('GHI Forecasting Model Comparison')
    st.write(f'''
             {DS.attrs["title"]}, 
             {DS.attrs["year"]} 
             {DS.attrs["month"]}. 
             Data was provided by
             [Solargis](https://solargis.com).
             ''')
    st.sidebar.subheader('Select data for 2D analysis')
    
    datetime_idx_start = DS.observed.reference_time[0].values
    datetime_idx_end = DS.observed.reference_time[-1].values
    lead_time_values = DS.model_1.lead_time.values
    latitude_values = DS.model_1.latitude.values
    longitude_values = DS.model_1.longitude.values
    
    date = st.sidebar.date_input(
        'Date:',
        value=pd.to_datetime(datetime_idx_start),
        min_value=pd.to_datetime(datetime_idx_start),
        max_value=pd.to_datetime(datetime_idx_end),
        )
    # times = np.array([datetime.datetime.combine(date, datetime.time()) 
    #                   + datetime.timedelta(minutes = i*10*5 + 20) 
    #                   for i in range(29)])
    times = pd.to_datetime(DS.observed.reference_time.values).to_pydatetime()
    datetime_idx = st.sidebar.selectbox(
        'Time:', 
        options=times[3::5],
        index=0,
        )
    lead_time_options = pd.to_timedelta(lead_time_values).astype(str).map(lambda x: x[7:])
    lead_time_idx = st.sidebar.selectbox(
        'Lead time:', 
        options=lead_time_options,
        index=2
        )
    
    st.sidebar.markdown("""---""")
    st.sidebar.subheader('Select data for 1D analysis')
    lat = st.sidebar.slider('Latitude', 
                            min_value=round(latitude_values[-1],1), 
                            max_value=round(latitude_values[0],1),
                            value=-26.0
                            )
    lon = st.sidebar.slider('Longitude', 
                            min_value=round(longitude_values[0],1), 
                            max_value=round(longitude_values[-1],1),
                            value=-71.0)
    lead_time_options = pd.to_timedelta(lead_time_values).astype(str).map(lambda x: x[7:])
    lead = st.sidebar.selectbox(
        'Lead time:', 
        options=lead_time_options,
        index=6,
        key='1D'
        )

    col1, col2 = st.columns(2)
    col1.subheader('Model 1')
    col2.subheader('Model 2')
    col1.plotly_chart(figure_2D(DS.model_1, datetime_idx, lead_time_idx))
    col2.plotly_chart(figure_2D(DS.model_2, datetime_idx, lead_time_idx))
    
    if pd.to_datetime(datetime_idx + pd.to_timedelta(lead_time_idx)) > datetime_idx_end:
        st.warning('No data available, choose shorter lead time.')
    else:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        MAE_value_m1 = MAE(DS.observed.loc[datetime_idx + pd.to_timedelta(lead_time_idx)], 
                           DS.model_1.loc[datetime_idx, lead_time_idx])
        MAE_value_m2 = MAE(DS.observed.loc[datetime_idx + pd.to_timedelta(lead_time_idx)], 
                           DS.model_2.loc[datetime_idx, lead_time_idx])
        RMSE_value_m1 = RMSE(DS.observed.loc[datetime_idx + pd.to_timedelta(lead_time_idx)], 
                             DS.model_1.loc[datetime_idx, lead_time_idx])
        RMSE_value_m2 = RMSE(DS.observed.loc[datetime_idx + pd.to_timedelta(lead_time_idx)], 
                             DS.model_2.loc[datetime_idx, lead_time_idx])
        kpi1.metric(
            label="📊 Mean absolute error",
            value=f"{MAE_value_m1:.1f} W/m2",
            delta=f"{MAE_value_m1 - MAE_value_m2:.1f} W/m2",
            delta_color="inverse"
            )
        kpi2.metric(
            label="📉 Root mean square error",
            value=f"{RMSE_value_m1:.1f} W/m2",
            delta=f"{RMSE_value_m1 - RMSE_value_m2:.1f} W/m2",
            delta_color="inverse"
            )
        kpi3.metric(
            label="📊 Mean absolute error",
            value=f"{MAE_value_m2:.1f} W/m2",
            delta=f"{MAE_value_m2 - MAE_value_m1:.1f} W/m2",
            delta_color="inverse"
            )
        kpi4.metric(
            label="📉 Root mean square error",
            value=f"{RMSE_value_m2:.1f} W/m2",
            delta=f"{RMSE_value_m2 - RMSE_value_m1:.1f} W/m2",
            delta_color="inverse"
            )
    
    
    
    # 1D analysis
    st.markdown("""---""")
    st.subheader('1D Timeseries Analysis')
    series_m1 = DS.model_1.sel(lead_time=lead,
                               latitude=lat,
                               longitude=lon, 
                               method='nearest').to_series()
    series_m1.index = series_m1.index.shift(1, freq=pd.to_timedelta(lead))

    series_m2 = DS.model_2.sel(lead_time=lead,
                               latitude=lat,
                               longitude=lon, 
                               method='nearest').to_series()
    series_m2.index = series_m2.index.shift(1, freq=pd.to_timedelta(lead))

    series_ob = DS.observed.sel(latitude=lat,
                                longitude=lon, 
                                method='nearest').to_series()

    df = pd.DataFrame({'observed':series_ob,
                       'model1':series_m1,
                       'model2':series_m2})
    
    fig_1D = px.line(df, markers=True, labels=dict(value="GHI [W/m2]"))
    fig_1D.update_layout(hovermode="x unified", width=900, height=500)
    fig_1D.update_traces(hovertemplate='%{y:.1f} [W/m2]', marker_size=5)
    st.plotly_chart(fig_1D)
    
    df = df.dropna()
    col1, col2 = st.columns(2)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    col1.subheader('Model 1')
    col2.subheader('Model 2')
    if pd.to_datetime(datetime_idx + pd.to_timedelta(lead)) > datetime_idx_end:
        st.warning('No data available, choose shorter lead time.')
    else:
        kpi1.metric(
            label="📊 Mean absolute error",
            value=f"{MAE(df.observed, df.model1):.1f} W/m2",
            delta=f"{MAE(df.observed, df.model1) - MAE(df.observed, df.model2):.1f} W/m2",
            delta_color="inverse"
            )
        kpi2.metric(
            label="📉 Root mean square error",
            value=f"{RMSE(df.observed, df.model1):.1f} W/m2",
            delta=f"{RMSE(df.observed, df.model1) - RMSE(df.observed, df.model2):.1f} W/m2",
            delta_color="inverse"
            )
        kpi3.metric(
            label="📊 Mean absolute error",
            value=f"{MAE(df.observed, df.model2):.1f} W/m2",
            delta=f"{MAE(df.observed, df.model2) - MAE(df.observed, df.model1):.1f} W/m2",
            delta_color="inverse"
            )
        kpi4.metric(
            label="📉 Root mean square error",
            value=f"{RMSE(df.observed, df.model2):.1f} W/m2",
            delta=f"{RMSE(df.observed, df.model2) - RMSE(df.observed, df.model1):.1f} W/m2",
            delta_color="inverse"
            )


if __name__ == "__main__":
    DS = xr.open_dataset("datasets/nowcast_validation_sample_goesr_chile_subset_n0.nc")
    main_app(DS)
