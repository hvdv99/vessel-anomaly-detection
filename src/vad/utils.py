import pandas as pd
import geopandas as gpd
from datetime import datetime
from shapely.geometry import Point

def create_geometry_column(df: pd.DataFrame,
                           latitude_col_name='latitude',
                           longitude_col_name='longitude') -> pd.DataFrame:

    tmp_df = df.copy()

    tmp_df['geometry'] = tmp_df[[latitude_col_name, longitude_col_name]].apply(
        (lambda x: Point(x[longitude_col_name], x[latitude_col_name])),
        axis=1
    )

    df['geometry'] = tmp_df['geometry']

    return df


def df_to_gdf_time_index(df: pd.DataFrame, time_col_name='time', geometry_col_name='geometry') -> gpd.GeoDataFrame:

    if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
        df = df.rename_axis('time', axis='index')
        return gpd.GeoDataFrame(
            df,
            geometry=geometry_col_name,
            crs='EPSG:4326')

    if (time_col_name not in df.columns) or (geometry_col_name not in df.columns):
        raise ValueError(f'{time_col_name} or {geometry_col_name} not in dataframe')

    return gpd.GeoDataFrame(
        df,
        geometry=geometry_col_name,
        crs='EPSG:4326').set_index(time_col_name)

def seconds_to_timestamp(epoch_seconds):
    
    if not isinstance(epoch_seconds, int):
        raise TypeError("The input must be an integer representing seconds since the epoch.")

    if epoch_seconds < 0:
        raise ValueError("Negative timestamps are invalid (unless representing times before 1970).")

    # Convert the given seconds since the epoch into a datetime object
    return datetime.utcfromtimestamp(epoch_seconds)
