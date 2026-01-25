"""Module for plotting sensor time series."""

from collections import defaultdict
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def plot_sensor_timeseries(df: pd.DataFrame,
                           sensor_cols: list,
                           failure_time_start=None,
                           failure_time_end=None,
                           dataset_id=None,
                           show_power=False,
                           show_outliers=False) -> go.Figure:
    """
    Plot multiple sensors + second y-axis for power.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'time_stamp' sensor columns.
    sensor_cols : list
        List of sensor column names to plot.
    failure_time_start : str or pd.Timestamp, optional
        Start time of failure period to highlight.
    failure_time_end : str or pd.Timestamp, optional
        End time of failure period to highlight.
    dataset_id : str or int, optional
        Filter by dataset_id. If None, plot all turbines.
    show_power : bool, optional
        Whether to show power on a second y-axis.
    show_outliers : bool, optional
        Whether to highlight outliers in the plot.
    
    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Filter by dataset_id if provided
    if dataset_id is not None:
        dataset_id = str(dataset_id)
        try:
            df = df.loc[dataset_id].copy()
        except KeyError:
            print(f"No data found for dataset_id '{dataset_id}'")
            return go.Figure()
        if df.empty:
            print(f"No data found for dataset_id '{dataset_id}'")
            return go.Figure()
        title = f"Sensors of (Turbine {dataset_id})"
    else:
        # drop dataset level index if exists
        if isinstance(df.index, pd.MultiIndex):
            dataset_ids = df.index.get_level_values(0).unique()
            if len(dataset_ids) == 1:
                df = df.reset_index(level=0, drop=True)
                title = f"Sensors of (Turbine {dataset_ids[0]})"
            else:
                raise NotImplementedError("Plotting multiple turbines at once is \
                                          not supported when dataset_id is None.")

    # filter out some of the sensor cols
    sensor_cols = [s for s in sensor_cols if s not in ('id', 'active_power_kWh', 'asset_id')]
    sensor_cols = [s for s in sensor_cols if s in df.columns]

    fig = px.line(df, x=df.index, y=sensor_cols, title=title)

    if show_power:
        # add a second y axis for power
        fig.update_layout(
            yaxis2=dict(
                title="Power (kWh)",
                overlaying="y",
                side="right",
                showgrid=False,
                #position=0.85
            )
        )
        # assign power to the second y axis as dots
        fig.add_scatter(
            y=df["active_power_kWh"],
            x=df.index,
            mode="markers",
            marker=dict(size=4, color=df["active_power_kWh"], opacity=0.01),
            name="active_power_kWh",
            yaxis="y2",
        )

    # Add a box for the failure period
    if failure_time_start and failure_time_end:
        fig.add_vrect(
            x0=failure_time_start, x1=failure_time_end,
            fillcolor="LightSalmon", opacity=0.5,
            layer="below", line_width=0,
        )

    if show_outliers:
        outlier_df = df[df["is_outlier"]].copy()
        # Collect points per flagged column and add a marker trace per column
        points = defaultdict(lambda: {"x": [], "y": []})
        for ts, row in outlier_df.iterrows():
            cols = row.get("is_outlier_cols", [])
            if not isinstance(cols, (list, tuple)):
                cols = [cols] if cols else []
            for col in cols:
                if col in outlier_df.columns:
                    points[col]["x"].append(ts)
                    points[col]["y"].append(row[col])

        for col, pts in points.items():
            fig.add_scatter(
                x=pts["x"],
                y=pts["y"],
                mode="markers",
                marker=dict(size=6, color="black", symbol="x"),
                name=f"Outliers: {col}",
            )
    # add vertical line for start of prediction data
    #if dataset_id is not None:
    #    prediction_start_time = df[df["train_test"] == "prediction"].index.min()
    #    fig.add_vline(
    #        x=prediction_start_time,
    #        line=dict(color="Blue", width=2, dash="dash"),
    #     )
    return fig
