# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "polars==1.36.1",
#     "google-api-python-client",
#     "google-auth-httplib2",
#     "numpy==2.4.0",
#     "python-dotenv==1.2.1",
#     "clerk-backend-api==4.2.0",
#     "anthropic==0.75.0",
#     "pandas==2.3.3",
#     "altair==6.0.0",
#     "openai==2.14.0",
#     "scikit-learn==1.8.0",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.19.0"
app = marimo.App(sql_output="polars")

async with app.setup(hide_code=True):
    import marimo as mo
    import sys

    if "pyodide" in sys.modules:
        import micropip

        await micropip.install("polars")

    import altair as alt
    import json
    import numpy as np
    import polars as pl

    from dataclasses import dataclass
    from pathlib import Path
    from sklearn.linear_model import HuberRegressor


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## PyPI Downloads

    Charted below are weekly downloads [from
    PyPI](https://pypistats.org/packages/marimo). This does not track downloads via
    `conda` or other indices. (We saw a weird spike in Oct 2025 from South
    Africa.)
    """)
    return


@app.cell
def _(fit_window, load_pypi_data):
    pypi_data = load_pypi_data(fit_window.value)
    pypi_data["chart"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Zoom out on the above chart to see projected growth. You can adjust the window size of the exponential fit with the slider below.
    """)
    return


@app.cell
def _():
    fit_window = mo.ui.slider(
        4, 52, value=24, label="Exponential fit window (weeks)"
    )
    fit_window
    return (fit_window,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix
    """)
    return


@app.function
def read_ndjson_file(filepath: Path) -> pl.DataFrame:
    """pl.read_ndjson does not work in Pyodide ..."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pl.DataFrame(data)


@app.function
def compute_exponential_fit(
    values: np.ndarray,
    fit_window: int | None,
) -> tuple[HuberRegressor, int]:
    """Compute a robust exponential fit on time series data.

    Fit y = ab^x via log(y) = log(a) + xlog(b), with a Huber loss.
    """
    fit_window = len(values) if fit_window is None else fit_window
    X = np.arange(fit_window).reshape(-1, 1)
    return HuberRegressor().fit(X, np.log(values[-fit_window:])), fit_window


@app.function
def create_pypi_chart(
    df_pypi_complete: pl.DataFrame,
    fit_window: int | None = 12,
) -> alt.Chart:
    """Create PyPI downloads chart with actual data and exponential fit."""
    dates = df_pypi_complete["date"].to_numpy()
    downloads = df_pypi_complete["pypi_weekly"].to_numpy()

    regressor, fit_window = compute_exponential_fit(downloads, fit_window)
    # PyPI data is weekly, so growth rate is per week
    growth_rate = (np.exp(regressor.coef_[0]) - 1) * 100
    dates_fit = np.concatenate(
        [
            dates[-fit_window:],
            dates[-1] + 7 * np.arange(1, 53, dtype="timedelta64[D]"),
        ]
    )
    X_predict = np.arange(len(dates_fit)).reshape(-1, 1)
    df_fit = pl.DataFrame(
        {
            "date": dates_fit,
            "fitted_value": np.exp(regressor.predict(X_predict)),
            "growth_rate": f"{growth_rate:+.1f}%/wk",
        }
    )

    return create_single_line_chart(
        df_pypi_complete,
        df_fit,
        actual_col="pypi_weekly",
        y_label="Weekly Downloads",
        title="PyPI downloads",
    )


@app.cell
def _(Data):
    def load_pypi_data(fit_window: int | None = None) -> Data:
        df_pypi = (
            read_ndjson_file("pypi.jsonl")
            .rename(mapping={"last_day": "pypi"})
            .with_columns(
                pl.col("date").str.to_date(),
            )
            .with_columns(pl.col("date").dt.truncate("1w").alias("week"))
            .group_by("week")
            .agg(pl.col("pypi").sum().alias("pypi_weekly"))
            .sort("week")
            .rename({"week": "date"})
        )

        # Exclude the latest incomplete week,
        # and compare the last two complete weeks
        df_pypi_complete = df_pypi.head(-1)

        chart = create_pypi_chart(
            df_pypi_complete=df_pypi_complete,
            fit_window=fit_window,
        )

        return dict(df=df_pypi_complete, chart=chart)
    return (load_pypi_data,)


@app.function
def create_single_line_chart(
    df_actual: pl.DataFrame,
    df_fit: pl.DataFrame,
    actual_col: str,
    y_label: str,
    title: str,
) -> alt.Chart:
    """Create a time series chart with actual data and fitted line.

    Args:
        df_actual: DataFrame with actual data (must have 'date' column)
        df_fit: DataFrame with fitted data (must have 'date', 'fitted_value', 'growth_rate')
        actual_col: Name of the column in df_actual to plot
        y_label: Label for y-axis
        title: Chart title

    Returns:
        Layered Altair chart with actual and fitted data
    """
    dates = df_actual["date"].to_numpy()
    values = df_actual[actual_col].to_numpy()

    # Set default zoom to show up to 4 weeks after the last data point
    default_end_date = dates[-1] + np.timedelta64(4 * 7, "D")

    # Set reasonable y-axis scale based on actual data with some headroom
    max_actual = values.max()
    default_y_max = max_actual * 1.3

    # Create base chart with actual data
    actual_chart = (
        alt.Chart(df_actual)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "date:T",
                scale=alt.Scale(
                    domain=[dates[0].astype(str), default_end_date.astype(str)]
                ),
            ),
            y=alt.Y(
                f"{actual_col}:Q",
                title=y_label,
                scale=alt.Scale(domain=[0, default_y_max]),
            ),
            color=alt.value("#1f77b4"),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip(f"{actual_col}:Q", title=y_label, format=",.0f"),
            ],
        )
    )

    # Create fitted line chart
    fit_chart = (
        alt.Chart(df_fit)
        .mark_line(strokeDash=[5, 5])
        .encode(
            x=alt.X(
                "date:T",
                scale=alt.Scale(
                    domain=[dates[0].astype(str), default_end_date.astype(str)]
                ),
            ),
            y=alt.Y(
                "fitted_value:Q",
                scale=alt.Scale(domain=[0, default_y_max]),
            ),
            color=alt.Color(
                "growth_rate:N",
                scale=alt.Scale(range=["#ff7f0e"]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip(
                    "fitted_value:Q", title=f"Fitted {y_label}", format=",.0f"
                ),
                alt.Tooltip("growth_rate:N", title="Growth Rate"),
            ],
        )
    )

    return (actual_chart + fit_chart).properties(title=title).interactive()


if __name__ == "__main__":
    app.run()
