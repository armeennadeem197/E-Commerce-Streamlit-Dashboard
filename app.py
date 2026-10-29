from __future__ import annotations

import hashlib
import html
from datetime import date, time
from io import BytesIO, StringIO
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = APP_DIR / "ecommerce_sales_data.csv"
MAX_FILTER_OPTIONS = 80
PLOT_CONFIG = {"displaylogo": False, "responsive": True}
CHART_COLORS = [
    "#0F9D8A",  # teal
    "#3B82F6",  # blue
    "#0F766E",  # deep teal
    "#2563EB",  # royal blue
    "#60A5FA",  # soft blue
    "#14B8A6",  # aqua teal
    "#64748B",  # slate
]
CORRELATION_COLORS = [
    [0.00, "#1D4ED8"],
    [0.25, "#93C5FD"],
    [0.50, "#FFFFFF"],
    [0.75, "#5EEAD4"],
    [1.00, "#0F9D8A"],
]
MONTH_TEXT_VALUES = {
    "january", "jan", "february", "feb", "march", "mar", "april", "apr",
    "may", "june", "jun", "july", "jul", "august", "aug", "september",
    "sep", "sept", "october", "oct", "november", "nov", "december", "dec",
}


st.set_page_config(
    page_title="E-Commerce Store Insights",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_styles() -> None:
    """Apply the portfolio dashboard theme without changing Streamlit's behavior."""
    st.markdown(
        """
        <style>
        :root {
            --page: #F5F7FA;
            --surface: #FFFFFF;
            --navy: #14213D;
            --teal: #0F9D8A;
            --blue: #3B82F6;
            --slate: #475569;
            --muted: #64748B;
            --border: #E2E8F0;
        }

        body {
            background: var(--page);
            color: var(--slate);
        }

        .hero-banner {
            position: relative;
            overflow: hidden;
            margin: .25rem 0 1.25rem;
            padding: 1.45rem 1.65rem;
            border: 1px solid var(--border);
            border-radius: 24px;
            background: var(--surface);
            box-shadow: 0 12px 30px rgba(20, 33, 61, .08), inset 0 1px 0 rgba(255,255,255,.85);
        }

        .hero-banner::after {
            content: "";
            position: absolute;
            width: 9rem;
            height: 9rem;
            right: -3rem;
            bottom: -4rem;
            border-radius: 50%;
            background: rgba(15, 157, 138, .11);
            filter: blur(8px);
        }

        .hero-eyebrow {
            color: var(--teal);
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .16em;
            text-transform: uppercase;
        }

        .hero-status {
            display: inline-block;
            margin-left: .65rem;
            padding: .2rem .55rem;
            border: 1px solid rgba(15, 157, 138, .24);
            border-radius: 999px;
            color: var(--teal);
            background: #ECFDF8;
            letter-spacing: .08em;
            font-size: .62rem;
        }

        .hero-content {
            position: relative;
            z-index: 1;
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1.25rem;
            margin-top: .6rem;
        }

        .hero-title {
            color: var(--navy);
            font-size: clamp(1.65rem, 3vw, 2.6rem);
            font-weight: 800;
            letter-spacing: -.045em;
            line-height: 1.05;
        }

        .hero-subtitle {
            max-width: 42rem;
            margin-top: .55rem;
            color: var(--slate);
            font-size: .95rem;
        }

        .hero-badge {
            min-width: 10.5rem;
            padding: .75rem .9rem;
            border: 1px solid var(--border);
            border-radius: 16px;
            background: #F5F7FA;
            color: var(--slate);
            text-align: right;
        }

        .hero-badge-label {
            display: block;
            color: var(--teal);
            font-size: .65rem;
            font-weight: 800;
            letter-spacing: .12em;
        }

        .hero-badge strong {
            display: block;
            margin: .12rem 0;
            color: var(--navy);
            font-size: 1.45rem;
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: .7rem;
            margin: .2rem 0 .55rem;
        }

        .sidebar-brand-mark {
            display: grid;
            width: 2.25rem;
            height: 2.25rem;
            place-items: center;
            border-radius: 12px;
            color: #FFFFFF;
            background: var(--teal);
            box-shadow: 0 7px 18px rgba(15, 157, 138, .18);
            font-size: 1.3rem;
            font-weight: 900;
        }

        .sidebar-brand-name {
            color: #FFFFFF;
            font-size: 1.1rem;
            font-weight: 800;
            letter-spacing: -.02em;
        }

        .sidebar-brand-subtitle {
            color: #CBD5E1;
            font-size: .72rem;
        }

        .st-key-sidebar_filter_form {
            margin-top: .5rem;
            padding: .85rem .8rem .75rem;
            border: 1px solid rgba(255, 255, 255, .14);
            border-radius: 18px;
            background: #1E3256;
            box-shadow: 0 12px 24px rgba(4, 13, 30, .22);
        }

        .st-key-sidebar_filter_form label,
        [class*="st-key-navigation"] label {
            color: #F8FAFC;
        }

        .st-key-sidebar_filter_form button {
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }

        .st-key-sidebar_filter_form button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 18px rgba(15, 157, 138, .16);
            border-color: rgba(15, 157, 138, .48);
        }

        [class*="st-key-navigation"] {
            margin-top: .8rem;
            padding: .7rem .65rem .55rem;
            border: 1px solid rgba(255, 255, 255, .14);
            border-radius: 16px;
            background: #1E3256;
            box-shadow: 0 10px 22px rgba(4, 13, 30, .18);
        }

        [class*="st-key-navigation"] [role="radiogroup"] {
            gap: .15rem;
        }

        [class*="st-key-navigation"] label {
            border-radius: 10px;
            transition: background .16s ease, color .16s ease, transform .16s ease;
        }

        [class*="st-key-navigation"] label:hover {
            background: rgba(15, 157, 138, .18);
            color: #8FE3D4;
            transform: translateX(2px);
        }

        .st-key-kpi-card-rows-after-filters,
        .st-key-kpi-card-total-revenue,
        .st-key-kpi-card-total-profit,
        .st-key-kpi-card-total-units-sold,
        .st-key-kpi-card-average-customer-rating,
        .st-key-kpi-card-profit-margin,
        .st-key-kpi-card-roas,
        .st-key-kpi-card-columns,
        .st-key-kpi-card-missing-cells,
        .st-key-kpi-card-duplicate-rows,
        .st-key-kpi-card-numerical-columns,
        [class*="st-key-quality-kpi-"],
        [class*="st-key-inline-kpi-"] {
            min-height: 6.65rem;
            padding: .35rem .3rem .05rem;
            border: 1px solid var(--border);
            border-radius: 17px;
            background: var(--surface);
            box-shadow: 0 8px 22px rgba(20, 33, 61, .08), inset 0 1px 0 rgba(255,255,255,.85);
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }

        .st-key-kpi-card-rows-after-filters:hover,
        .st-key-kpi-card-total-revenue:hover,
        .st-key-kpi-card-total-profit:hover,
        .st-key-kpi-card-total-units-sold:hover,
        .st-key-kpi-card-average-customer-rating:hover,
        .st-key-kpi-card-profit-margin:hover,
        .st-key-kpi-card-roas:hover,
        .st-key-kpi-card-columns:hover,
        .st-key-kpi-card-missing-cells:hover,
        .st-key-kpi-card-duplicate-rows:hover,
        .st-key-kpi-card-numerical-columns:hover,
        [class*="st-key-quality-kpi-"]:hover,
        [class*="st-key-inline-kpi-"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 14px 28px rgba(20, 33, 61, .13), 0 0 0 1px rgba(15, 157, 138, .12);
        }

        .st-key-kpi-card-rows-after-filters { border-top: 3px solid #3B82F6; }
        .st-key-kpi-card-total-revenue { border-top: 3px solid #0F9D8A; }
        .st-key-kpi-card-total-profit { border-top: 3px solid #2563EB; }
        .st-key-kpi-card-total-units-sold { border-top: 3px solid #14B8A6; }
        .st-key-kpi-card-average-customer-rating { border-top: 3px solid #3B82F6; }
        .st-key-kpi-card-profit-margin { border-top: 3px solid #0F766E; }
        .st-key-kpi-card-roas { border-top: 3px solid #2563EB; }

        .section-caption {
            color: var(--muted);
            font-size: .86rem;
        }

        [class*="st-key-plot-card-"] {
            padding: .35rem .45rem .1rem;
            border: 1px solid var(--border);
            border-radius: 16px;
            background: var(--surface);
            box-shadow: 0 8px 20px rgba(20, 33, 61, .06);
        }

        [class*="st-key-content-card-"] {
            padding: .8rem .9rem .55rem;
            border: 1px solid var(--border);
            border-radius: 16px;
            background: var(--surface);
            box-shadow: 0 8px 20px rgba(20, 33, 61, .06);
        }

        h1, h2, h3, h4 {
            color: var(--navy);
            letter-spacing: -.035em;
        }

        .st-key-sidebar_filter_form [role="combobox"],
        .st-key-sidebar_filter_form input {
            border-color: rgba(255, 255, 255, .20);
            background: #14213D;
            color: #FFFFFF;
            border-radius: 10px;
        }

        .st-key-sidebar_filter_form [role="combobox"]:focus-within,
        .st-key-sidebar_filter_form input:focus {
            border-color: var(--teal);
            box-shadow: 0 0 0 2px rgba(15, 157, 138, .12);
        }

        @media (max-width: 800px) {
            .hero-content { align-items: flex-start; flex-direction: column; }
            .hero-badge { width: 100%; text-align: left; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_styles()


def unique_columns(columns: Iterable[object]) -> list[str]:
    """Return readable, unique column names without losing the source order."""
    counts: dict[str, int] = {}
    result: list[str] = []
    for raw_name in columns:
        name = str(raw_name).strip() or "Unnamed column"
        count = counts.get(name, 0)
        counts[name] = count + 1
        result.append(name if count == 0 else f"{name}_{count}")
    return result


@st.cache_data(show_spinner=False, max_entries=12)
def read_csv_bytes(contents: bytes) -> pd.DataFrame:
    """Read a CSV with a few common encodings and useful error messages."""
    if not contents:
        raise ValueError("The uploaded file is empty.")

    last_decode_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            frame = pd.read_csv(BytesIO(contents), encoding=encoding)
            frame.columns = unique_columns(frame.columns)
            return frame
        except UnicodeDecodeError as error:
            last_decode_error = error
        except pd.errors.EmptyDataError as error:
            raise ValueError("The CSV contains no data.") from error
        except pd.errors.ParserError as error:
            raise ValueError(
                "The CSV could not be parsed. Check that the file uses a consistent delimiter and quoting."
            ) from error

    raise ValueError("The CSV encoding could not be detected.") from last_decode_error


def dataframe_signature(contents: bytes, source_name: str) -> str:
    return hashlib.sha256(source_name.encode("utf-8") + contents).hexdigest()


def initialise_dataset(frame: pd.DataFrame, source_name: str, signature: str) -> None:
    """Reset per-dataset state only when the source file really changes."""
    if st.session_state.get("dataset_signature") == signature:
        return

    st.session_state["dataset_signature"] = signature
    st.session_state["source_name"] = source_name
    st.session_state["original_df"] = frame.copy(deep=True)
    st.session_state["clean_df"] = frame.copy(deep=True)
    st.session_state["cleaning_log"] = []
    st.session_state["derived_features"] = []
    st.session_state["applied_filters"] = {}
    st.session_state["filter_widget_keys"] = []

    for key in list(st.session_state):
        if str(key).startswith(("filter_cat_", "filter_num_", "filter_date_")):
            del st.session_state[key]


def clean_text(value: object) -> str:
    return str(value).strip().lower()


def normalise_name(value: object) -> str:
    return "".join(character for character in clean_text(value) if character.isalnum())


def semantic_score(column: str, hints: Iterable[str]) -> int:
    normalized = normalise_name(column)
    score = 0
    for hint in hints:
        token = normalise_name(hint)
        if normalized == token:
            score += 5
        elif token and token in normalized:
            score += 2
    return score


def suggest_column(columns: Iterable[str], hints: Iterable[str], exclude: Iterable[str] = ()) -> str | None:
    excluded = set(exclude)
    candidates = [(semantic_score(column, hints), column) for column in columns if column not in excluded]
    candidates = [item for item in candidates if item[0] > 0]
    return max(candidates)[1] if candidates else None


def parse_datetime_series(series: pd.Series, fmt: str = "Auto") -> pd.Series:
    """Parse date/time data without allowing one bad value to crash the app."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    values = series.astype("string").str.strip()
    if fmt == "%H:%M:%S":
        return pd.to_timedelta(values, errors="coerce")
    if fmt not in {"Auto", "Date/time"}:
        try:
            return pd.to_datetime(values, format=fmt, errors="coerce")
        except (TypeError, ValueError):
            return pd.to_datetime(values, errors="coerce")

    try:
        return pd.to_datetime(values, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(values, errors="coerce")


def infer_column_types(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Infer useful column groups while leaving the user's data unchanged."""
    numeric: list[str] = []
    numeric_like: list[str] = []
    categorical: list[str] = []
    date_columns: list[str] = []
    time_columns: list[str] = []

    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric.append(column)
            numeric_like.append(column)
        else:
            converted = pd.to_numeric(series, errors="coerce")
            non_null = series.notna().sum()
            numeric_ratio = converted.notna().sum() / non_null if non_null else 0
            if numeric_ratio >= 0.85 and converted.nunique(dropna=True) > 2:
                numeric_like.append(column)

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
        ):
            categorical.append(column)

        normalized = normalise_name(column)
        date_hint = any(token in normalized for token in ("date", "datetime", "timestamp", "day"))
        time_hint = any(token in normalized for token in ("time", "hour", "minute"))
        month_values = series.dropna().astype(str).str.strip().str.lower()
        is_month_only = (
            "month" in normalized
            and not month_values.empty
            and month_values.isin(MONTH_TEXT_VALUES).all()
        )
        if is_month_only:
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            date_columns.append(column)
            continue

        parsed_date = parse_datetime_series(series)
        parse_ratio = parsed_date.notna().sum() / series.notna().sum() if series.notna().any() else 0
        sample_text = series.dropna().astype(str).head(100)
        looks_like_time = (
            len(sample_text) > 0
            and sample_text.str.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?").mean() >= 0.7
        )
        if looks_like_time and (time_hint or not date_hint):
            time_columns.append(column)
        elif parse_ratio >= 0.7 and (date_hint or not pd.api.types.is_numeric_dtype(series)):
            date_columns.append(column)

    return {
        "numeric": list(dict.fromkeys(numeric)),
        "numeric_like": list(dict.fromkeys(numeric_like)),
        "categorical": list(dict.fromkeys(categorical)),
        "date": list(dict.fromkeys(date_columns)),
        "time": list(dict.fromkeys(time_columns)),
    }


def numeric_frame(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    columns = list(columns) if columns is not None else infer_column_types(frame)["numeric_like"]
    result = pd.DataFrame(index=frame.index)
    for column in columns:
        if column in frame.columns:
            result[column] = pd.to_numeric(frame[column], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def missing_summary(frame: pd.DataFrame) -> pd.DataFrame:
    missing = frame.isna().sum()
    return pd.DataFrame(
        {
            "column": frame.columns,
            "dtype": frame.dtypes.astype(str).values,
            "non_null": frame.notna().sum().values,
            "missing": missing.values,
            "missing_percent": (missing / max(len(frame), 1) * 100).round(2).values,
            "unique_values": [frame[column].nunique(dropna=True) for column in frame.columns],
        }
    ).sort_values(["missing", "column"], ascending=[False, True], ignore_index=True)


def info_text(frame: pd.DataFrame) -> str:
    buffer = StringIO()
    frame.info(buf=buffer)
    return buffer.getvalue()


def describe_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = numeric_frame(frame)
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.agg(["count", "mean", "median", "std", "min", "max"]).T
    return summary.round(3)


def describe_categorical(frame: pd.DataFrame) -> pd.DataFrame:
    columns = infer_column_types(frame)["categorical"]
    if not columns:
        return pd.DataFrame()
    rows = []
    for column in columns:
        values = frame[column].dropna()
        rows.append(
            {
                "column": column,
                "unique_values": values.nunique(),
                "most_common": values.mode().iloc[0] if not values.mode().empty else None,
                "frequency": int(values.value_counts().iloc[0]) if not values.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def csv_download(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def log_action(message: str) -> None:
    st.session_state.setdefault("cleaning_log", []).append(message)


def outlier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in infer_column_types(frame)["numeric_like"]:
        values = numeric_series(frame, column).dropna()
        if values.empty or values.nunique() < 2:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            lower, upper = q1, q3
        else:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (values < lower) | (values > upper)
        rows.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(mask.sum()),
                "outlier_percent": round(float(mask.mean() * 100), 2),
            }
        )
    return pd.DataFrame(rows)


def outlier_bounds(frame: pd.DataFrame, column: str) -> tuple[float, float] | None:
    values = numeric_series(frame, column).dropna()
    if values.empty:
        return None
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def category_options(frame: pd.DataFrame) -> list[str]:
    return infer_column_types(frame)["categorical"]


def date_candidates(frame: pd.DataFrame) -> list[str]:
    return infer_column_types(frame)["date"]


def filter_column_key(prefix: str, column: str) -> str:
    digest = hashlib.md5(column.encode("utf-8")).hexdigest()[:8]
    return f"filter_{prefix}_{digest}"


def remember_filter_widget(key: str) -> None:
    keys = st.session_state.setdefault("filter_widget_keys", [])
    if key not in keys:
        keys.append(key)


def filter_value_text(value: object) -> str:
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value)


def prioritised_category_columns(frame: pd.DataFrame) -> list[str]:
    columns = category_options(frame)
    ordered: list[str] = []
    for hints in (["year"], ["month"], ["category", "product", "segment"], ["region", "country", "state", "location"]):
        candidate = suggest_column(columns, hints)
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    ordered.extend(column for column in columns if column not in ordered)
    return ordered[:8]


def year_filter_column(frame: pd.DataFrame) -> str | None:
    return suggest_column(infer_column_types(frame)["numeric_like"], ["year", "fiscal year", "calendar year"])


def time_seconds(series: pd.Series) -> pd.Series:
    durations = pd.to_timedelta(series.astype("string").str.strip(), errors="coerce")
    return durations.dt.total_seconds()


def seconds_to_time(seconds: float) -> time:
    seconds = max(0, min(86399, int(seconds)))
    return time(seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def reset_sidebar_filter_state() -> None:
    for key in st.session_state.get("filter_widget_keys", []):
        st.session_state.pop(key, None)
    st.session_state["filter_widget_keys"] = []
    st.session_state["applied_filters"] = {}


def filter_frame(frame: pd.DataFrame, state: dict) -> tuple[pd.DataFrame, list[str]]:
    mask = pd.Series(True, index=frame.index)
    notes: list[str] = []

    for column, selected in state.get("categorical", {}).items():
        if column not in frame.columns:
            continue
        options = frame[column].dropna().astype(str).unique().tolist()
        if selected and len(selected) < len(options):
            mask &= frame[column].astype(str).isin(selected)
            notes.append(f"{column}: {len(selected)} selected")

    for column, query in state.get("categorical_search", {}).items():
        if column in frame.columns and query:
            mask &= frame[column].astype("string").str.contains(query, case=False, na=False)
            notes.append(f"{column}: contains '{query}'")

    for discrete_filter in (state.get("year", {}), state.get("month", {})):
        if discrete_filter and discrete_filter.get("column") in frame.columns:
            column = discrete_filter["column"]
            values = numeric_series(frame, column).map(filter_value_text)
            options = values.dropna().unique().tolist()
            selected = discrete_filter.get("values", [])
            if selected and len(selected) < len(options):
                mask &= values.isin(selected)
                notes.append(f"{column}: {len(selected)} selected")

    for column, selected_range in state.get("numeric_ranges", {}).items():
        if column not in frame.columns or not selected_range:
            continue
        values = numeric_series(frame, column)
        minimum, maximum = selected_range
        full_minimum, full_maximum = values.min(), values.max()
        if pd.notna(full_minimum) and (minimum > full_minimum or maximum < full_maximum):
            mask &= values.between(minimum, maximum).fillna(False)
            notes.append(f"{column}: range applied")

    for column, selected_range in state.get("date_ranges", {}).items():
        if column not in frame.columns or not selected_range:
            continue
        parsed = parse_datetime_series(frame[column])
        if pd.api.types.is_datetime64_any_dtype(parsed):
            start, end = selected_range
            full_start, full_end = parsed.min().date(), parsed.max().date()
            if start > full_start or end < full_end:
                mask &= parsed.dt.date.between(start, end).fillna(False)
                notes.append(f"{column}: date range applied")

    for column, selected_range in state.get("time_ranges", {}).items():
        if column not in frame.columns or not selected_range:
            continue
        values = time_seconds(frame[column])
        valid = values.dropna()
        if not valid.empty:
            full_start, full_end = valid.min(), valid.max()
            start, end = selected_range
            if start > full_start or end < full_end:
                mask &= values.between(start, end).fillna(False)
                notes.append(f"{column}: time range applied")

    return frame.loc[mask].copy(), notes


def apply_sidebar_filters(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Render staged, schema-aware filters and return the applied active slice."""
    pending: dict = {
        "categorical": {},
        "categorical_search": {},
        "year": {},
        "month": {},
        "numeric_ranges": {},
        "date_ranges": {},
        "time_ranges": {},
    }
    with st.sidebar:
        st.divider()
        st.subheader("Filters")
        st.caption("Choose values, ranges, or dates, then apply them to every KPI, chart, table, and insight.")
        with st.form("sidebar_filter_form", border=False):
            for column in prioritised_category_columns(frame):
                options = sorted(frame[column].dropna().astype(str).unique().tolist())
                key = filter_column_key("category", column)
                remember_filter_widget(key)
                if len(options) <= MAX_FILTER_OPTIONS:
                    current = st.session_state.get(key)
                    if current is None:
                        st.session_state[key] = options
                    else:
                        st.session_state[key] = [value for value in current if value in options]
                    pending["categorical"][column] = st.multiselect(
                        f"Filter {column}",
                        options,
                        key=key,
                        help="Clear the selection to include all values.",
                    )
                else:
                    st.session_state.setdefault(key, "")
                    pending["categorical_search"][column] = st.text_input(
                        f"Search values in {column}",
                        key=key,
                        placeholder="Type part of a value",
                    ).strip()

            year_column = year_filter_column(frame)
            if year_column:
                year_options = sorted(
                    numeric_series(frame, year_column).dropna().map(filter_value_text).unique().tolist()
                )
                key = filter_column_key("year", year_column)
                remember_filter_widget(key)
                current = st.session_state.get(key)
                if current is None:
                    st.session_state[key] = year_options
                else:
                    st.session_state[key] = [value for value in current if value in year_options]
                pending["year"] = {
                    "column": year_column,
                    "values": st.multiselect(
                        f"Filter {year_column}",
                        year_options,
                        key=key,
                        help="Year is treated as a discrete filter when it has a manageable number of values.",
                    ),
                }

            month_column = suggest_column(
                infer_column_types(frame)["numeric_like"],
                ["month"],
                exclude=[year_column] if year_column else [],
            )
            if month_column and frame[month_column].nunique(dropna=True) <= 20:
                month_options = sorted(
                    numeric_series(frame, month_column).dropna().map(filter_value_text).unique().tolist()
                )
                key = filter_column_key("month", month_column)
                remember_filter_widget(key)
                current = st.session_state.get(key)
                if current is None:
                    st.session_state[key] = month_options
                else:
                    st.session_state[key] = [value for value in current if value in month_options]
                pending["month"] = {
                    "column": month_column,
                    "values": st.multiselect(
                        f"Filter {month_column}",
                        month_options,
                        key=key,
                        help="Month is treated as a discrete filter when it has a manageable number of values.",
                    ),
                }

            numeric_columns = [
                column
                for column in infer_column_types(frame)["numeric_like"]
                if column not in {year_column, month_column}
            ]
            for column in numeric_columns[:6]:
                values = numeric_series(frame, column).dropna()
                if values.empty or values.min() == values.max():
                    continue
                minimum, maximum = float(values.min()), float(values.max())
                key = filter_column_key("numeric", column)
                remember_filter_widget(key)
                previous = st.session_state.get(key, (minimum, maximum))
                if not isinstance(previous, (tuple, list)) or len(previous) != 2:
                    previous = (minimum, maximum)
                previous = (
                    max(minimum, min(maximum, float(previous[0]))),
                    max(minimum, min(maximum, float(previous[1]))),
                )
                if previous[0] > previous[1]:
                    previous = (minimum, maximum)
                selected_range = st.slider(
                    f"Range: {column}",
                    min_value=minimum,
                    max_value=maximum,
                    value=previous,
                    key=key,
                )
                pending["numeric_ranges"][column] = tuple(selected_range)

            for column in date_candidates(frame)[:3]:
                parsed = parse_datetime_series(frame[column])
                if not pd.api.types.is_datetime64_any_dtype(parsed) or not parsed.notna().any():
                    continue
                minimum_date, maximum_date = parsed.min().date(), parsed.max().date()
                key = filter_column_key("date", column)
                remember_filter_widget(key)
                previous = st.session_state.get(key, (minimum_date, maximum_date))
                if not isinstance(previous, (tuple, list)) or len(previous) != 2:
                    previous = (minimum_date, maximum_date)
                previous = (
                    max(minimum_date, previous[0]),
                    min(maximum_date, previous[1]),
                )
                if previous[0] > previous[1]:
                    previous = (minimum_date, maximum_date)
                st.session_state[key] = previous
                selected_dates = st.date_input(
                    f"Date range: {column}",
                    min_value=minimum_date,
                    max_value=maximum_date,
                    key=key,
                )
                if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
                    pending["date_ranges"][column] = tuple(selected_dates)

            for column in infer_column_types(frame)["time"][:3]:
                values = time_seconds(frame[column]).dropna()
                if values.empty:
                    continue
                minimum_seconds, maximum_seconds = float(values.min()), float(values.max())
                start_key = filter_column_key("time_start", column)
                end_key = filter_column_key("time_end", column)
                remember_filter_widget(start_key)
                remember_filter_widget(end_key)
                st.session_state.setdefault(start_key, seconds_to_time(minimum_seconds))
                st.session_state.setdefault(end_key, seconds_to_time(maximum_seconds))
                start_time = st.time_input(
                    f"Start time: {column}",
                    key=start_key,
                )
                end_time = st.time_input(
                    f"End time: {column}",
                    key=end_key,
                )
                start_seconds = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
                end_seconds = end_time.hour * 3600 + end_time.minute * 60 + end_time.second
                pending["time_ranges"][column] = (start_seconds, end_seconds)

            apply_clicked, reset_clicked = st.columns(2)
            with apply_clicked:
                apply_filters = st.form_submit_button(
                    "Apply filters",
                    type="primary",
                    icon=":material/filter_alt:",
                    width="stretch",
                )
            with reset_clicked:
                reset_filters = st.form_submit_button(
                    "Reset filters",
                    icon=":material/refresh:",
                    on_click=reset_sidebar_filter_state,
                    width="stretch",
                )

        if apply_filters:
            st.session_state["applied_filters"] = pending
        if reset_filters:
            st.session_state["applied_filters"] = {}

        applied = st.session_state.get("applied_filters", {})
        active, filter_notes = filter_frame(frame, applied)
        st.metric("Rows remaining", f"{len(active):,}", border=True)
        st.caption("Cleaning actions apply to the full cleaned dataset; filters apply to the active view.")
    return active, filter_notes


def plotly_style(figure: go.Figure, height: int = 420) -> go.Figure:
    figure.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        colorway=CHART_COLORS,
        font={"color": "#475569", "family": "Inter, Segoe UI, sans-serif"},
        title={"font": {"color": "#14213D", "size": 18}},
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        legend={"orientation": "h", "y": -0.2, "font": {"color": "#475569"}},
        hoverlabel={"namelength": -1},
    )
    figure.update_xaxes(
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
        zerolinecolor="#CBD5E1",
        title_font={"color": "#475569"},
        tickfont={"color": "#475569"},
    )
    figure.update_yaxes(
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
        zerolinecolor="#CBD5E1",
        title_font={"color": "#475569"},
        tickfont={"color": "#475569"},
    )
    return figure


def render_plot(figure: go.Figure, alt: str, height: int = 420) -> None:
    plot_key = f"plot-card-{hashlib.md5(alt.encode('utf-8')).hexdigest()[:10]}"
    with st.container(border=True, key=plot_key):
        st.plotly_chart(
            plotly_style(figure, height),
            width="stretch",
            config=PLOT_CONFIG,
            alt=alt,
        )


def format_number(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.{decimals}f}"


def css_slug(value: str) -> str:
    """Create a stable CSS-friendly key for themed Streamlit containers."""
    return "".join(character if character.isalnum() else "-" for character in value.lower()).strip("-")


def render_kpis(active: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    """Show business KPIs only when their required semantic columns exist."""
    numeric = infer_column_types(active)["numeric_like"]
    revenue_column = suggest_column(numeric, ["revenue", "sales", "amount", "income", "turnover"])
    profit_column = suggest_column(numeric, ["profit", "earnings", "net income", "gain"])
    units_column = suggest_column(numeric, ["units sold", "units", "quantity", "volume", "count"])
    rating_column = suggest_column(numeric, ["customer rating", "rating", "score", "satisfaction"])
    ad_spend_column = suggest_column(numeric, ["advertising spend", "ad spend", "advertising", "marketing spend", "marketing cost"])

    specifications: list[tuple[str, str, str, str]] = [
        ("Rows after filters", f"{len(active):,}", ":material/table_rows:", "All KPIs are calculated from the applied filtered view."),
    ]
    if revenue_column:
        specifications.append(
            (
                "Total revenue",
                format_number(numeric_series(active, revenue_column).sum(), 2),
                ":material/paid:",
                f"Sum of {revenue_column}.",
            )
        )
    if profit_column:
        specifications.append(
            (
                "Total profit",
                format_number(numeric_series(active, profit_column).sum(), 2),
                ":material/trending_up:",
                f"Sum of {profit_column}.",
            )
        )
    if units_column:
        specifications.append(
            (
                "Total units sold",
                format_number(numeric_series(active, units_column).sum(), 0),
                ":material/shopping_cart:",
                f"Sum of {units_column}.",
            )
        )
    if rating_column:
        specifications.append(
            (
                "Average customer rating",
                format_number(numeric_series(active, rating_column).mean(), 2),
                ":material/star:",
                f"Mean of {rating_column}.",
            )
        )
    if revenue_column and profit_column:
        revenue_total = numeric_series(active, revenue_column).sum()
        profit_total = numeric_series(active, profit_column).sum()
        specifications.append(
            (
                "Profit margin",
                f"{profit_total / revenue_total * 100:.1f}%" if revenue_total else "n/a",
                ":material/percent:",
                f"{profit_column} divided by {revenue_column}.",
            )
        )
    if revenue_column and ad_spend_column:
        revenue_total = numeric_series(active, revenue_column).sum()
        ad_spend_total = numeric_series(active, ad_spend_column).sum()
        specifications.append(
            (
                "ROAS",
                f"{revenue_total / ad_spend_total:.2f}x" if ad_spend_total else "n/a",
                ":material/campaign:",
                f"{revenue_column} divided by {ad_spend_column}.",
            )
        )

    if len(specifications) == 1:
        specifications.extend(
            [
                ("Columns", f"{len(active.columns):,}", ":material/view_column:", "Columns in the active dataset."),
                ("Missing cells", f"{int(active.isna().sum().sum()):,}", ":material/error_outline:", "Missing cells in the active view."),
                ("Duplicate rows", f"{int(cleaned.duplicated().sum()):,}", ":material/content_copy:", "Duplicate rows in the working cleaned dataset."),
                ("Numerical columns", f"{len(numeric):,}", ":material/functions:", "Detected numerical columns."),
            ]
        )

    for offset in range(0, len(specifications), 6):
        row = specifications[offset : offset + 6]
        columns = st.columns(len(row))
        for column, (label, value, icon, help_text) in zip(columns, row):
            card_key = f"kpi-card-{css_slug(label)}"
            with column:
                with st.container(key=card_key):
                    st.metric(label, value, icon=icon, help=help_text, border=False, width="stretch")


def render_data_understanding(active: pd.DataFrame, original: pd.DataFrame) -> None:
    st.header("Data understanding", anchor=False)
    st.caption("Inspect the active filtered view and compare it with the untouched uploaded dataset.")

    if active.empty:
        st.warning("The current filters return no rows. Reset filters in the sidebar to continue.")
        return

    with st.container(border=True, key="content-card-data-understanding"):
        st.subheader("Dataset snapshot")
        snapshot = {
            "Rows": len(active),
            "Columns": len(active.columns),
            "Shape (df.shape)": f"({len(active)}, {len(active.columns)})",
            "Original rows": len(original),
            "Original columns": len(original.columns),
            "Memory usage": f"{active.memory_usage(deep=True).sum() / 1024:.1f} KB",
        }
        snapshot_table = pd.DataFrame(
            {"metric": list(snapshot.keys()), "value": [str(value) for value in snapshot.values()]}
        )
        st.table(snapshot_table, border="horizontal", width="content")

    tabs = st.tabs(["Rows", "Columns and types", "Statistics", "Missing values", "Dataset info"])
    with tabs[0]:
        sample_size = st.slider(
            "Random sample size",
            min_value=1,
            max_value=min(20, len(active)),
            value=min(5, len(active)),
            key="understanding_sample_size",
        )
        left, right = st.columns(2)
        with left:
            st.markdown("**First 5 rows (df.head())**")
            st.dataframe(active.head(), hide_index=True, alt="First five rows")
            st.markdown("**Last 5 rows (df.tail())**")
            st.dataframe(active.tail(), hide_index=True, alt="Last five rows")
        with right:
            st.markdown("**First and last rows (df)**")
            head_tail = pd.concat([active.head(3), active.tail(3)]).drop_duplicates()
            st.dataframe(head_tail, hide_index=True, alt="First and last rows")
            st.markdown("**Random sample (df.sample())**")
            st.dataframe(
                active.sample(n=sample_size, random_state=42),
                hide_index=True,
                alt="Random sample of rows",
            )
        with st.expander("Interactive table of the active dataset", expanded=False):
            st.dataframe(active, height=420, hide_index=True, alt="Interactive active dataset table")

    with tabs[1]:
        type_table = missing_summary(active)[["column", "dtype", "non_null", "unique_values"]]
        st.dataframe(type_table, hide_index=True, alt="Column names and data types")
        info = infer_column_types(active)
        left, right = st.columns(2)
        with left:
            st.markdown("**Numerical columns**")
            st.write(info["numeric_like"] or "None detected")
            st.markdown("**Categorical columns**")
            st.write(info["categorical"] or "None detected")
        with right:
            st.markdown("**Date columns**")
            st.write(info["date"] or "None detected")
            st.markdown("**Time columns**")
            st.write(info["time"] or "None detected")

    with tabs[2]:
        numerical_stats = describe_numeric(active)
        categorical_stats = describe_categorical(active)
        if numerical_stats.empty:
            st.info("No numerical columns were detected for descriptive statistics.")
        else:
            st.markdown("**Numerical descriptive statistics**")
            st.dataframe(numerical_stats, alt="Mean median standard deviation minimum and maximum")
        if categorical_stats.empty:
            st.info("No categorical columns were detected.")
        else:
            st.markdown("**Categorical descriptive statistics**")
            st.dataframe(categorical_stats, hide_index=True, alt="Categorical summary statistics")

    with tabs[3]:
        summary = missing_summary(active)
        st.dataframe(summary, hide_index=True, alt="Non-null and missing-value counts")
        st.download_button(
            "Download data quality report",
            data=csv_download(summary),
            file_name="data_quality_report.csv",
            mime="text/csv",
            icon=":material/download:",
        )

    with tabs[4]:
        st.code(info_text(active), language="text")


def update_cleaned(frame: pd.DataFrame, message: str) -> None:
    st.session_state["clean_df"] = frame.copy(deep=True)
    log_action(message)
    st.rerun()


def render_missing_cleaning(cleaned: pd.DataFrame) -> None:
    st.subheader("Missing values and duplicates")
    summary = missing_summary(cleaned)
    missing_rows = summary[summary["missing"] > 0]
    left, right = st.columns(2)
    with left:
        with st.container(key="inline-kpi-missing-cells"):
            st.metric("Missing cells", f"{int(cleaned.isna().sum().sum()):,}", border=False, width="stretch")
        st.dataframe(summary, hide_index=True, alt="Missing-value counts and percentages")
    with right:
        with st.container(key="inline-kpi-duplicate-rows"):
            st.metric("Duplicate rows", f"{int(cleaned.duplicated().sum()):,}", border=False, width="stretch")
        duplicates = cleaned[cleaned.duplicated(keep=False)]
        if duplicates.empty:
            st.success("No duplicate rows detected.")
        else:
            st.dataframe(duplicates.head(200), hide_index=True, alt="Duplicate rows")
            if st.button("Remove duplicate rows", type="primary", icon=":material/delete_sweep:"):
                update_cleaned(cleaned.drop_duplicates().reset_index(drop=True), "Removed duplicate rows.")

    st.divider()
    st.markdown("**Choose a missing-value operation**")
    operation = st.selectbox(
        "Operation",
        ["Drop rows with missing values", "Drop columns with missing values", "Fill numerical values", "Fill categorical values"],
        key="missing_operation",
    )
    available_columns = missing_rows["column"].tolist()
    if not available_columns:
        st.success("The cleaned dataset has no missing values.")
        return

    if operation == "Drop rows with missing values":
        st.caption("This removes any row containing at least one missing value.")
        if st.button("Drop missing rows", type="primary", icon=":material/filter_alt_off:"):
            update_cleaned(cleaned.dropna().reset_index(drop=True), "Dropped rows with missing values.")
    elif operation == "Drop columns with missing values":
        columns = st.multiselect("Columns to drop", available_columns, default=available_columns, key="drop_missing_columns")
        if st.button("Drop selected columns", type="primary", icon=":material/view_column:") and columns:
            update_cleaned(cleaned.drop(columns=columns), f"Dropped columns with missing values: {', '.join(columns)}.")
    elif operation == "Fill numerical values":
        columns = [column for column in available_columns if column in infer_column_types(cleaned)["numeric_like"]]
        if not columns:
            st.info("No missing numerical columns were detected.")
            return
        selected = st.multiselect("Numerical columns", columns, default=columns, key="fill_numeric_columns")
        strategy = st.selectbox("Fill strategy", ["Mean", "Median", "Custom value"], key="numeric_fill_strategy")
        custom_value = st.number_input("Custom numerical value", value=0.0, key="numeric_fill_value") if strategy == "Custom value" else 0.0
        if st.button("Fill numerical values", type="primary", icon=":material/auto_fix_high:") and selected:
            result = cleaned.copy()
            for column in selected:
                values = numeric_series(result, column)
                fill_value = custom_value if strategy == "Custom value" else (values.mean() if strategy == "Mean" else values.median())
                result[column] = values.fillna(fill_value)
            update_cleaned(result, f"Filled numerical missing values using {strategy.lower()}.")
    else:
        columns = [column for column in available_columns if column in infer_column_types(cleaned)["categorical"]]
        if not columns:
            st.info("No missing categorical columns were detected.")
            return
        selected = st.multiselect("Categorical columns", columns, default=columns, key="fill_categorical_columns")
        strategy = st.selectbox("Fill strategy", ["Mode", "Custom value"], key="categorical_fill_strategy")
        custom_value = st.text_input("Custom categorical value", value="Unknown", key="categorical_fill_value") if strategy == "Custom value" else ""
        if st.button("Fill categorical values", type="primary", icon=":material/auto_fix_high:") and selected:
            result = cleaned.copy()
            for column in selected:
                mode = result[column].mode(dropna=True)
                fill_value = custom_value if strategy == "Custom value" else (mode.iloc[0] if not mode.empty else "Unknown")
                result[column] = result[column].fillna(fill_value)
            update_cleaned(result, f"Filled categorical missing values using {strategy.lower()}.")


def render_type_cleaning(cleaned: pd.DataFrame) -> None:
    st.subheader("Data types and date/time conversion")
    info = infer_column_types(cleaned)
    st.dataframe(missing_summary(cleaned)[["column", "dtype"]], hide_index=True, alt="Updated data types")

    numeric_like = info["numeric_like"]
    if numeric_like:
        selected_numeric = st.multiselect(
            "Columns to convert to numeric",
            numeric_like,
            default=[column for column in numeric_like if column not in info["numeric"]],
            key="convert_numeric_columns",
        )
        if st.button("Convert selected columns to numeric", icon=":material/123:") and selected_numeric:
            result = cleaned.copy()
            for column in selected_numeric:
                result[column] = pd.to_numeric(result[column], errors="coerce")
            update_cleaned(result, f"Converted to numeric: {', '.join(selected_numeric)}.")

    candidates = list(dict.fromkeys(info["date"] + info["time"]))
    if not candidates:
        st.info("No date or time columns were confidently detected. You can still select a column below and provide its format.")
        candidates = list(cleaned.columns)
    selected_column = st.selectbox("Date/time column", candidates, key="temporal_column")
    format_options = ["Auto", "Date/time", "%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "Custom format"]
    format_choice = st.selectbox("Input format", format_options, key="temporal_format")
    custom_format = st.text_input("Custom datetime format", value="%Y-%m-%d %H:%M:%S", key="custom_temporal_format") if format_choice == "Custom format" else format_choice
    if st.button("Convert date/time column", type="primary", icon=":material/calendar_month:"):
        parsed = parse_datetime_series(cleaned[selected_column], custom_format)
        valid_count = int(parsed.notna().sum())
        if valid_count == 0:
            st.error("No values could be parsed with that format. The data was not changed.")
        else:
            result = cleaned.copy()
            result[selected_column] = parsed
            invalid_count = len(parsed) - valid_count
            message = f"Converted {selected_column} to {parsed.dtype}."
            if invalid_count:
                st.warning(f"{invalid_count:,} value(s) could not be parsed and became missing.")
            update_cleaned(result, message)


def render_outlier_cleaning(cleaned: pd.DataFrame) -> None:
    st.subheader("Numerical outliers")
    summary = outlier_summary(cleaned)
    if summary.empty:
        st.info("At least two distinct numerical values are needed to calculate IQR outliers.")
        return
    st.dataframe(summary, hide_index=True, alt="IQR outlier summary")

    outlier_columns = summary["column"].tolist()
    column = st.selectbox("Inspect a numerical column", outlier_columns, key="outlier_column")
    bounds = outlier_bounds(cleaned, column)
    if bounds is None:
        st.info("No valid values are available for this column.")
        return
    lower, upper = bounds
    values = numeric_series(cleaned, column)
    outlier_mask = (values < lower) | (values > upper)
    outliers = cleaned.loc[outlier_mask.fillna(False)]

    left, right = st.columns(2)
    with left:
        figure, axis = plt.subplots(figsize=(8, 2.8))
        figure.patch.set_facecolor("#FFFFFF")
        axis.set_facecolor("#FFFFFF")
        sns.boxplot(x=values.dropna(), ax=axis, color="#0F9D8A")
        axis.set_title(f"Seaborn box plot: {column}", color="#14213D")
        axis.set_xlabel(column, color="#475569")
        axis.tick_params(colors="#475569")
        for spine in axis.spines.values():
            spine.set_color("#CBD5E1")
        st.pyplot(figure, clear_figure=True)
        st.caption("The box plot uses the 1.5 × IQR rule to flag points beyond the whiskers.")
    with right:
        with st.container(key="inline-kpi-outlier-rows"):
            st.metric("Outlier rows", f"{len(outliers):,}", border=False, width="stretch")
        st.dataframe(outliers.head(200), hide_index=True, alt="Rows identified as outliers")

    action = st.selectbox("Outlier action", ["Inspect only", "Remove outliers", "Cap outliers"], key="outlier_action")
    if action != "Inspect only" and st.button(f"{action} in {column}", type="primary", icon=":material/tune:"):
        result = cleaned.copy()
        if action == "Remove outliers":
            result = result.loc[~outlier_mask.fillna(False)].reset_index(drop=True)
            update_cleaned(result, f"Removed IQR outliers from {column}.")
        else:
            result[column] = values.clip(lower=lower, upper=upper)
            update_cleaned(result, f"Capped IQR outliers in {column}.")


def render_data_cleaning(original: pd.DataFrame, cleaned: pd.DataFrame, active: pd.DataFrame) -> None:
    st.header("Data cleaning", anchor=False)
    st.caption("Every operation applies to a working copy. The original upload remains unchanged for comparison.")
    original_missing = int(original.isna().sum().sum())
    cleaned_missing = int(cleaned.isna().sum().sum())
    original_duplicates = int(original.duplicated().sum())
    cleaned_duplicates = int(cleaned.duplicated().sum())
    quality_metrics = [
        ("Original rows", f"{len(original):,}"),
        ("Cleaned rows", f"{len(cleaned):,}"),
        ("Rows changed", f"{len(original) - len(cleaned):,}"),
        ("Original missing", f"{original_missing:,}"),
        ("Cleaned missing", f"{cleaned_missing:,}"),
        ("Cleaned duplicates", f"{cleaned_duplicates:,}"),
    ]
    columns = st.columns(len(quality_metrics))
    for column, (label, value) in zip(columns, quality_metrics):
        with column:
            with st.container(key=f"quality-kpi-{css_slug(label)}"):
                st.metric(label, value, border=False, width="stretch")
    if original_duplicates:
        st.caption(f"The original upload contains {original_duplicates:,} duplicate row(s).")

    tabs = st.tabs(["Missing values", "Types and dates", "Outliers", "Updated info and download"])
    with tabs[0]:
        render_missing_cleaning(cleaned)
    with tabs[1]:
        render_type_cleaning(cleaned)
    with tabs[2]:
        render_outlier_cleaning(cleaned)
    with tabs[3]:
        st.subheader("Updated dataset information")
        st.code(info_text(cleaned), language="text")
        st.dataframe(missing_summary(cleaned), hide_index=True, alt="Updated data quality summary")
        download_cleaned, download_filtered = st.columns(2)
        with download_cleaned:
            st.download_button(
                "Download cleaned CSV",
                data=csv_download(cleaned),
                file_name="cleaned_data.csv",
                mime="text/csv",
                type="primary",
                icon=":material/download:",
                width="stretch",
            )
        with download_filtered:
            st.download_button(
                "Download cleaned and filtered CSV",
                data=csv_download(active),
                file_name="cleaned_filtered_data.csv",
                mime="text/csv",
                icon=":material/filter_alt:",
                width="stretch",
            )
        if st.session_state.get("cleaning_log"):
            st.markdown("**Cleaning audit trail**")
            st.table(pd.DataFrame({"action": st.session_state["cleaning_log"]}), border="horizontal")


def month_number(value: object) -> int | None:
    text = clean_text(value)
    month_names = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
        "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
        "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    }
    if text in month_names:
        return month_names[text]
    try:
        number = int(float(text))
        return number if 1 <= number <= 12 else None
    except (TypeError, ValueError):
        return None


def safe_feature_name(value: str, existing: Iterable[str]) -> str:
    base = "".join(character if character.isalnum() else "_" for character in value.strip()).strip("_") or "feature"
    candidate = base
    index = 1
    existing_set = set(existing)
    while candidate in existing_set:
        index += 1
        candidate = f"{base}_{index}"
    return candidate


def render_feature_engineering(cleaned: pd.DataFrame) -> None:
    st.subheader("Feature engineering")
    st.caption("Create reusable date parts, inferred dates, and simple numeric ratios without changing the original upload.")
    info = infer_column_types(cleaned)
    date_columns = info["date"]
    if date_columns:
        date_column = st.selectbox("Date column", date_columns, key="feature_date_column")
        parts = st.multiselect(
            "Parts to extract",
            ["Year", "Month", "Day", "Weekday"],
            default=["Year", "Month", "Day", "Weekday"],
            key="feature_date_parts",
        )
        if st.button("Add date features", type="primary", icon=":material/event:") and parts:
            parsed = parse_datetime_series(cleaned[date_column])
            result = cleaned.copy()
            for part in parts:
                column = safe_feature_name(f"{date_column}_{part.lower()}", result.columns)
                if part == "Year":
                    result[column] = parsed.dt.year
                elif part == "Month":
                    result[column] = parsed.dt.month
                elif part == "Day":
                    result[column] = parsed.dt.day
                else:
                    result[column] = parsed.dt.day_name()
            update_cleaned(result, f"Added date features from {date_column}.")
    else:
        year_column = suggest_column(cleaned.columns, ["year"])
        month_column = suggest_column(cleaned.columns, ["month"])
        if year_column and month_column:
            st.info(f"No date column was detected, but {year_column} and {month_column} can be combined.")
            if st.button("Create inferred date", icon=":material/calendar_add_on:"):
                result = cleaned.copy()
                year_values = pd.to_numeric(result[year_column], errors="coerce")
                month_values = result[month_column].map(month_number)
                result[safe_feature_name("inferred_date", result.columns)] = pd.to_datetime(
                    {"year": year_values, "month": month_values, "day": 1},
                    errors="coerce",
                )
                update_cleaned(result, f"Created an inferred date from {year_column} and {month_column}.")
        else:
            st.info("No date column was detected. Convert one on the Data cleaning page or upload a dataset with a date field.")

    numeric = info["numeric_like"]
    if len(numeric) >= 2:
        st.divider()
        st.markdown("**Create a numeric ratio**")
        numerator = st.selectbox("Numerator", numeric, key="ratio_numerator")
        denominator_options = [column for column in numeric if column != numerator]
        denominator = st.selectbox("Denominator", denominator_options, key="ratio_denominator")
        feature_name = st.text_input("New feature name", value="calculated_ratio", key="ratio_feature_name")
        if st.button("Add ratio feature", icon=":material/calculate:"):
            denominator_values = numeric_series(cleaned, denominator).replace(0, np.nan)
            result = cleaned.copy()
            result[safe_feature_name(feature_name, result.columns)] = numeric_series(cleaned, numerator) / denominator_values
            update_cleaned(result, f"Added a ratio feature using {numerator} and {denominator}.")

    selected_columns = st.multiselect("Preview selected columns", list(cleaned.columns), default=list(cleaned.columns[: min(6, len(cleaned.columns))]), key="feature_preview_columns")
    if selected_columns:
        st.dataframe(cleaned[selected_columns].head(100), hide_index=True, alt="Feature engineering preview")


def render_univariate(frame: pd.DataFrame) -> None:
    st.subheader("Univariate analysis")
    info = infer_column_types(frame)
    mode = st.segmented_control("Variable type", ["Numerical", "Categorical"], default="Numerical", key="univariate_mode")
    if mode == "Numerical":
        if not info["numeric_like"]:
            st.info("No numerical columns detected.")
            return
        column = st.selectbox("Numerical variable", info["numeric_like"], key="univariate_numeric")
        values = numeric_series(frame, column).dropna()
        if values.empty:
            st.info("This column has no valid numerical values.")
            return
        left, right = st.columns(2)
        with left:
            render_plot(px.histogram(values.to_frame(name=column), x=column, nbins=30, title=f"Distribution of {column}"), f"Distribution of {column}")
        with right:
            render_plot(px.box(values.to_frame(name=column), y=column, points="outliers", title=f"Box plot of {column}"), f"Box plot of {column}")
        st.dataframe(values.describe().to_frame("value").round(3), alt=f"Descriptive statistics for {column}")
    else:
        if not info["categorical"]:
            st.info("No categorical columns detected.")
            return
        column = st.selectbox("Categorical variable", info["categorical"], key="univariate_categorical")
        counts = frame[column].astype("string").fillna("<missing>").value_counts().head(30).rename_axis(column).reset_index(name="count")
        render_plot(px.bar(counts, x=column, y="count", title=f"Frequency of {column}", text_auto=True), f"Category counts for {column}")
        st.dataframe(counts, hide_index=True, alt=f"Frequency table for {column}")


def render_bivariate(frame: pd.DataFrame) -> None:
    st.subheader("Bivariate analysis")
    info = infer_column_types(frame)
    numeric = info["numeric_like"]
    categorical = info["categorical"]
    if not numeric and not categorical:
        st.info("At least one numerical or categorical column is needed.")
        return
    x_type = st.segmented_control("X-axis type", ["Numerical", "Categorical"], default="Numerical" if numeric else "Categorical", key="bivariate_x_type")
    x_options = numeric if x_type == "Numerical" else categorical
    x_column = st.selectbox("X-axis", x_options, key="bivariate_x")
    y_options = [column for column in numeric if column != x_column] if x_type == "Numerical" else [column for column in (numeric + categorical) if column != x_column]
    if not y_options:
        st.info("Choose another column type or add another column.")
        return
    y_column = st.selectbox("Y-axis", y_options, key="bivariate_y")
    if x_type == "Numerical" and y_column in numeric:
        color = st.selectbox("Color grouping (optional)", ["None"] + categorical, key="bivariate_color")
        kwargs = {} if color == "None" else {"color": color}
        render_plot(px.scatter(frame, x=x_column, y=y_column, **kwargs, title=f"{y_column} vs {x_column}", trendline=None), f"Relationship between {x_column} and {y_column}")
    elif x_type == "Categorical" and y_column in numeric:
        render_plot(px.box(frame, x=x_column, y=y_column, points="outliers", title=f"{y_column} by {x_column}"), f"{y_column} distribution by {x_column}")
    elif x_type == "Categorical" and y_column in categorical:
        grouped = frame.groupby([x_column, y_column], dropna=False).size().reset_index(name="count")
        render_plot(px.bar(grouped, x=x_column, y="count", color=y_column, barmode="group", title=f"{y_column} by {x_column}"), f"Counts of {y_column} grouped by {x_column}")
    else:
        st.info("This combination is not available for a meaningful chart.")


def render_multivariate(frame: pd.DataFrame) -> None:
    st.subheader("Multivariate analysis")
    info = infer_column_types(frame)
    numeric = info["numeric_like"]
    if len(numeric) < 2:
        st.info("At least two numerical columns are needed for the multivariate scatter plot.")
        return
    x_column = st.selectbox("X-axis", numeric, key="multi_x")
    y_column = st.selectbox("Y-axis", [column for column in numeric if column != x_column], key="multi_y")
    color = st.selectbox("Color grouping", ["None"] + info["categorical"], key="multi_color")
    size = st.selectbox("Point size", ["None"] + [column for column in numeric if column not in {x_column, y_column}], key="multi_size")
    hover_columns = st.multiselect("Additional tooltip columns", list(frame.columns), default=[], key="multi_hover")
    kwargs = {}
    if color != "None":
        kwargs["color"] = color
    if size != "None":
        kwargs["size"] = size
    render_plot(px.scatter(frame, x=x_column, y=y_column, hover_data=hover_columns, **kwargs, title=f"{y_column} vs {x_column}"), f"Multivariate relationship between {x_column} and {y_column}")


def render_time_series(frame: pd.DataFrame) -> None:
    st.subheader("Time-series analysis")
    info = infer_column_types(frame)
    date_columns = info["date"]
    numeric = info["numeric_like"]
    if not date_columns:
        st.info("No date column is available. Convert a date/time field or create an inferred date from year and month on the Feature engineering page.")
        return
    if not numeric:
        st.info("No numerical measure is available for a time-series chart.")
        return
    date_column = st.selectbox("Date column", date_columns, key="time_date")
    value_column = st.selectbox("Measure", numeric, key="time_value")
    aggregation = st.selectbox("Aggregate by", ["Day", "Month", "Quarter", "Year"], key="time_aggregation")
    group_options = ["None"] + [column for column in info["categorical"] if frame[column].nunique(dropna=True) <= 20]
    group_column = st.selectbox("Group by (optional)", group_options, key="time_group")
    parsed = parse_datetime_series(frame[date_column])
    if not pd.api.types.is_datetime64_any_dtype(parsed) or parsed.notna().sum() < 2:
        st.info("The selected column does not contain enough valid dates.")
        return
    work = frame.copy()
    work["_analysis_date"] = parsed
    work["_analysis_value"] = numeric_series(frame, value_column)
    work = work.dropna(subset=["_analysis_date", "_analysis_value"])
    frequency = {"Day": "D", "Month": "MS", "Quarter": "QS", "Year": "YS"}[aggregation]
    group_fields = ["_analysis_date"] if group_column == "None" else ["_analysis_date", group_column]
    grouped = work.groupby(group_fields).agg(value=("_analysis_value", "sum")).reset_index()
    grouped["_period"] = grouped["_analysis_date"].dt.to_period(frequency[0] if frequency != "MS" else "M").dt.to_timestamp()
    if group_column == "None":
        trend = grouped.groupby("_period", as_index=False)["value"].sum()
        render_plot(px.line(trend, x="_period", y="value", markers=True, title=f"{value_column} by {aggregation.lower()}"), f"Time-series trend for {value_column}")
    else:
        trend = grouped.groupby(["_period", group_column], as_index=False)["value"].sum()
        render_plot(px.line(trend, x="_period", y="value", color=group_column, markers=True, title=f"{value_column} by {aggregation.lower()} and {group_column}"), f"Time-series trend for {value_column} grouped by {group_column}")


def render_eda(active: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    st.header("Exploratory data analysis", anchor=False)
    st.caption("Choose one analysis module at a time so the workspace stays focused and responsive.")
    module = st.selectbox(
        "EDA module",
        ["Feature engineering", "Univariate analysis", "Bivariate analysis", "Multivariate analysis", "Time-series analysis"],
        key="eda_module",
    )
    if module == "Feature engineering":
        render_feature_engineering(cleaned)
    elif module == "Univariate analysis":
        render_univariate(active)
    elif module == "Bivariate analysis":
        render_bivariate(active)
    elif module == "Multivariate analysis":
        render_multivariate(active)
    else:
        render_time_series(active)


def render_gallery_chart(chart_type: str, frame: pd.DataFrame, slot: object) -> None:
    info = infer_column_types(frame)
    numeric = info["numeric_like"]
    categorical = info["categorical"]
    with slot:
        st.markdown(f"**{chart_type}**")
        if chart_type == "Bar chart":
            if not categorical or not numeric:
                st.info("A bar chart needs at least one categorical and one numerical column.")
                return
            category = st.selectbox("Category", categorical, key="gallery_bar_category")
            measure = st.selectbox("Measure", numeric, key="gallery_bar_measure")
            grouped = frame.groupby(category, dropna=False, as_index=False)[measure].sum().sort_values(measure, ascending=False).head(30)
            render_plot(px.bar(grouped, x=category, y=measure, title=f"{measure} by {category}", text_auto=True), f"{measure} by {category}")
        elif chart_type == "Line chart":
            if not info["date"] or not numeric:
                st.info("A line chart needs a detected date column and a numerical measure.")
                return
            date_column = st.selectbox("Date", info["date"], key="gallery_line_date")
            measure = st.selectbox("Measure", numeric, key="gallery_line_measure")
            work = pd.DataFrame({"date": parse_datetime_series(frame[date_column]), "value": numeric_series(frame, measure)}).dropna()
            trend = work.groupby("date", as_index=False)["value"].sum().sort_values("date")
            render_plot(px.line(trend, x="date", y="value", markers=True, title=f"{measure} over time"), f"{measure} over time")
        elif chart_type == "Histogram":
            if not numeric:
                st.info("A histogram needs a numerical column.")
                return
            measure = st.selectbox("Measure", numeric, key="gallery_histogram_measure")
            render_plot(px.histogram(frame, x=measure, nbins=30, title=f"Distribution of {measure}"), f"Distribution of {measure}")
        elif chart_type == "Scatter plot":
            if len(numeric) < 2:
                st.info("A scatter plot needs at least two numerical columns.")
                return
            x_column = st.selectbox("X-axis", numeric, key="gallery_scatter_x")
            y_column = st.selectbox("Y-axis", [column for column in numeric if column != x_column], key="gallery_scatter_y")
            color = st.selectbox("Color (optional)", ["None"] + categorical, key="gallery_scatter_color")
            kwargs = {} if color == "None" else {"color": color}
            render_plot(px.scatter(frame, x=x_column, y=y_column, **kwargs, title=f"{y_column} vs {x_column}"), f"Scatter plot of {y_column} versus {x_column}")
        elif chart_type == "Box plot":
            if not numeric:
                st.info("A box plot needs a numerical column.")
                return
            measure = st.selectbox("Measure", numeric, key="gallery_box_measure")
            category = st.selectbox("Group (optional)", ["None"] + categorical, key="gallery_box_category")
            kwargs = {} if category == "None" else {"x": category}
            render_plot(px.box(frame, y=measure, **kwargs, points="outliers", title=f"Distribution of {measure}"), f"Box plot of {measure}")


def render_correlation_heatmap(frame: pd.DataFrame, slot: object | None = None) -> None:
    def draw() -> None:
        numerical = numeric_frame(frame)
        numerical = numerical.loc[:, numerical.nunique(dropna=True) > 1]
        if numerical.shape[1] < 2:
            st.info("A correlation heatmap needs at least two numerical columns with more than one distinct value.")
            return
        correlation = numerical.corr().dropna(axis=0, how="all").dropna(axis=1, how="all")
        if correlation.shape[0] < 2:
            st.info("Correlation could not be calculated because the numerical columns are constant or all missing.")
            return
        rounded_values = correlation.round(2).astype(str).values
        figure = go.Figure(
            data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns.tolist(),
                y=correlation.index.tolist(),
                zmin=-1,
                zmax=1,
                colorscale=CORRELATION_COLORS,
                text=rounded_values,
                texttemplate="%{text}",
                textfont={"color": "#14213D"},
                hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>",
                colorbar={"title": {"text": "r", "font": {"color": "#14213D"}}, "tickfont": {"color": "#475569"}},
            )
        )
        figure.update_layout(title="Numerical correlation heatmap")
        render_plot(figure, "Correlation heatmap for numerical columns", height=500)

    if slot is None:
        draw()
    else:
        with slot:
            draw()


def render_visualizations(active: pd.DataFrame) -> None:
    st.header("Visualizations", anchor=False)
    st.caption("The correlation heatmap is always included. Add up to three other charts, for a maximum of four main charts at a time.")
    with st.container(border=True, key="content-card-correlation"):
        st.markdown("**Mandatory correlation heatmap**")
        render_correlation_heatmap(active)

    chart_choices = st.multiselect(
        "Additional charts",
        ["Bar chart", "Line chart", "Histogram", "Scatter plot", "Box plot"],
        default=["Bar chart", "Line chart", "Histogram"],
        max_selections=3,
        key="gallery_charts",
    )
    if not chart_choices:
        st.info("Select one or more chart types to continue.")
        return
    columns = st.columns(2)
    for index, chart_type in enumerate(chart_choices):
        render_gallery_chart(chart_type, active, columns[index % 2])


def relationship_strength(correlation: float) -> str:
    absolute = abs(correlation)
    if absolute >= 0.7:
        return "strong"
    if absolute >= 0.4:
        return "moderate"
    if absolute >= 0.2:
        return "weak"
    return "very weak"


def render_insights(active: pd.DataFrame) -> None:
    st.header("Automated insights", anchor=False)
    st.caption("Observations below are calculated from the current active data. Association is not proof of causation.")
    info = infer_column_types(active)
    numeric = info["numeric_like"]
    categorical = info["categorical"]
    if active.empty:
        st.warning("There are no rows in the active view. Reset filters to generate insights.")
        return

    st.subheader("Advertising spend versus revenue")
    spend_suggestion = suggest_column(numeric, ["advertising", "ad", "marketing", "spend", "cost"])
    revenue_suggestion = suggest_column(numeric, ["revenue", "sales", "amount", "income", "turnover"], exclude=[spend_suggestion] if spend_suggestion else [])
    left, right = st.columns(2)
    with left:
        spend_column = st.selectbox("Advertising or spend column", ["Skip"] + numeric, index=(numeric.index(spend_suggestion) + 1 if spend_suggestion in numeric else 0), key="insight_spend")
    with right:
        revenue_column = st.selectbox("Revenue or outcome column", ["Skip"] + [column for column in numeric if column != spend_column], index=([column for column in numeric if column != spend_column].index(revenue_suggestion) + 1 if revenue_suggestion in numeric and revenue_suggestion != spend_column else 0), key="insight_revenue")
    if spend_column == "Skip" or revenue_column == "Skip":
        st.info("Select equivalent columns above to run this case, or skip it when the dataset has no spend/revenue measures.")
    else:
        paired = pd.DataFrame({"spend": numeric_series(active, spend_column), "revenue": numeric_series(active, revenue_column)}).dropna()
        if len(paired) < 3 or paired.nunique().min() < 2:
            st.info("There are not enough valid, varying observations to calculate a reliable correlation.")
        else:
            correlation = float(paired["spend"].corr(paired["revenue"]))
            with st.container(key="inline-kpi-pearson-correlation"):
                st.metric("Pearson correlation", f"{correlation:.3f}", border=False, width="stretch")
            render_plot(px.scatter(paired, x="spend", y="revenue", trendline=None, title=f"{revenue_column} versus {spend_column}"), f"Revenue versus advertising spend with correlation {correlation:.3f}")
            direction = "positive" if correlation >= 0 else "negative"
            st.info(
                f"Observation: The filtered data shows a {relationship_strength(correlation)} {direction} association "
                f"between {spend_column} and {revenue_column} (r = {correlation:.3f}). "
                f"Meaning: The two measures tend to move {direction}ly in this sample. "
                f"Possible decision: Segment by campaign, product, or time period before changing budget. "
                f"Limitation: Correlation does not establish causation and may reflect seasonality, pricing, or other confounders."
            )

    st.divider()
    st.subheader("Category versus revenue")
    category_suggestion = suggest_column(categorical, ["category", "product", "segment", "type", "class"])
    revenue_options = numeric
    category_column = st.selectbox("Category column", ["Skip"] + categorical, index=(categorical.index(category_suggestion) + 1 if category_suggestion in categorical else 0), key="insight_category")
    selected_metric = st.selectbox("Metric to compare", ["Skip"] + revenue_options, index=(revenue_options.index(revenue_suggestion) + 1 if revenue_suggestion in revenue_options else 0), key="insight_category_metric")
    aggregation = st.selectbox("Aggregation", ["Sum", "Mean"], key="insight_category_aggregation")
    if category_column == "Skip" or selected_metric == "Skip":
        st.info("Choose equivalent category and outcome columns to compare performance, or skip this case.")
    else:
        measure = numeric_series(active, selected_metric)
        grouped = pd.DataFrame({"category": active[category_column].astype("string").fillna("<missing>"), "measure": measure}).dropna()
        grouped = grouped.groupby("category", as_index=False)["measure"].agg("sum" if aggregation == "Sum" else "mean").sort_values("measure", ascending=False)
        if grouped.empty:
            st.info("The selected columns do not contain enough valid values.")
        else:
            render_plot(px.bar(grouped.head(30), x="category", y="measure", text_auto=True, title=f"{aggregation} of {selected_metric} by {category_column}"), f"{aggregation} of {selected_metric} by category")
            leaders = ", ".join(grouped.head(3)["category"].astype(str).tolist())
            st.info(
                f"Observation: The highest-performing category group(s) by {aggregation.lower()} {selected_metric} are {leaders}. "
                f"Meaning: Performance is concentrated in the leading groups in the current slice. "
                f"Possible decision: Investigate their pricing, availability, and customer mix before allocating more resources. "
                f"Limitation: This is descriptive and does not control for category size or profitability."
            )

    st.divider()
    st.subheader("Category versus region")
    region_suggestion = suggest_column(categorical, ["region", "country", "state", "city", "area", "location"], exclude=[category_column] if category_column != "Skip" else [])
    region_options = [column for column in categorical if column != category_column]
    region_column = st.selectbox("Region column", ["Skip"] + region_options, index=(region_options.index(region_suggestion) + 1 if region_suggestion in region_options else 0), key="insight_region")
    matrix_metric = st.selectbox("Metric for category-region comparison", ["Skip"] + numeric, index=(numeric.index(revenue_suggestion) + 1 if revenue_suggestion in numeric else 0), key="insight_region_metric")
    if category_column == "Skip" or region_column == "Skip" or matrix_metric == "Skip":
        st.info("Choose equivalent category, region, and metric columns to run this case, or skip it when those concepts are unavailable.")
    else:
        matrix = active.copy()
        matrix["_category"] = matrix[category_column].astype("string").fillna("<missing>")
        matrix["_region"] = matrix[region_column].astype("string").fillna("<missing>")
        matrix["_metric"] = numeric_series(matrix, matrix_metric)
        matrix = matrix.dropna(subset=["_metric"])
        pivot = matrix.pivot_table(index="_category", columns="_region", values="_metric", aggfunc="sum", fill_value=0)
        if pivot.empty:
            st.info("No valid category-region values were found.")
        else:
            heatmap = px.imshow(pivot, text_auto=".2f", aspect="auto", color_continuous_scale=CORRELATION_COLORS, title=f"{matrix_metric} by {category_column} and {region_column}")
            render_plot(heatmap, f"{matrix_metric} by category and region", height=520)
            best_pair = matrix.groupby(["_category", "_region"], as_index=False)["_metric"].sum().sort_values("_metric", ascending=False).iloc[0]
            st.info(
                f"Observation: The strongest category-region combination is {best_pair['_category']} in {best_pair['_region']} "
                f"with {best_pair['_metric']:,.2f} in {matrix_metric}. "
                f"Meaning: The selected metric is unevenly distributed across the two dimensions. "
                f"Possible decision: Compare local demand, fulfillment, and campaign context for the leading and lagging cells. "
                f"Limitation: Empty or small cells can make comparisons unstable."
            )


def render_app_banner(source_name: str, page: str, active_rows: int, filter_notes: list[str]) -> None:
    """Render the branded dashboard header above every workspace view."""
    safe_source = html.escape(str(source_name))
    safe_page = html.escape(str(page))
    active_filter_text = html.escape(" · ".join(filter_notes) if filter_notes else "No filters applied")
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-eyebrow">E-COMMERCE STORE INSIGHTS <span class="hero-status">LIVE DATA</span></div>
            <div class="hero-content">
                <div>
                    <div class="hero-title">E-Commerce Store Insights</div>
                    <div class="hero-subtitle">A focused workspace for performance, data quality, and evidence-driven decisions.</div>
                    <div class="section-caption">{safe_page} · {safe_source} · {active_filter_text}</div>
                </div>
                <div class="hero-badge">
                    <span class="hero-badge-label">ACTIVE VIEW</span>
                    <strong>{active_rows:,}</strong>
                    <span>rows after filters</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_source_and_navigation() -> tuple[object | None, str]:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-mark">✦</div>
                <div>
                    <div class="sidebar-brand-name">Store Insights</div>
                    <div class="sidebar-brand-subtitle">E-commerce analytics studio</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Upload a CSV or explore the bundled sample dataset.")
        uploaded_file = st.file_uploader(
            "Upload a CSV",
            type=["csv"],
            help="Upload your own dataset or use the bundled e-commerce CSV.",
        )
        page = st.radio(
            "Workspace tabs",
            ["Overview", "Data understanding", "Data cleaning", "EDA", "Visualizations", "Insights"],
            format_func=lambda value: {
                "Overview": ":material/dashboard: Overview",
                "Data understanding": ":material/table_view: Data understanding",
                "Data cleaning": ":material/cleaning_services: Data cleaning",
                "EDA": ":material/analytics: EDA",
                "Visualizations": ":material/insert_chart: Visualizations",
                "Insights": ":material/lightbulb: Insights",
            }[value],
            key="navigation",
        )
        st.divider()
        st.caption("Default source")
        st.code(DEFAULT_DATASET.name)
    return uploaded_file, page


def overview_period_series(frame: pd.DataFrame) -> tuple[pd.Series | None, str]:
    """Build a safe period series from a date field or year/month pair."""
    info = infer_column_types(frame)
    for column in info["date"]:
        parsed = parse_datetime_series(frame[column])
        if pd.api.types.is_datetime64_any_dtype(parsed) and parsed.notna().sum() >= 2:
            return parsed, column

    year_column = year_filter_column(frame)
    month_column = suggest_column(frame.columns, ["month"])
    if year_column and month_column:
        year_values = pd.to_numeric(frame[year_column], errors="coerce")
        month_values = frame[month_column].map(month_number)
        period = pd.to_datetime(
            {"year": year_values, "month": month_values, "day": 1},
            errors="coerce",
        )
        if period.notna().sum() >= 2:
            return period, f"{year_column} + {month_column}"

    if year_column:
        year_values = pd.to_numeric(frame[year_column], errors="coerce")
        period = pd.to_datetime(year_values.astype("Int64").astype("string"), format="%Y", errors="coerce")
        if period.notna().sum() >= 2:
            return period, year_column

    return None, ""


def grouped_metric(frame: pd.DataFrame, dimension: str, measure: str) -> pd.DataFrame:
    """Aggregate a selected measure by a selected categorical dimension."""
    work = pd.DataFrame(
        {
            "dimension": frame[dimension].astype("string").fillna("<missing>"),
            "metric": numeric_series(frame, measure),
        }
    ).dropna(subset=["metric"])
    if work.empty:
        return pd.DataFrame(columns=["dimension", "metric"])
    return (
        work.groupby("dimension", as_index=False)["metric"]
        .sum()
        .sort_values("metric", ascending=False)
    )


def render_overview_performance(active: pd.DataFrame) -> None:
    """Render filtered revenue, category, regional, and top-product views."""
    st.subheader("Performance overview", anchor=False)
    if active.empty:
        st.info("No rows are available for the performance overview. Adjust the sidebar filters.")
        return

    info = infer_column_types(active)
    numeric = info["numeric_like"]
    categorical = info["categorical"]
    if not numeric:
        st.info("No numerical measures were detected for performance charts.")
        return

    revenue_suggestion = suggest_column(numeric, ["revenue", "sales", "amount", "income", "turnover"])
    category_suggestion = suggest_column(categorical, ["category", "product", "item", "segment", "type"])
    region_suggestion = suggest_column(categorical, ["region", "country", "state", "city", "area", "location"])
    metric_index = numeric.index(revenue_suggestion) if revenue_suggestion in numeric else 0

    with st.container(border=True, key="content-card-overview-performance"):
        controls = st.columns(3)
        with controls[0]:
            trend_measure = st.selectbox("Trend measure", numeric, index=metric_index, key="overview_trend_measure")
        with controls[1]:
            category_options = ["None"] + categorical
            category_index = category_options.index(category_suggestion) if category_suggestion in category_options else 0
            category_dimension = st.selectbox("Category dimension", category_options, index=category_index, key="overview_category_dimension")
        with controls[2]:
            region_options = ["None"] + categorical
            region_index = region_options.index(region_suggestion) if region_suggestion in region_options else 0
            region_dimension = st.selectbox("Region dimension", region_options, index=region_index, key="overview_region_dimension")

        chart_row = st.columns(2)
        with chart_row[0]:
            period, period_label = overview_period_series(active)
            if period is None:
                st.info("No usable date or year/month fields were detected for a trend chart.")
            else:
                trend_data = pd.DataFrame(
                    {"period": period, "metric": numeric_series(active, trend_measure)}
                ).dropna(subset=["period", "metric"])
                trend_data = trend_data.groupby("period", as_index=False)["metric"].sum().sort_values("period")
                if trend_data.empty:
                    st.info("The selected trend measure has no valid values in the filtered data.")
                else:
                    figure = px.line(
                        trend_data,
                        x="period",
                        y="metric",
                        markers=True,
                        title=f"{trend_measure} trend",
                        labels={"period": period_label, "metric": trend_measure},
                        color_discrete_sequence=[CHART_COLORS[1]],
                    )
                    render_plot(figure, f"{trend_measure} trend over time", height=390)

        with chart_row[1]:
            if category_dimension == "None":
                st.info("Choose a category dimension to view category performance.")
            else:
                category_data = grouped_metric(active, category_dimension, trend_measure).head(15)
                if category_data.empty:
                    st.info("No valid category values are available for the selected measure.")
                else:
                    figure = px.bar(
                        category_data,
                        x="dimension",
                        y="metric",
                        text_auto=True,
                        title=f"{trend_measure} by {category_dimension}",
                        labels={"dimension": category_dimension, "metric": trend_measure},
                        color_discrete_sequence=[CHART_COLORS[0]],
                    )
                    render_plot(figure, f"{trend_measure} by {category_dimension}", height=390)

        chart_row = st.columns(2)
        with chart_row[0]:
            if region_dimension == "None":
                st.info("Choose a region dimension to view regional performance.")
            else:
                region_data = grouped_metric(active, region_dimension, trend_measure).head(15)
                if region_data.empty:
                    st.info("No valid region values are available for the selected measure.")
                else:
                    figure = px.bar(
                        region_data,
                        x="dimension",
                        y="metric",
                        text_auto=True,
                        title=f"{trend_measure} by {region_dimension}",
                        labels={"dimension": region_dimension, "metric": trend_measure},
                        color_discrete_sequence=[CHART_COLORS[1]],
                    )
                    render_plot(figure, f"{trend_measure} by {region_dimension}", height=390)

        with chart_row[1]:
            with st.container(border=True, key="content-card-overview-top-products"):
                st.subheader("Top products", anchor=False)
                product_options = [
                    column for column in categorical
                    if semantic_score(column, ["product", "item", "sku", "product name", "name"]) > 0
                ]
                if category_suggestion and category_suggestion not in product_options:
                    product_options.insert(0, category_suggestion)
                if not product_options:
                    product_options = categorical
                if not product_options:
                    st.info("No categorical column is available for a top-products table.")
                else:
                    product_column = st.selectbox("Product or category column", product_options, key="overview_product_column")
                    product_data = grouped_metric(active, product_column, trend_measure).head(10).reset_index(drop=True)
                    if product_data.empty:
                        st.info("No valid product values are available for the selected measure.")
                    else:
                        product_data.insert(0, "Rank", np.arange(1, len(product_data) + 1))
                        product_data = product_data.rename(
                            columns={"dimension": product_column, "metric": trend_measure}
                        )
                        st.dataframe(
                            product_data,
                            hide_index=True,
                            width="stretch",
                            alt=f"Top products by {trend_measure}",
                        )


def render_overview(active: pd.DataFrame, cleaned: pd.DataFrame, original: pd.DataFrame, source_name: str, filter_notes: list[str]) -> None:
    st.header("Overview", anchor=False)
    st.caption("Upload a CSV, understand its quality, clean it safely, and explore evidence-driven patterns.")
    with st.container(border=True, key="content-card-overview-workspace"):
        st.subheader("Current workspace")
        st.write(f"Source: {source_name}")
        if filter_notes:
            st.write("Active filters: " + " · ".join(filter_notes))
        else:
            st.write("Active filters: None — showing the complete cleaned dataset.")
        st.write("Use the navigation in the sidebar to move from data understanding to cleaning, EDA, visualizations, and insights.")
    render_kpis(active, cleaned)
    render_overview_performance(active)

    left, right = st.columns(2)
    with left:
        with st.container(border=True, key="content-card-overview-quick-start"):
            st.subheader("Quick start")
            st.markdown(
                "1. Upload a CSV or use the bundled sample.\n"
                "2. Review missing values and data types.\n"
                "3. Clean a working copy and download it.\n"
                "4. Explore relationships and generate insights."
            )
    with right:
        with st.container(border=True, key="content-card-overview-quality"):
            st.subheader("Data quality at a glance")
            quality = missing_summary(active)
            missing = quality[quality["missing"] > 0]
            if missing.empty:
                st.success("No missing values detected in the filtered view.")
            else:
                st.warning(f"{len(missing)} column(s) contain missing values.")
                st.dataframe(missing.head(10), hide_index=True, alt="Top columns with missing values")

    st.subheader("Correlation preview", anchor=False)
    render_correlation_heatmap(active)


def load_source(uploaded_file: object | None) -> tuple[pd.DataFrame, str, str]:
    if uploaded_file is not None:
        contents = uploaded_file.getvalue()
        source_name = uploaded_file.name
    else:
        if not DEFAULT_DATASET.exists():
            raise FileNotFoundError(f"Default dataset not found: {DEFAULT_DATASET}")
        contents = DEFAULT_DATASET.read_bytes()
        source_name = DEFAULT_DATASET.name
    return read_csv_bytes(contents), source_name, dataframe_signature(contents, source_name)


def main() -> None:
    uploaded_file, page = sidebar_source_and_navigation()
    try:
        raw, source_name, signature = load_source(uploaded_file)
        initialise_dataset(raw, source_name, signature)
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(str(error), icon=":material/error:")
        st.stop()

    original = st.session_state["original_df"]
    cleaned = st.session_state["clean_df"]
    active, filter_notes = apply_sidebar_filters(cleaned)
    render_app_banner(source_name, page, len(active), filter_notes)

    if cleaned.empty:
        st.warning("The cleaned dataset has no rows. Reset or reload the dataset from the Data cleaning page.")
    elif page == "Overview":
        if active.empty:
            st.warning("No data matches the applied filters. Use Reset Filters or broaden the selections in the sidebar.")
        render_overview(active, cleaned, original, source_name, filter_notes)
    else:
        render_kpis(active, cleaned)
        if active.empty:
            st.warning("No data matches the applied filters. Use Reset Filters or broaden the selections in the sidebar.")
        if page == "Data understanding":
            render_data_understanding(active, original)
        elif page == "Data cleaning":
            render_data_cleaning(original, cleaned, active)
        elif page == "EDA":
            render_eda(active, cleaned)
        elif page == "Visualizations":
            render_visualizations(active)
        elif page == "Insights":
            render_insights(active)

    st.divider()
    st.caption("All statistics and recommendations are calculated from the active dataset view. Correlation is association, not causation.")


if __name__ == "__main__":
    main()
