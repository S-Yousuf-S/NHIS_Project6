"""
================================================================================
 PREREQUISITES.PY

 Project      : Sales Forecasting Across Multiple Retail Stores
 
 Author       : Yousuf S. R. Sakkaf

 Description  : Shared, project-agnostic utility module imported by all 4 notebooks:
                01_salesfc_preprocessing.ipynb
                02_salesfc_ml_modeling.ipynb
                03_salesfc_deep_learning.ipynb
                04_salesfc_mlflow_tracking.ipynb

 Scope        : Only contains logic that is genuinely reusable across ANY project/dataset - 
                no dataset-specific decisions (e.g. which strategy to use on which column, which features to engineer).
                Those decisions are made explicitly and visibly inside each notebook, not hidden in a function here.

 Contents     : - Data loading cascade: Local -> Google Drive -> GitHub raw
                - Logging configuration
                - HTML theme, color palette & plot style (matplotlib/seaborn)
                - Generic dataset overview (shape, dtypes, describe - styled)
                - Generic audit functions (missing values, duplicates - report only)
                - Generic statistical formulas (IQR, Z-score, skew, kurtosis)

 Usage        : from prerequisites import *
================================================================================
"""

import os
import sys
import glob
import logging
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, zscore
from datetime import datetime
from IPython.display import display, HTML, Image



# ==============================================================================
# 1. DATA LOADING CASCADE: Local -> Google Drive -> GitHub raw
# ==============================================================================

def attempt_drive_mount(mount_point="/content/drive"):
    """
    Attempts to mount Google Drive. 
    If successful, returns True. If not (e.g., running locally or user declines), returns False.
    """
    try:
        from google.colab import drive
        drive.mount(mount_point)
        return True
    except Exception as e:
        warning_box(f"Drive mount skipped or unavailable: {e}")
        return False


def load_data_file(filename, local_subdir="Assets",
                    drive_path=None, github_raw_base=None, logger=None):
    """
    Loads a CSV by trying, in order:
      1. Local Assets/ folder
      2. Google Drive (if drive_path is given and Drive is mounted)
      3. GitHub raw URL (if github_raw_base is given) - and saves a local
         copy to Assets/ afterward so future runs hit step 1 first

    Parameters
    ----------
    filename         : e.g. "train.csv"
    local_subdir     : folder to look in locally, default "Assets"
    drive_path       : full Drive path to the file, optional
    github_raw_base  : base GitHub raw URL (no trailing slash), optional

    Returns
    -------
    pd.DataFrame
    """
    local_path = os.path.join(local_subdir, filename)

    # 1. Local
    if os.path.exists(local_path):
        success_box(f"'{filename}' loaded from Local Assets folder.")
        if logger:
            logger.info(f"Loaded '{filename}' from local {local_subdir}/ folder.")
        return pd.read_csv(local_path, low_memory=False)

    # 2. Google Drive
    if drive_path and os.path.exists(drive_path):
        success_box(f"'{filename}' loaded from Google Drive.")
        if logger:
            logger.info(f"Loaded '{filename}' from Google Drive.")
        return pd.read_csv(drive_path, low_memory=False)

    # 3. GitHub raw
    if github_raw_base:
        try:
            url = f"{github_raw_base}/{filename}"
            print(f"🌐 Attempting to load '{filename}' from GitHub...")
            data = pd.read_csv(url, low_memory=False)
            success_box(f"'{filename}' loaded from GitHub repository.")
            os.makedirs(local_subdir, exist_ok=True)
            data.to_csv(local_path, index=False)
            print(f"💾 Data physically saved to '{local_path}' for future runs.")
            if logger:
                logger.info(f"Loaded '{filename}' from GitHub raw and cached locally.")
            return data
        except Exception as e:
            warning_box(f"GitHub load failed for '{filename}': {e}")
            if logger:
                logger.warning(f"GitHub load failed for '{filename}': {e}")

    raise FileNotFoundError(
        f"'{filename}' not found in Local, Google Drive, or GitHub."
    )


def load_latest_model_artifact(pattern, loader, local_subdir="Assets",
                                drive_dir=None, github_api_dir=None, logger=None):
    """
    Loads the most recently timestamped model artifact matching `pattern`
    (e.g. "final_tuned_rf_pipeline_*.pkl"), trying in order:
      1. Local `local_subdir` folder
      2. Google Drive `drive_dir` (if given and genuinely mounted)
      3. GitHub, via the repo contents API at `github_api_dir`
         (e.g. "https://api.github.com/repos/USER/REPO/contents/Assets") -
         raw URLs can't be globbed, so this lists the directory instead.

    Parameters
    ----------
    pattern        : glob-style filename pattern with a `*` for the timestamp
    loader         : callable that takes a filepath and returns the loaded
                      object, e.g. joblib.load or keras.models.load_model
    drive_dir      : full Drive directory path (not a file), optional
    github_api_dir : GitHub contents-API URL for the containing directory, optional
    """
    def _latest_by_timestamp(names):
        def _ts(n):
            m = re.search(r'(\d{2}-\d{2}-\d{4}-\d{2}-\d{2}-\d{2})', os.path.basename(n))
            return datetime.strptime(m.group(1), "%d-%m-%Y-%H-%M-%S") if m else datetime.min
        return max(names, key=_ts)

    local_matches = glob.glob(os.path.join(local_subdir, pattern))
    if local_matches:
        latest = _latest_by_timestamp(local_matches)
        success_box(f"Latest matching artifact loaded from Local: <b>{latest}</b>.")
        if logger:
            logger.info(f"Loaded latest local model artifact: {latest}")
        return loader(latest)

    if drive_dir and os.path.exists(drive_dir):
        drive_matches = glob.glob(os.path.join(drive_dir, pattern))
        if drive_matches:
            latest = _latest_by_timestamp(drive_matches)
            success_box(f"Latest matching artifact loaded from Google Drive: <b>{latest}</b>.")
            if logger:
                logger.info(f"Loaded latest Drive model artifact: {latest}")
            return loader(latest)

    if github_api_dir:
        try:
            import urllib.request, json as _json, fnmatch
            req = urllib.request.Request(github_api_dir, headers={"Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req) as resp:
                listing = _json.loads(resp.read())
            candidates = [f["name"] for f in listing if fnmatch.fnmatch(f["name"], pattern)]
            if candidates:
                latest_name = _latest_by_timestamp(candidates)
                raw_url = next(f["download_url"] for f in listing if f["name"] == latest_name)
                os.makedirs(local_subdir, exist_ok=True)
                local_path = os.path.join(local_subdir, latest_name)
                urllib.request.urlretrieve(raw_url, local_path)
                success_box(f"Latest matching artifact downloaded from GitHub and cached locally: <b>{local_path}</b>.")
                if logger:
                    logger.info(f"Downloaded latest GitHub model artifact: {latest_name}")
                return loader(local_path)
        except Exception as e:
            warning_box(f"GitHub artifact lookup failed: {e}")
            if logger:
                logger.warning(f"GitHub artifact lookup failed: {e}")

    raise FileNotFoundError(f"No artifact matching '{pattern}' found in Local, Drive, or GitHub.")

# ==============================================================================
# 2. LOGGING CONFIGURATION
# ==============================================================================

def setup_logging(notebook_name="notebook", log_dir="Logs", level=logging.DEBUG):
    """
    Configures a logger writing to both a date-stamped file (Logs/) and
    Console, with the notebook name included in every line for cross-notebook
    differentiation. 
    Default level is DEBUG so all severity levels are captured. 
    Use logger.debug/info/warning/error/critical as appropriate.
    """
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(log_dir, f"YS_{notebook_name}_{date_str}.log")

    logger = logging.getLogger(notebook_name)
    logger.setLevel(level)
    logger.handlers.clear()  # avoid duplicate handlers on re-import/reload
    logger.propagate = False  # prevent root logger from also printing to console

    formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging session started for '{notebook_name}' -> {log_path}")
    return logger

def log_debug(logger, message):
    logger.debug(f"DEBUG: {message}")

def log_success(logger, message):
    logger.info(f"SUCCESS: {message}")


def log_warning(logger, message):
    logger.warning(f"WARNING: {message}")


def log_info(logger, message):
    logger.info(f"INFO: {message}")


def log_error(logger, message):
    logger.error(f"ERROR: {message}")


# ==============================================================================
# 3. HTML THEME & COLOR PALETTE
# ==============================================================================

PRIMARY_COLOR   = "indigo"      # headers
SECONDARY_COLOR = "thistle"      # light accent / subsection background
SUCCESS_COLOR   = "#2E8B57"      # green - success / validation
WARNING_COLOR   = "#C0392B"      # red   - warnings / issues
INFO_COLOR      = "#1F77B4"      # blue  - info callouts
TABLE_COLOR     = "mediumpurple"      # table header background
TITLE_COLOR     = "midnightblue" # chart title text color



def section_header(title):
    display(HTML(f"""
    <div style="background:{PRIMARY_COLOR};color:white;padding:16px;border-radius:8px;font-size:28px;font-weight:bold;margin-top:20px;margin-bottom:12px;text-align:center;">
        {title}
    </div>"""))
 
 
def subsection_header(title):
    display(HTML(f"""
    <div style="background:{SECONDARY_COLOR};border-left:6px solid {PRIMARY_COLOR};color:{PRIMARY_COLOR};padding:10px;border-radius:6px;font-size:20px;font-weight:bold;margin-top:15px;margin-bottom:10px;text-align:center;">
        <i>{title}</i>
    </div>"""))
 
 
def _themed_box(icon, label, message, color, bg):
    display(HTML(f'<div style="background:{bg};border-left:6px solid {color};padding:12px 16px;border-radius:6px;margin:10px 0;line-height:1.4;"><div style="color:{color};font-weight:bold;">{icon} {label}:</div><div style="color:black;margin-top:4px;">{message}</div></div>'))
 
 
def success_box(message):
    _themed_box("✅", "Success", message, SUCCESS_COLOR, "#EAF7EA")
 
 
def warning_box(message):
    _themed_box("⚠️", "Note", message, WARNING_COLOR, "#FDEDEC")
 
 
def info_box(message):
    _themed_box("ℹ️", "Information", message, INFO_COLOR, "#F1F3F6")
 
 
def centered_table(df, index=False, float_format='{:.2f}'):
    """Renders a DataFrame as a centered, well-spaced HTML table matching the theme."""
    html = df.to_html(index=index, border=0, escape=False, float_format=float_format.format)
    html = html.replace('class="dataframe"', '')
    html = html.replace(
        '<table', '<table style="margin:auto;border-collapse:separate;border-spacing:0;'
        'border:1px solid #ccc;border-radius:6px;overflow:hidden;"'
    ).replace(
        '<th>', f'<th style="background:{TABLE_COLOR};color:white;padding:10px 24px;'
                f'text-align:center !important;border-bottom:1px solid #ccc;">'
    ).replace(
        '<td>', '<td style="padding:8px 24px;text-align:center !important;border-bottom:1px solid #eee;">'
    )
    display(HTML(f'<div style="overflow-x:auto;margin:15px 0;">{html}</div>'))
 

def set_global_visualization_settings(palette="viridis"):
    """
    Applies the project's global display + visualization settings -
    pandas display options, seaborn theme/palette, and matplotlib rcParams
    (including TITLE_COLOR for chart titles). Call once per notebook,
    right after imports.

    Parameters
    ----------
    palette : str, default "viridis"
        Any seaborn/matplotlib colormap name - e.g. "viridis", "mako",
        "icefire", "coolwarm", "cividis", "rocket", "plasma".
    """
    # Display Settings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', '{:.3f}'.format)

    # Visualization Style Configuration
    sns.set_theme(style="whitegrid", palette=palette)

    # Text, Label Scaling & Title Color
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'figure.dpi': 150,
        'axes.titlesize': 18,
        'axes.titlecolor': TITLE_COLOR,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.frameon': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.family': 'DejaVu Sans'
    })

    print("✅ Global visualization settings are ready.")

# ==============================================================================
# 4. GENERIC AUDIT FUNCTIONS (report only - no treatment/decisions made here)
# ==============================================================================

def missing_value_audit(df, dataset_name="dataset", logger=None):
    """
    Returns a summary DataFrame of missing values (count + %) per column.
    Pure reporting - does NOT decide or apply any treatment. Treatment
    decisions (which strategy per column, and why) belong in the notebook.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    summary = pd.DataFrame({
        "Column": missing.index,
        "Missing Count": missing.values,
        "Missing %": (missing.values / len(df) * 100).round(2)
    }).sort_values("Missing %", ascending=False).reset_index(drop=True)

    if logger:
        logger.info(f"[{dataset_name}] Missing value audit: "
                     f"{len(summary)} column(s) with missing values.")
    return summary


def duplicate_check(df, dataset_name="dataset", logger=None):
    """Checks and reports exact duplicate rows in a DataFrame. Report only."""
    dup_count = df.duplicated().sum()
    if logger:
        logger.info(f"[{dataset_name}] Duplicate rows found: {dup_count}")
    return dup_count


# ==============================================================================
# 5. GENERIC STATISTICAL FORMULAS (universal - no dataset-specific assumptions)
# ------------------------------------------------------------------------------
# IQR, Z-score, skewness, and kurtosis are all standard, dataset-agnostic
# formulas - detection/description only. Any treatment decision (cap, drop,
# transform) is made and shown explicitly in the notebook.
# ==============================================================================

def iqr_bounds(df, column):
    """Returns (lower_bound, upper_bound) for a column using the IQR method."""
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def detect_outliers_iqr(df, column):
    """Returns a boolean mask of outliers in `column` using the IQR method."""
    lower, upper = iqr_bounds(df, column)
    return (df[column] < lower) | (df[column] > upper)


def detect_outliers_zscore(df, column, threshold=3):
    """Returns a boolean mask of outliers in `column` using the Z-score method."""
    z_scores = zscore(df[column].dropna())
    return np.abs(z_scores) > threshold


def distribution_summary(df, column):
    """
    Returns a dict of standard dispersion/shape statistics for a numeric
    column: mean, median, std, skewness, kurtosis, and IQR bounds.
    Useful for Non-Graphical Univariate Analysis on any dataset.
    """
    data = df[column].dropna()
    lower, upper = iqr_bounds(df, column)
    return {
        "Column": column,
        "Mean": data.mean(),
        "Median": data.median(),
        "Std Dev": data.std(),
        "Skewness": skew(data),
        "Kurtosis": kurtosis(data),
        "IQR Lower Bound": lower,
        "IQR Upper Bound": upper,
        "Outliers (IQR)": int(detect_outliers_iqr(df, column).sum()),
        "Outliers (Z-score)": int(detect_outliers_zscore(df, column).sum()),
    }