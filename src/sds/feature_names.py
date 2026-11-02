"""
Maps short feature names to display names + units, and defines variable
groupings for multi-panel plots.
"""


FEATURE_NAME_MAP = {
    "t2m":          ("Temperature", "°C"),
    "t2m_roll_30d":  ("Temperature (30d RM)", "°C"),
    "t2m_roll_90d":  ("Temperature (90d RM)", "°C"),
    "t2m_roll_180d": ("Temperature (180d RM)", "°C"),

    "t2m_lag_1d": ("Temperature (Lag1)", "°C"),
    "t2m_lag_2d": ("Temperature (Lag2)", "°C"),
    "t2m_lag_3d": ("Temperature (Lag3)", "°C"),
    "t2m_lag_4d": ("Temperature (Lag4)", "°C"),
    "t2m_lag_5d": ("Temperature (Lag5)", "°C"),
    "t2m_lag_6d": ("Temperature (Lag6)", "°C"),

    "ws_mean":      ("Mean Wind Speed", "m/s"),
    "ws_max":       ("Max Wind Speed", "m/s"),
    "wd":           ("Wind Direction", "°"),
    "rv":           ("Relative Vorticity", "1/s"),
   
    "ws_mean_lag_1d": ("Mean Wind Speed (Lag1)", "m/s"),
    "ws_mean_lag_2d": ("Mean Wind Speed (Lag2)", "m/s"),
    "ws_mean_lag_3d": ("Mean Wind Speed (Lag3)", "m/s"),
    "ws_mean_lag_4d": ("Mean Wind Speed (Lag4)", "m/s"),
    "ws_mean_lag_5d": ("Mean Wind Speed (Lag5)", "m/s"),
    "ws_mean_lag_6d": ("Mean Wind Speed (Lag6)", "m/s"),

    "rh":        ("Relative Humidity", "%"),
    "rh_lag_1d": ("Relative Humidity (Lag1)", "%"),
    "rh_lag_2d": ("Relative Humidity (Lag2)", "%"),
    "rh_lag_3d": ("Relative Humidity (Lag3)", "%"),
    "rh_lag_4d": ("Relative Humidity (Lag4)", "%"),
    "rh_lag_5d": ("Relative Humidity (Lag5)", "%"),
    "rh_lag_6d": ("Relative Humidity (Lag6)", "%"),

    "log_tp":  ("Log Total Precipitation", "log(mm)"),
    "log_tp_roll_30d":  ("Log Total Precipitation (30d RS)", "log(mm)"),
    "log_tp_roll_90d":  ("Log Total Precipitation (90d RS)", "log(mm)"),
    "log_tp_roll_180d": ("Log Total Precipitation (180d RS)", "log(mm)"),

    "log_tp_lag_1d": ("Log Total Precipitation (Lag1)", "log(mm)"),
    "log_tp_lag_2d": ("Log Total Precipitation (Lag2)", "log(mm)"),
    "log_tp_lag_3d": ("Log Total Precipitation (Lag3)", "log(mm)"),
    "log_tp_lag_4d": ("Log Total Precipitation (Lag4)", "log(mm)"),
    "log_tp_lag_5d": ("Log Total Precipitation (Lag5)", "log(mm)"),
    "log_tp_lag_6d": ("Log Total Precipitation (Lag6)", "log(mm)"),

    "NDVI_mean": ("NDVI Mean", "–"),
    "NDVI_std":  ("NDVI Standard Deviation", "–"),

    "acled_total_count_log1p": ("Log ACLED Events", "log(count)"),
    "acled_total_count_30d_sum_log1p":  ("Log ACLED Events (30d RS)", "log(count)"),
    "acled_total_count_180d_sum_log1p":  ("Log ACLED Events (180d RS)", "log(count)"),
    "acled_total_count_365d_sum_log1p": ("Log ACLED Events (365d RS)", "log(count)"),

    "lai_lv":       ("LAI Low Vegetation", "m²/m²"),
    "lai_hv":       ("LAI High Vegetation", "m²/m²"),

    "sdor":         ("Standard Deviation Orography", "m"),
    "slor":         ("Slope Orography", "–"),
    "anor":         ("Anisotropy Orography", "–"),
    "isor":         ("Angle Orography", "rad"),
    "cvl":          ("Low Vegetation Cover", "–"),
    "cvh":          ("High Vegetation Cover", "–"),
    "wind_terrain_align": ("Wind-Terrain Alignment", "–"),

    "NDVI_mean":    ("NDVI Mean", "–"),
    "NDVI_std":     ("NDVI Standard Deviation", "–"),

    "bdod":         ("Bulk Density", "cg/cm³"),
    "cec":          ("Cation Exchange Capacity", "cmol(c)/kg"),
    "clay":         ("Clay Content", "%"),
    "soc":          ("Soil Organic Carbon", "g/kg"),
    "sand":         ("Sand Content", "%"),
    "silt":         ("Silt Content", "%"),
    "fine_fraction": ("Fine Fraction (Silt + Clay)", "%"),
    "sand_to_fines": ("Sand-to-Fines Ratio", "–"),

    "snow_frac" : ("Fraction of Snow/Ice Cover", "%")
}  


def get_full_name(short_name):
    if short_name in FEATURE_NAME_MAP:
        name, unit = FEATURE_NAME_MAP[short_name]
        return f"{name} [{unit}]" if unit != "–" else name

   
def get_feature_name_map(feature_vars):
    """Build a dict mapping short names → full names for a list of features."""
    return {f: get_full_name(f) for f in feature_vars}


VARIABLE_GROUPS = [
    ("Precipitation", [
        "log_tp", "log_tp_lag_1d", "log_tp_lag_2d",
        "log_tp_roll_30d", "log_tp_roll_90d", "log_tp_roll_180d",
    ]),
    ("Relative Humidity", [
        "rh", "rh_lag_1d", "rh_lag_2d", "rh_lag_3d", "rh_lag_4d", "rh_lag_5d",
    ]),
    ("Wind", [
        "ws_mean", "ws_mean_lag_1d", "ws_mean_lag_2d",
        "ws_max", "rv", "wd",
    ]),
    ("Temperature", [
        "t2m", "t2m_lag_1d", "t2m_lag_2d",
        "t2m_roll_30d", "t2m_roll_90d", "t2m_roll_180d",
    ]),
    ("Vegetation", [
        "lai_lv", "lai_hv", "NDVI_mean", "NDVI_std", "cvl", "cvh",
    ]),
    ("Soil Properties", [
        "sand", "silt", "clay", "bdod", "cec", "soc",
        "fine_fraction", "sand_to_fines",
    ]),
    ("Orography", [
        "sdor", "slor", "anor", "isor",
        "wind_terrain_align",
    ]),
    ("Conflict (ACLED)", [
        "acled_total_count_log1p", "acled_total_count_30d_sum_log1p", "acled_total_count_180d_sum_log1p", "acled_total_count_365d_sum_log1p", 
    ]),
]


def get_active_groups(feature_vars):
    """
    Filter VARIABLE_GROUPS to only include features that exist in feature_vars.
    Returns list of (group_title, [existing_features]).
    """
    feat_set = set(feature_vars)
    active = []
    for title, members in VARIABLE_GROUPS:
        existing = [f for f in members if f in feat_set]
        if existing:
            active.append((title, existing))

    return active