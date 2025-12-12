import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def _parse_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse date columns and fix common issues (missing end, inverted ranges)."""
    df_ = frame.copy()

    # Parse the date columns safely
    for col in ['common.dateFrom', 'common.dateTo']:
        if col in df_.columns:
            df_[col] = pd.to_datetime(df_[col], errors='coerce', utc=True).dt.tz_localize(None)
        else:
            df_[col] = pd.NaT

    # If dateTo is missing, use dateFrom (single-day event)
    df_['common.dateTo'] = df_['common.dateTo'].fillna(df_['common.dateFrom'])

    # Drop rows without start dates
    df_ = df_.dropna(subset=['common.dateFrom']).copy()

    # Ensure start <= end (swap if needed)
    mask_swap = (df_['common.dateFrom'] > df_['common.dateTo'])
    df_.loc[mask_swap, ['common.dateFrom', 'common.dateTo']] = \
        df_.loc[mask_swap, ['common.dateTo', 'common.dateFrom']].values

    return df_


def _expand_to_days(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand each event row to one row per day in its [dateFrom, dateTo] range."""
    df_ = frame.copy()

    # Build a daily date_range per row and explode
    # NOTE: if your data is very large, this can expand a lot. It's still the simplest, most explicit approach.
    df_['event_day'] = df_.apply(
        lambda r: pd.date_range(r['common.dateFrom'].date(), r['common.dateTo'].date(), freq='D')
        if pd.notna(r['common.dateFrom']) and pd.notna(r['common.dateTo']) else pd.DatetimeIndex([]),
        axis=1
    )

    df_ = df_.explode('event_day', ignore_index=True)
    # Guard against accidental empties
    df_ = df_.dropna(subset=['event_day'])

    return df_


def _dedupe_by_event_day(frame: pd.DataFrame, preferred_id_cols=('itemId', '_id')) -> pd.DataFrame:
    """
    Deduplicate records so that the same event isn't counted multiple times on the same day.
    We try 'itemId' first (usually stable across locales), then fallback to '_id' if needed.
    """
    df_ = frame.copy()
    id_col = None
    for c in preferred_id_cols:
        if c in df_.columns:
            id_col = c
            break

    if id_col is not None:
        df_ = df_.drop_duplicates(subset=[id_col, 'event_day'])
    else:
        # Fallback: if no ID columns exist, dedupe by (title, event_day) as a last resort
        if 'title' in df_.columns:
            df_ = df_.drop_duplicates(subset=['title', 'event_day'])
        else:
            df_ = df_.drop_duplicates(subset=['event_day'])

    return df_


def build_pivot_events_per_doy_by_year(frame: pd.DataFrame,
                                       dedupe=True,
                                       preferred_id_cols=('itemId', '_id')) -> pd.DataFrame:
    """
    Build a pivot where:
      - index = day-of-year (1..365/366)
      - columns = year (int)
      - values = count of events active that day
    """
    # Parse dates and expand
    df_ = _parse_dates(frame)
    if df_.empty:
        return pd.DataFrame()

    df_ = _expand_to_days(df_)

    # Optional: dedupe the same event per day (useful for all-locale views)
    if dedupe:
        df_ = _dedupe_by_event_day(df_, preferred_id_cols=preferred_id_cols)

    # Year and day-of-year
    df_['year'] = df_['event_day'].dt.year
    df_['doy'] = df_['event_day'].dt.dayofyear

    # Group and pivot
    grp = df_.groupby(['doy', 'year'], as_index=False).size()
    pivot = grp.pivot(index='doy', columns='year', values='size').sort_index()

    # Fill missing days to make plotting cleaner (from 1 to max in this frame)
    if not pivot.empty:
        full_index = pd.RangeIndex(1, int(pivot.index.max()) + 1)
        pivot = pivot.reindex(full_index)
        pivot.index.name = 'doy'
        pivot = pivot.fillna(0)

    return pivot


def plot_pivot_grey_lines(pivot: pd.DataFrame, title: str,
                          outfile: str = 'events_per_day_by_year.png'):
    """Plot grey polylines and fills for each year with month ticks based on day-of-year."""
    if pivot is None or pivot.empty:
        print(f"[WARN] Nothing to plot for: {title}")
        return

    sns.set(style='whitegrid')
    plt.figure(figsize=(14, 7))

    # Draw each year as a lightly transparent grey line + fill
    for y in pivot.columns:
        color = 'grey'
        plt.plot(
            pivot.index, pivot[y],
            label=str(y),
            linewidth=2,
            alpha=0.15,
            color=color
        )
        plt.fill_between(
            pivot.index,
            pivot[y],
            0,
            color=color,
            alpha=0.08
        )

    # --- Month ticks based on day-of-year ---
    max_doy = int(pd.Index(pivot.index).max())
    # Use leap year if 366 appears
    ref_year = 2020 if max_doy >= 366 else 2021

    month_starts = pd.date_range(f'{ref_year}-01-01', f'{ref_year}-12-31', freq='MS')
    tick_pos = month_starts.dayofyear.to_list()
    tick_labels = month_starts.strftime('%b').to_list()

    ax = plt.gca()
    ax.set_xlim(1, max_doy)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)

    plt.title(title, fontsize=16)
    plt.xlabel('Month (based on day-of-year)', fontsize=12)
    plt.ylabel('Number of Events', fontsize=12)
    plt.tight_layout()

    plt.savefig(outfile, dpi=200)
    print(f"✅ Plot saved: {outfile}")


# =========================
# Build and plot: overall
# =========================

# Load the data (assuming it's already cleaned)
ev_df = pd.read_csv('data/events_data_all.csv', encoding='utf-8')

# IMPORTANT: Using dedupe=True avoids double-counting across locales in the 'ev_df' (full dataset).
pivot_all = build_pivot_events_per_doy_by_year(ev_df, dedupe=True, preferred_id_cols=('itemId', '_id'))
plot_pivot_grey_lines(pivot_all, title='Events per Day of Year by Year — All Locales (deduped by itemId)',
                      outfile='visuals/events_per_day_by_year_all_locales.png')

# =========================
# Build and plot: per locale
# =========================

# For per-locale dataframes we usually do NOT need dedupe (each is a single language view).
ev_df_en = pd.read_csv('data/events_en_US.csv', encoding='utf-8')
pivot_en = build_pivot_events_per_doy_by_year(ev_df_en, dedupe=False)
plot_pivot_grey_lines(pivot_en, title='Events per Day of Year by Year — en_US',
                      outfile='visuals/events_per_day_by_year_en_US.png')

ev_df_cn = pd.read_csv('data/events_zh_TW.csv', encoding='utf-8')
pivot_cn = build_pivot_events_per_doy_by_year(ev_df_cn, dedupe=False)
plot_pivot_grey_lines(pivot_cn, title='Events per Day of Year by Year — zh_TW',
                      outfile='visuals/events_per_day_by_year_zh_TW.png')

ev_df_pt = pd.read_csv('data/events_pt_PT.csv', encoding='utf-8')
pivot_pt = build_pivot_events_per_doy_by_year(ev_df_pt, dedupe=False)
plot_pivot_grey_lines(pivot_pt, title='Events per Day of Year by Year — pt_PT',
                      outfile='visuals/events_per_day_by_year_pt_PT.png')