import pandas as pd
import json

# ========= Step 1: Load the JSON file =========
with open('data/events_data_all.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# ========= Step 2: Locate the events list robustly =========
def find_events_container(obj):
    """
    Try to find the list of event dicts, regardless of how the API wrapped it.
    Looks for common container keys and for lists that contain dicts with 'details' or 'common'.
    """
    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        # Common container keys seen in APIs
        for key in ['_embedded', 'events', 'items', 'data', 'results', 'records']:
            if key in obj:
                found = find_events_container(obj[key])
                if found is not None:
                    return found

        # Fallback: find any list of dicts that looks like events
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and (
                'details' in v[0] or 'common' in v[0]
            ):
                return v

    return None

events = find_events_container(data)
if not isinstance(events, list):
    raise ValueError("Could not find a list of events in the JSON payload.")

# Ensure 'details' exists even if empty (prevents record_path errors on some pandas versions)
for ev in events:
    if not isinstance(ev.get('details'), list):
        ev['details'] = []

# ========= Step 3: Normalize with correct meta paths (no meta_prefix) =========
# IMPORTANT: remove meta_prefix to avoid 'common._id' and 'common.common.*'
ev_df = pd.json_normalize(
    events,
    record_path='details',
    meta=[
        '_id',                         # top-level meta (remains '_id')
        'itemId',                      # top-level meta
        'lastModified',                # top-level meta
        ['common', 'publishDate'],     # nested meta under 'common'
        ['common', 'dateFrom'],
        ['common', 'dateTo'],
        ['common', 'posterUrl'],
        ['common', 'timeFrom'],
        ['common', 'timeTo'],
        ['common', 'recurringDaysOfWeek'],
        ['common', 'attachmentUrls'],
        ['common', 'eventUrl'],
        ['common', 'eventRegistrationUrl']
    ],
    errors='ignore'  # ignore missing meta fields
)

# ========= Step 4: Flatten any remaining list-type columns =========
for col in ev_df.columns:
    if ev_df[col].apply(lambda x: isinstance(x, list)).any():
        ev_df[col] = ev_df[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)

# ========= Step 5: EDA (safer checks) =========
print("=== EXPLORATORY DATA ANALYSIS ===")

id_col = None
for candidate in ['_id', 'common._id']:
    if candidate in ev_df.columns:
        id_col = candidate
        break

if id_col:
    print(f"Total unique events by '{id_col}': {ev_df[id_col].nunique()}")
else:
    print("Warning: Neither '_id' nor 'common._id' columns found.")
    print("Available columns:", ev_df.columns.tolist())

print(f"Total language versions: {len(ev_df)}")

if 'locale' in ev_df.columns:
    print(f"Available languages: {sorted(ev_df['locale'].dropna().unique().tolist())}")
    print(f"Events per language:\n{ev_df['locale'].value_counts(dropna=False).to_dict()}")

print(f"\nShape: {ev_df.shape[0]} rows, {ev_df.shape[1]} columns")

# Show missing values only for top problematic columns
missing = ev_df.isnull().mean().sort_values(ascending=False)
print(f"\nTop 10 Columns with Most Missing Values:")
print(missing.head(10))

# ========= Step 6: Save the full CSV =========
ev_df.to_csv('data/events_data_all.csv', index=False, encoding='utf-8-sig')
print("\n✅ CSV saved as 'data/events_data_all.csv'")

# ========= Step 7: Split by locale into ev_df_en, ev_df_cn (zh_TW), ev_df_pt =========
# NOTE:
#   en_US -> English
#   zh_TW -> Chinese (Traditional). You asked for ev_df_cn; we'll map zh_TW to ev_df_cn as requested.
#   pt_PT -> Portuguese (Portugal)
def safe_slice_locale(frame, locale_value):
    if 'locale' not in frame.columns:
        return pd.DataFrame(columns=frame.columns)
    return frame[frame['locale'] == locale_value].copy()

ev_df_en = safe_slice_locale(ev_df, 'en_US')
ev_df_cn = safe_slice_locale(ev_df, 'zh_TW')  # as requested, using zh_TW for Chinese
ev_df_pt = safe_slice_locale(ev_df, 'pt_PT')

print("\n=== SPLIT SUMMARY ===")
print(f"ev_df_en: {len(ev_df_en)} rows  | locale=en_US")
print(f"ev_df_cn: {len(ev_df_cn)} rows  | locale=zh_TW")
print(f"ev_df_pt: {len(ev_df_pt)} rows  | locale=pt_PT")

# Optional: save per-locale CSVs
ev_df_en.to_csv('data/events_en_US.csv', index=False, encoding='utf-8-sig')
ev_df_cn.to_csv('data/events_zh_TW.csv', index=False, encoding='utf-8-sig')
ev_df_pt.to_csv('data/events_pt_PT.csv', index=False, encoding='utf-8-sig')
print("\n✅ Per-locale CSVs saved: events_en_US.csv, events_zh_TW.csv, events_pt_PT.csv")

# ========= Step 8: Show all column names (cleaned) =========
print("\n📋 ALL COLUMN NAMES:")
print(ev_df.columns.tolist())