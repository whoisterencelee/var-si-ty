import pandas as pd
import json

# ========= Step 1: Load the JSON file =========
try:
    with open( 'data/courses_catalog_data_all.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except UnicodeDecodeError:
    # Try with latin-1 if utf-8 fails
    with open( 'data/courses_catalog_data_all.json', 'r', encoding='latin-1') as f:
        data = json.load(f)

# ========= Step 2: Locate the course list robustly =========
def find_courses_container(obj):
    """
    Try to find the list of event dicts, regardless of how the API wrapped it.
    Looks for common container keys and for lists that contain dicts with 'details' or 'common'.
    """
    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        # Common container keys seen in APIs
        for key in ['_embedded' ]:
            if key in obj:
                return obj[key]

        # Fallback: find any list of dicts that looks like courses
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and (
                'details' in v[0] or 'common' in v[0]
            ):
                return v

    return None

courses = find_courses_container(data)
if not isinstance(courses, list):
    raise ValueError("Could not find a list of courses in the JSON payload.")

# ========= Step 3: Normalize with correct meta paths (no meta_prefix) =========
df = pd.json_normalize(
    courses,
    meta=[
        'courseCode',
        'itemId',
        'lastModified',
        'courseCode',
        'oldCourseCode',
        'courseTitle',
        'offeringUnit',
        'offeringProgLevel',
        'courseType',
        'suggestedYearOfStudy',
        'duration',
        'gradingSystem',
        'mediumOfInstruction',
        'courseDescription',
        'ilo'
    ],
    errors='ignore'  # ignore missing meta fields
)

# ========= Drop duplicates =============
df = df.drop_duplicates(keep='last',subset='courseCode')  # remove duplicates

# ========= Step 5: EDA (safer checks) =========
print("=== EXPLORATORY DATA ANALYSIS ===")

id_col = None
for candidate in ['courseCode']:
    if candidate in df.columns:
        id_col = candidate
        break

if id_col:
    print(f"Total unique courses by '{id_col}': {df[id_col].nunique()}")
else:
    print("Warning: 'courseCode' not found.")
    print("Available columns:", df.columns.tolist())

print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")

# Show missing values only for top problematic columns
missing = df.isnull().mean().sort_values(ascending=False)
print(f"\nTop 10 Columns with Most Missing Values:")
print(missing.head(10))

# ========= Step 6: Save the full CSV =========
df.to_csv('data/courses_data_all.csv', index=False, encoding='utf-8-sig')
print("\n✅ CSV saved as 'data/courses_data_all.csv'")

# ========= Step 8: Show all column names (cleaned) =========
print("\n📋 ALL COLUMN NAMES:")
print(df.columns.tolist())