# var-si-ty
Spatial Temporal Social Events for All

## Project Overview
This project processes and analyzes events and courses data from the University of Macau API, performing data cleaning, visualization, faculty detection, and topic modeling.

## Setup
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the complete pipeline:
   ```bash
   python main.py
   ```

## Project Structure
- `data/` - Directory for data files and outputs
- `scripts/` - Individual Python scripts for each processing step
- `main.py` - Main orchestrator script
- `requirements.txt` - Python dependencies

## Data Pipeline Details

This project implements a comprehensive data processing pipeline that fetches, cleans, analyzes, and visualizes events and courses data from the University of Macau API. The pipeline consists of 9 sequential scripts orchestrated by `main.py`.

### Pipeline Overview

1. **Data Fetching**: Retrieve raw data from UM APIs
2. **Data Cleaning**: Process and normalize JSON data into structured CSVs
3. **Event Analysis**: Extract metadata, detect faculties, and create visualizations
4. **Course Analysis**: Fetch courses, clean data, and perform topic modeling

### Scripts Detailed Overview

#### 1. `data_fetch_events.py` - Events Data Fetching
**Purpose**: Downloads events data from the University of Macau events API with incremental updates.

**Key Variables**:
- `BASE_URL`: `"https://api.data.um.edu.mo/service/media/events/all"` - API endpoint
- `AUTH_KEY`: API subscription key for authentication
- `OUT_FILE`: `"data/events_data_all.json"` - Output JSON file
- `total_pages`, `pagesize`: Pagination metadata from API
- `idx_by_id`: Dictionary mapping `itemId` to array indices for duplicate detection

**Process**:
- Makes initial `?count` request to get pagination info
- Fetches data page by page, handling duplicates by `itemId`
- Updates existing records or prepends new ones
- Saves merged data with `"_embedded"` array and `"_returned"` count

**Output**: `data/events_data_all.json` (raw events data)

#### 2. `data_clean_events.py` - Events Data Cleaning
**Purpose**: Normalizes JSON events data into a flat CSV structure with proper data types.

**Key Variables**:
- `data`: Loaded JSON object from API response
- `events`: Extracted array of event objects from `data["_embedded"]`
- `ev_df`: Main pandas DataFrame after normalization
- `ev_df_en`, `ev_df_cn`, `ev_df_pt`: Locale-specific DataFrames (en_US, zh_TW, pt_PT)

**Process**:
- Uses `pd.json_normalize()` with `record_path='details'` and meta fields like `itemId`, `lastModified`, `common.dateFrom`
- Flattens list-type columns (e.g., `attachmentUrls`)
- Performs EDA: checks unique events by `_id`, language distribution
- Splits data by `locale` column into separate CSVs

**Outputs**:
- `data/events_data_all.csv` (all locales combined)
- `data/events_en_US.csv`, `data/events_zh_TW.csv`, `data/events_pt_PT.csv` (per locale)

#### 3. `visualize_events.py` - Events Visualization
**Purpose**: Creates temporal visualizations of event distributions across years and locales.

**Key Functions**:
- `_parse_dates()`: Converts date strings to datetime, handles missing end dates
- `_expand_to_days()`: Expands each event into daily rows from `dateFrom` to `dateTo`
- `_dedupe_by_event_day()`: Removes duplicate events per day using `itemId` or `_id`
- `build_pivot_events_per_doy_by_year()`: Creates day-of-year by year pivot table
- `plot_pivot_grey_lines()`: Generates line plots with month-based x-axis

**Key Variables**:
- `pivot_all`, `pivot_en`, `pivot_cn`, `pivot_pt`: Pivot tables for different scopes
- `event_day`: Expanded daily datetime index per event
- `doy` (day-of-year), `year`: Derived temporal features

**Process**:
- Parses and expands date ranges into daily events
- Deduplicates across locales for combined views
- Creates pivot tables counting events per day-of-year by year
- Plots grey line charts with month ticks

**Outputs**:
- `visuals/events_per_day_by_year_all_locales.png`
- `visuals/events_per_day_by_year_en_US.png`
- `visuals/events_per_day_by_year_zh_TW.png`
- `visuals/events_per_day_by_year_pt_PT.png`

#### 4. `extract_organized_by.py` - OrganizedBy Extraction
**Purpose**: Extracts and compares `organizedBy` values across different language locales.

**Key Functions**:
- `extract_from_csv()`: Processes individual locale CSV files
- `extract_from_combined_csv()`: Handles combined CSV with `locale` column
- `print_results()`: Displays unique values and creates comparison grid

**Key Variables**:
- `organized_by_col`: Detected column name (case-insensitive match)
- `results`: Dictionary with locale keys (`en_US`, `zh_TW`, `pt_PT`) mapping to sets of values
- `ev_df`: Pandas DataFrame loaded from CSV

**Process**:
- Searches for `organizedBy` column across available CSV files
- Extracts unique non-null values per locale
- Creates comparison grid and saves to CSV

**Output**: `data/organizedBy_by_language_comparison.csv`

#### 5. `faculty_detection.py` - Faculty Detection
**Purpose**: Automatically detects faculty affiliations from `organizedBy` text using regex patterns.

**Key Variables**:
- `FACULTY_SIGNALS`: Dictionary mapping faculty codes (FAH, FBA, FED, etc.) to regex patterns
- `COMPILED`: Pre-compiled regex objects for performance
- `ev_df['faculties_list']`: List of detected faculty codes per event
- `ev_df['faculties']`: Comma-separated string of faculty codes

**Faculty Patterns Include**:
- **FAH**: Humanities, arts, languages, translation
- **FBA**: Business, finance, management, tourism
- **FED**: Education, teaching, pedagogy
- **FHS**: Health sciences, medicine, biomedical
- **FLL**: Law, legal studies, constitutional law
- **FST**: Engineering, computer science, physics, math
- **FSS**: Social sciences, psychology, economics, communication

**Process**:
- Applies `detect_faculties()` function to each `organizedBy` value
- Uses case-insensitive regex matching against expanded signal patterns
- Calculates coverage statistics and saves enhanced dataset

**Output**: `data/events_en_US_cleaned.csv` (with faculty assignments)

#### 6. `visualize_faculty.py` - Faculty Visualization
**Purpose**: Creates visualizations of faculty event distributions and temporal patterns.

**Key Variables**:
- `faculty_counts`: Dictionary counting events per faculty (multi-label aware)
- `ev_df_with_dates`: Filtered DataFrame with valid parsed dates
- `month_counts`: Monthly event aggregation
- `heatmap_pivot`: Faculty × Month matrix for heatmap

**Visualizations**:
1. **Faculty Distribution**: Bar chart + pie chart of events per faculty
2. **Monthly Distribution**: Bar chart + line chart of events by month
3. **Faculty-Month Heatmap**: Cross-tabulation visualization

**Process**:
- Parses dates and extracts month/year features
- Counts multi-label faculty assignments
- Creates matplotlib/seaborn plots with custom styling

**Outputs**:
- `visuals/faculty_distribution.png`
- `visuals/monthly_distribution.png`
- `visuals/faculty_month_heatmap.png`

#### 7. `data_fetch_courses.py` - Courses Data Fetching
**Purpose**: Downloads courses catalog data from University of Macau courses API.

**Key Variables**:
- `BASE_URL`: `"https://api.data.um.edu.mo/service/academic/course_catalog/all"`
- `AUTH_KEY`: Same API key as events
- `OUT_FILE`: `"data/courses_catalog_data_all.json"`
- `idx_by_id`: Dictionary mapping `courseCode` for duplicate detection

**Process**: Similar to events fetching but uses `courseCode` for deduplication instead of `itemId`.

**Output**: `data/courses_catalog_data_all.json`

#### 8. `data_clean_courses.py` - Courses Data Cleaning
**Purpose**: Normalizes courses JSON data into structured CSV format.

**Key Variables**:
- `courses`: Extracted array from `data["_embedded"]`
- `df`: Main pandas DataFrame after normalization
- Meta fields: `courseCode`, `courseTitle`, `offeringUnit`, `courseDescription`, `ilo`

**Process**:
- Uses `pd.json_normalize()` to flatten course objects
- Removes duplicates by `courseCode`
- Performs EDA on data completeness

**Output**: `data/courses_data_all.csv`

#### 9. `topic_modeling.py` - BERTopic Modeling
**Purpose**: Applies BERTopic clustering to discover course topics from text content.

**Key Functions**:
- `prepare_documents()`: Combines `courseCode`, `courseTitle`, `courseDescription`, `ilo` into documents
- `setup_bertopic()`: Configures BERTopic with sentence transformers, UMAP, HDBSCAN
- `find_similar_courses()`: Discovers similar courses using embeddings

**Key Variables**:
- `documents`: List of concatenated text documents per course
- `topic_model`: BERTopic model instance
- `topics`: Array of topic assignments per document
- `probabilities`: Topic probability distributions
- `df['topic']`, `df['topic_probability']`: Added columns to course DataFrame

**Model Configuration**:
- Embedding: `all-MiniLM-L6-v2` sentence transformer
- Dimensionality reduction: UMAP (15 neighbors, 5 components)
- Clustering: HDBSCAN (min_cluster_size=10)
- Vectorization: CountVectorizer with custom stop words

**Process**:
- Prepares text documents from course metadata
- Fits BERTopic model to discover topics
- Assigns topics and probabilities to courses
- Saves model and enhanced course data

**Outputs**:
- `data/courses_with_topics.csv` (courses with topic assignments)
- `data/bertopic_model/` (saved BERTopic model directory)

## Output Files
- `data/events_data_all.json/csv` - Raw and processed events data
- `data/courses_catalog_data_all.json/csv` - Raw and processed courses data
- Various PNG visualization files in `data/`
- `data/bertopic_model/` - Trained topic model
- `data/courses_with_topics.csv` - Courses with topic assignments
