import pandas as pd
import os

def extract_from_csv(filename, locale_name):
    """Extract organizedBy values from a CSV file."""
    print(f"\nProcessing: {filename}")

    if not os.path.exists(filename):
        print(f"  ⚠ File not found")
        return set()

    # Load the CSV
    ev_df = pd.read_csv(filename, encoding='utf-8')
    print(f"  ✓ Loaded {len(ev_df)} rows")
    print(f"  Columns: {list(ev_df.columns)[:10]}")  # Show first 10 columns

    # Look for organizedBy column (case-insensitive)
    organized_by_col = None
    for col in ev_df.columns:
        if 'organizedby' in col.lower():
            organized_by_col = col
            break

    if organized_by_col is None:
        print(f"  ⚠ 'organizedBy' column not found")
        return set()

    # Extract unique non-null values
    unique_values = set()
    values = ev_df[organized_by_col].dropna().unique()
    unique_values.update([str(v) for v in values if str(v).strip() and str(v) != 'nan'])

    print(f"  ✓ Found {len(unique_values)} unique organizedBy values")
    return unique_values

def extract_from_combined_csv(filename):
    """Extract organizedBy values from combined CSV with locale column."""
    print(f"\nProcessing combined file: {filename}")

    if not os.path.exists(filename):
        print(f"  ⚠ File not found")
        return None

    # Load the CSV
    ev_df = pd.read_csv(filename, encoding='utf-8')
    print(f"  ✓ Loaded {len(ev_df)} rows")
    print(f"  Columns: {list(ev_df.columns)}")

    # Look for locale/language column
    locale_col = None
    for col in ev_df.columns:
        if col.lower() in ['locale', 'language', 'lang']:
            locale_col = col
            break

    # Look for organizedBy column
    organized_by_col = None
    for col in ev_df.columns:
        if 'organizedby' in col.lower():
            organized_by_col = col
            break

    if organized_by_col is None:
        print(f"  ⚠ 'organizedBy' column not found")
        return None

    results = {
        'en_US': set(),
        'zh_TW': set(),
        'pt_PT': set()
    }

    if locale_col:
        print(f"  ✓ Using locale column: '{locale_col}'")
        print(f"  Available locales: {ev_df[locale_col].unique()}")

        # Extract by locale
        for locale in ['en_US', 'zh_TW', 'pt_PT']:
            locale_df = ev_df[ev_df[locale_col] == locale]
            if len(locale_df) > 0:
                values = locale_df[organized_by_col].dropna().unique()
                results[locale].update([str(v) for v in values if str(v).strip() and str(v) != 'nan'])
                print(f"  ✓ {locale}: {len(results[locale])} unique values from {len(locale_df)} rows")
    else:
        # No locale column - treat all data as English
        print(f"  ⚠ No locale column found, treating all data as English")
        values = ev_df[organized_by_col].dropna().unique()
        results['en_US'].update([str(v) for v in values if str(v).strip() and str(v) != 'nan'])
        print(f"  ✓ Found {len(results['en_US'])} unique values")

    return results

def print_results(results):
    """Print results in list and grid formats."""
    print("\n" + "=" * 80)
    print("UNIQUE 'organizedBy' VALUES BY LANGUAGE")
    print("=" * 80)

    language_names = {
        'en_US': 'ENGLISH',
        'zh_TW': 'CHINESE (TRADITIONAL)',
        'pt_PT': 'PORTUGUESE'
    }

    sorted_values = {}

    for locale, lang_name in language_names.items():
        print(f"\n### {lang_name} ({locale}) ###")
        print(f"Total unique values: {len(results[locale])}")

        sorted_list = sorted(list(results[locale]))
        sorted_values[locale] = sorted_list

        if sorted_list:
            for i, value in enumerate(sorted_list, 1):
                print(f"{i:3d}. {value}")
        else:
            print("  (No values found)")

        print("-" * 80)

    # Grid format
    print("\n" + "=" * 80)
    print("GRID FORMAT COMPARISON")
    print("=" * 80)

    max_len = max(len(sorted_values['en_US']),
                  len(sorted_values['zh_TW']),
                  len(sorted_values['pt_PT']))

    if max_len > 0:
        # Pad lists
        en_padded = sorted_values['en_US'] + [''] * (max_len - len(sorted_values['en_US']))
        cn_padded = sorted_values['zh_TW'] + [''] * (max_len - len(sorted_values['zh_TW']))
        pt_padded = sorted_values['pt_PT'] + [''] * (max_len - len(sorted_values['pt_PT']))

        ev_df = pd.DataFrame({
            'English (en_US)': en_padded,
            'Chinese (zh_TW)': cn_padded,
            'Portuguese (pt_PT)': pt_padded
        })

        print("\n")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            print(ev_df.to_string(index=True))

        # Save to CSV
        csv_filename = 'data/organizedBy_by_language_comparison.csv'
        ev_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"\n✅ Grid comparison saved to '{csv_filename}'")
    else:
        print("\n⚠ No data found to display in grid format")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    for locale, lang_name in language_names.items():
        print(f"{lang_name:30s} ({locale}): {len(results[locale]):3d} unique values")

    total_unique = len(results['en_US'] | results['zh_TW'] | results['pt_PT'])
    print(f"{'Total across all languages':30s}        : {total_unique:3d} unique values")

def main():
    """Main function."""
    print("=" * 80)
    print("EXTRACT 'organizedBy' VALUES BY LANGUAGE")
    print("=" * 80)

    # Strategy 1: Try separate CSV files for each locale
    separate_files = {
        'en_US': 'data/events_en_US.csv',
        'zh_TW': 'data/events_zh_TW.csv',
        'pt_PT': 'data/events_pt_PT.csv'
    }

    # Check if separate files exist
    separate_files_exist = all(os.path.exists(f) for f in separate_files.values())

    if separate_files_exist:
        print("\n✓ Found separate locale-specific CSV files")
        results = {}
        for locale, filename in separate_files.items():
            lang_name = {'en_US': 'English', 'zh_TW': 'Chinese', 'pt_PT': 'Portuguese'}[locale]
            results[locale] = extract_from_csv(filename, lang_name)

        print_results(results)
        return

    # Strategy 2: Try combined CSV file
    combined_file = 'data/events_data_all.csv'
    if os.path.exists(combined_file):
        print(f"\n✓ Found combined CSV file: {combined_file}")
        results = extract_from_combined_csv(combined_file)
        if results:
            print_results(results)
            return

    # Strategy 3: Try other possible filenames
    possible_files = [
        'events.csv',
        'events_all.csv',
        'all_events.csv'
    ]

    for filename in possible_files:
        if os.path.exists(filename):
            print(f"\n✓ Found data file: {filename}")
            results = extract_from_combined_csv(filename)
            if results:
                print_results(results)
                return

    # No files found
    print("\n❌ No event data files found!")
    print("\nPlease make sure one of these files exists:")
    print("  • data/events_data_all.csv (combined file with locale column)")
    print("  OR")
    print("  • data/events_en_US.csv")
    print("  • data/events_zh_TW.csv")
    print("  • data/events_pt_PT.csv")

    # Show what files are in directory
    print("\nCurrent directory files:")
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if files:
        for f in files:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  • {f} ({size_mb:.2f} MB)")
    else:
        print("  (No CSV files found)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()