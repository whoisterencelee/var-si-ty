import re
import pandas as pd
import os

# EXPANDED Faculty signal patterns for University of Macau
# Added more keywords based on typical event types, activities, and topics for each faculty
FACULTY_SIGNALS = {
    "FAH": [
        # Original patterns
        r"\bhumanities?\b",
        r"\barts?\b(?!\s+and\s+physical\s+education)",
        r"\benglish language centre\b",
        r"\belc\b",
        r"\bdepartment of english\b",
        r"\bdepartment of chinese( language and literature)?\b",
        r"\bdepartment of portuguese\b",
        r"\bphilosophy\b",
        r"\breligious studies\b",
        r"\bcentre for japanese studies\b",
        r"\bcentre for luso[- ]asian studies\b",
        r"\bcchc\b",
        r"\bcentre for chinese history and culture\b",
        r"\bcstic\b",
        r"\bcentre for studies of translation, interpreting and cognition\b",
        r"\bconfucius institute\b",
        r"\bresearch centre for humanities in south china\b",
        # NEW: Department/discipline keywords
        r"\bhistory\b",
        r"\bhistorical\b",
        r"\bart(s)? and design\b",
        r"\bjapanese( studies| language)?\b",
        r"\btranslat(ion|ing)\b",
        r"\binterpret(ing|ation)\b",
        r"\blinguistics?\b",
        r"\bliterary\b",
        r"\bliterature\b",
        r"\bpoetry\b",
        r"\bcultur(e|al)\b(?!.*tourism)",
        r"\bheritage\b",
        r"\blanguage learning\b",
        # NEW: Event types
        r"\bfilm( screening| festival)?\b",
        r"\bexhibition\b",
        r"\bmuseum\b",
        r"\bgallery\b",
        r"\bcreative writing\b",
        r"\breading\b(?!.*literacy)",
        r"\bbook( launch| talk| club)?\b",
        r"\bwriting workshop\b",
        r"\bchinese[- ]portuguese( bilingual)?\b",
        r"\bluso[- ]asian\b",
        r"\bmacanese culture\b",
        r"\bcalligraphy\b",
        r"\bperforming arts\b",
        r"\btheatre\b",
        r"\bdrama\b",
        r"\bmusic(?!.*business)\b",
    ],
    "FBA": [
        # Original patterns
        r"\baccounting\b",
        r"\bfinance\b",
        r"\bmarketing\b",
        r"\bmanagement\b",
        r"\bintegrated resort\b",
        r"\btourism management\b",
        r"\bcommercial gaming\b",
        r"\bbrtc\b",
        r"\bbusiness research and training center\b",
        r"\b(private|greater china) centre for private equity\b",
        r"\binstitute for (the )?study of commercial gaming\b",
        # NEW: Disciplines
        r"\bbusiness( administration)?\b",
        r"\bcommerce\b",
        r"\bentrepreneurship\b",
        r"\binformation management\b",
        r"\bMBA\b",
        r"\bdata analytics?\b",
        r"\bbusiness intelligence\b",
        # NEW: Topics
        r"\bcasino\b",
        r"\bgaming industry\b",
        r"\bhospitality\b",
        r"\btourism\b",
        r"\bhotel\b",
        r"\bresort\b",
        r"\binvestment\b",
        r"\bprivate equity\b",
        r"\bfinancial\b",
        r"\bstock market\b",
        r"\bcorporate\b",
        r"\bstartup\b",
        r"\bsupply chain\b",
        r"\boperations\b",
        r"\bstrategy\b",
        r"\bleadership\b",
        r"\bcareer development\b",
        r"\bprofessional development\b",
        r"\bnetworking\b(?!.*computer)",
        r"\bbusiness plan\b",
    ],
    "FED": [
        # Original patterns
        r"\beducation(al)?\b",
        r"\beducational research centre\b",
        r"\bcape\b",
        r"\bcentre for arts and physical education\b",
        r"\bclled\b",
        r"\bcentre for language and literacy education\b",
        r"\bcentre for stem and ai education\b",
        r"\bliteracy\b",
        # NEW: Disciplines
        r"\bteaching\b",
        r"\bpedagogy\b",
        r"\bcurriculum\b",
        r"\beducational psychology\b",
        r"\bearly childhood\b",
        r"\bpreschool\b",
        r"\bprimary education\b",
        r"\bsecondary education\b",
        r"\bstem education\b",
        # NEW: Topics
        r"\bteacher( training| development)?\b",
        r"\bstudent( learning| development)?\b",
        r"\bclassroom\b",
        r"\bphysical education\b",
        r"\bsports? education\b",
        r"\bassessment\b(?!.*psychological)",
        r"\btesting( and assessment)?\b",
        r"\beducational technology\b",
        r"\be-learning\b",
        r"\binstructional design\b",
        r"\bspecial education\b",
        r"\binclusive education\b",
        r"\bmultilingual education\b",
        r"\bcounsell?ing\b(?!.*clinical)",
    ],
    "FHS": [
        # Original patterns
        r"\bhealth sciences?\b",
        r"\bbiomedical\b",
        r"\boncology\b",
        r"\bprecision oncology\b",
        r"\bfscpo\b",
        r"\bfrontiers science center for precision oncology\b",
        r"\bicms\b",
        r"\binstitute of chinese medical sciences\b",
        r"\bbgi\b",
        r"\bbeijing genomics institute\b",
        r"\bcancer\b",
        # NEW: Disciplines
        r"\bpharmacy\b",
        r"\bpharmaceutical\b",
        r"\bpublic health\b",
        r"\bmolecular biology\b",
        r"\bbiomedicine\b",
        r"\bchinese medicine\b",
        r"\btraditional medicine\b",
        r"\btcm\b",
        r"\bacupuncture\b",
        r"\bherbal medicine\b",
        # NEW: Topics
        r"\bmedical\b",
        r"\bclinical\b(?!.*psychology)",
        r"\bhospital\b",
        r"\bhealth(care| care)?\b",
        r"\bdrug\b",
        r"\bgenomics?\b",
        r"\bstem cell\b",
        r"\bregenerative medicine\b",
        r"\btranslational medicine\b",
        r"\bprecision medicine\b",
        r"\bepidemic\b",
        r"\bpandemic\b",
        r"\binfectious disease\b",
        r"\bvaccine\b",
        r"\bpatient\b",
        r"\bdiagnos(is|tic)\b",
        r"\btherapy\b",
        r"\btreatment\b",
        r"\bbioinformatics\b",
        r"\bsystematic biology\b",
    ],
    "FLL": [
        # Original patterns
        r"\blaw\b",
        r"\blegal\b",
        r"\bconstitutional\b",
        r"\bbasic law\b",
        r"\bmoot\b",
        r"\badvocacy\b",
        r"\bcentre for law studies\b",
        r"\binstitute for advanced legal studies\b",
        r"\bcentre for constitutional law and basic law studies\b",
        # NEW: Disciplines
        r"\bjuris\b",
        r"\bjurisprudence\b",
        r"\blegislat(ion|ive)\b",
        r"\bcriminal law\b",
        r"\bcivil law\b",
        r"\binternational law\b",
        r"\beuropean( union)? law\b",
        r"\bcomparative law\b",
        r"\bcommercial law\b",
        r"\bbusiness law\b",
        r"\bcorporate law\b",
        r"\bcontract law\b",
        # NEW: Topics
        r"\blawsuit\b",
        r"\blitigation\b",
        r"\bcourt\b",
        r"\bjudge\b",
        r"\bjustice\b",
        r"\blawyer\b",
        r"\battorney\b",
        r"\bbarrister\b",
        r"\blegal practice\b",
        r"\blegal profession\b",
        r"\blegal culture\b",
        r"\blegal system\b",
        r"\blegal reform\b",
        r"\bhuman rights\b",
        r"\bcompliance\b",
        r"\bregulatory\b(?!.*biology)",
        r"\bintellectual property\b",
        r"\bIP law\b",
        r"\bpenal( law| code)?\b",
        r"\btax law\b",
    ],
    "FST": [
        # Original patterns
        r"\bengineering\b",
        r"\bcivil( and environmental)? engineering\b",
        r"\belectrical (and (computer|electronics) )?engineering\b",
        r"\belectromechanical engineering\b",
        r"\bcomputer (and )?information science\b",
        r"\bmathematics?\b",
        r"\bphysics\b",
        r"\bchemistry\b",
        r"\bocean science and technology\b",
        r"\biapme\b",
        r"\binstitute of applied physics and materials engineering\b",
        r"\bamsv\b",
        r"\bstate key laboratory of analog and mixed[- ]signal vlsi\b",
        r"\bskl[- ]iotsc\b",
        r"\bstate key laboratory of internet of things for smart city\b",
        r"\bcentre of science and engineering promotion\b",
        # NEW: Disciplines
        r"\bcomputer science\b",
        r"\belectronics?\b",
        r"\bmechanical engineering\b",
        r"\benvironmental engineering\b",
        r"\bmaterials? (science|engineering)\b",
        r"\bapplied physics\b",
        r"\bcoastal studies\b",
        r"\binformation technology\b",
        r"\bIT\b",
        # NEW: Topics and technologies
        r"\bartificial intelligence\b",
        r"\bAI\b(?!.*education)",
        r"\bmachine learning\b",
        r"\bdeep learning\b",
        r"\bneural network\b",
        r"\brobotics?\b",
        r"\bautomation\b",
        r"\binternet of things\b",
        r"\bIoT\b",
        r"\bsmart city\b",
        r"\bVLSI\b",
        r"\bmicroelectronics?\b",
        r"\bsemiconductor\b",
        r"\bchip design\b",
        r"\bcircuit\b",
        r"\bsignal processing\b",
        r"\b5G\b",
        r"\bwireless\b(?!.*optical)",
        r"\bnetwork(ing)?\b(?!.*business)",
        r"\bcybersecurity\b",
        r"\bdata science\b",
        r"\bbig data\b",
        r"\balgorithm\b",
        r"\bcoding\b",
        r"\bprogramming\b",
        r"\bsoftware\b",
        r"\bhardware\b",
        r"\bstructural (engineering|design)\b",
        r"\bconstruction\b",
        r"\binfrastructure\b",
        r"\benvironmental science\b",
        r"\bsustainabilit(y|able)\b",
        r"\brenewable energy\b",
        r"\bclimate\b(?!.*change.*social)",
        r"\bquantum\b",
        r"\bnanotechnology\b",
        r"\bbiomaterials?\b",
        r"\bcomposite materials?\b",
    ],
    "FSS": [
        # Original patterns
        r"\bsocial sciences?\b",
        r"\bsociolog(y|ies)\b",
        r"\bpsychology\b",
        r"\bcommunication\b",
        r"\bjournalism\b",
        r"\bpublic communication\b",
        r"\beconomics?\b",
        r"\bgovernment\b",
        r"\bpublic administration\b",
        r"\bigpa\b",
        r"\binstitute of global and public affairs\b",
        r"\brussian centre\b",
        r"\b(contemporary )?china\b",
        r"\bsocial science research centre on contemporary china\b",
        # NEW: Disciplines
        r"\bpolitical science\b",
        r"\bpublic policy\b",
        r"\bpublic affairs\b",
        r"\binternational relations\b",
        r"\bmacro[-]?economics\b",
        r"\bmicro[-]?economics\b",
        r"\bapplied economics\b",
        r"\bdevelopmental psychology\b",
        r"\bcognitive( science| psychology)?\b(?!.*brain)",
        r"\bclinical psychology\b",
        r"\beducational psychology\b",
        r"\bsocial psychology\b",
        r"\bmedia( studies)?\b",
        r"\bmass communication\b",
        r"\bdigital media\b",
        # NEW: Topics
        r"\bpolicy\b",
        r"\bgovernance\b",
        r"\bdemocracy\b",
        r"\bpublic service\b",
        r"\bpolitics\b",
        r"\bpolitical\b",
        r"\bdiplomatic\b",
        r"\bdiplomat\b",
        r"\bsociety\b",
        r"\bsocial welfare\b",
        r"\bcommunity\b",
        r"\burban\b",
        r"\brural\b",
        r"\bmigration\b",
        r"\bpopulation\b",
        r"\bdemographic\b",
        r"\binequality\b",
        r"\bpoverty\b",
        r"\bsocial justice\b",
        r"\bgender\b",
        r"\bfeminis(m|t)\b",
        r"\bmental health\b",
        r"\bbehavior(al)?\b",
        r"\bcriminolog(y|ical)\b",
        r"\bdeviance\b",
        r"\bmedia industry\b",
        r"\bpublic opinion\b",
        r"\bsurvey\b",
        r"\bmarket research\b(?!.*business)",
        r"\bsocial media\b",
        r"\bnews\b",
        r"\bbroadcast\b",
        r"\bpublic relations\b",
        r"\bPR\b",
        r"\badvertising\b(?!.*business)",
        r"\beconomic development\b",
        r"\beconomic growth\b",
        r"\btrade\b",
        r"\blabor( market)?\b",
        r"\bemployment\b",
        r"\bgreater bay area\b",
        r"\bGBA\b",
        r"\bmacau studies\b",
        r"\bchina studies\b",
    ],
}

# Compile regex patterns once (case-insensitive)
COMPILED = {
    fac: [re.compile(pat, flags=re.IGNORECASE) for pat in patterns]
    for fac, patterns in FACULTY_SIGNALS.items()
}

def detect_faculties(text: str) -> list:
    """
    Return a sorted list of faculty codes detected from free text using signals only.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    found = set()
    for fac, regexes in COMPILED.items():
        if any(r.search(text) for r in regexes):
            found.add(fac)
    return sorted(found)

# Load events_en_US.csv
filename = 'data/events_en_US.csv'

if not os.path.exists(filename):
    print(f"❌ File '{filename}' not found!")
    print(f"Available CSV files: {[f for f in os.listdir('.') if f.endswith('.csv')]}")
else:
    # Read the CSV
    ev_df = pd.read_csv(filename, encoding='utf-8')
    print(f"✓ Loaded {len(ev_df)} events from {filename}")
    print(f"✓ Columns: {list(ev_df.columns)}\n")

    # Find organizedBy column (case-insensitive)
    organized_by_col = None
    for col in ev_df.columns:
        if 'organizedby' in col.lower():
            organized_by_col = col
            break

    if organized_by_col is None:
        print("❌ 'organizedBy' column not found!")
    else:
        print(f"✓ Found column: '{organized_by_col}'\n")

        # Apply faculty detection to organizedBy column
        print("Processing organizedBy column...")
        ev_df['faculties_list'] = ev_df[organized_by_col].apply(detect_faculties)

        # Convert list to comma-separated string
        ev_df['faculties'] = ev_df['faculties_list'].apply(lambda x: ', '.join(x) if x else '')

        # Show statistics
        print(f"\n{'='*80}")
        print("FACULTY DETECTION RESULTS (EXPANDED SIGNALS)")
        print(f"{'='*80}")

        # Count how many events have each number of faculties
        faculty_counts = ev_df['faculties_list'].apply(len)
        print(f"\nEvents by number of faculties assigned:")
        print(faculty_counts.value_counts().sort_index())

        # Count by individual faculty
        print(f"\n\nEvents per faculty:")
        all_faculties = {}
        for fac_list in ev_df['faculties_list']:
            for fac in fac_list:
                all_faculties[fac] = all_faculties.get(fac, 0) + 1

        if all_faculties:
            for fac in sorted(all_faculties.keys()):
                print(f"  {fac}: {all_faculties[fac]} events")
        else:
            print("  (No faculties detected)")

        # Calculate coverage
        total_rows = len(ev_df)
        rows_with_faculties = len(ev_df[ev_df['faculties'] != ''])
        rows_without_faculties = len(ev_df[ev_df['faculties'] == ''])

        print(f"\n\nCoverage:")
        print(f"  Total events: {total_rows}")
        print(f"  Events with faculties: {rows_with_faculties} ({100*rows_with_faculties/total_rows:.1f}%)")
        print(f"  Events without faculties (Other): {rows_without_faculties} ({100*rows_without_faculties/total_rows:.1f}%)")

        # Show sample results
        print(f"\n\nSample results (first 10 rows with detected faculties):")
        print(f"{'='*80}")
        sample = ev_df[ev_df['faculties'] != ''][[organized_by_col, 'faculties']].head(10)
        if len(sample) > 0:
            for idx, row in sample.iterrows():
                print(f"\nOriginal: {row[organized_by_col]}")
                print(f"Faculties: {row['faculties']}")
        else:
            print("⚠ No faculties detected. Showing sample organizedBy values:")
            print(ev_df[organized_by_col].dropna().unique()[:20])

        # Save cleaned data
        output_filename = 'data/events_en_US_cleaned.csv'
        ev_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n{'='*80}")
        print(f"✅ Cleaned data saved to '{output_filename}'")
        print(f"\n📊 Summary:")
        print(f"   Pattern expansion: Original → Expanded")
        print(f"   Total rows: {total_rows}")
        print(f"   Classified events: {rows_with_faculties} ({100*rows_with_faculties/total_rows:.1f}%)")
        print(f"   Unclassified (Other): {rows_without_faculties} ({100*rows_without_faculties/total_rows:.1f}%)")