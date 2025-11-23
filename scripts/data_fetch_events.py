import os
import json
import urllib.request

# --- Config ---
BASE_URL = "https://api.data.um.edu.mo/service/media/events/all"
AUTH_KEY = "3e8ffa69aadf42ceb889fd04b0d9825a"  # <-- your API key
OUT_FILE = "data/events_data_all.json"

HEADERS = {
    "Cache-Control": "no-cache",
    "Authorization": AUTH_KEY,  # UM API expects the subscription key here
}

def http_get(url):
    """Perform a GET request and return parsed JSON."""
    req = urllib.request.Request(url, headers=HEADERS)
    req.get_method = lambda: "GET"
    with urllib.request.urlopen(req) as resp:
        status = resp.getcode()
        if status != 200:
            raise RuntimeError("HTTP {}".format(status))
        data = resp.read().decode('utf-8')
    return json.loads(data)

def fetch_page(pagesize, page_num):
    """Fetch one page of events."""
    # IMPORTANT: Use '&page=' not '&amp;page='
    url = f"{BASE_URL}?pagesize={pagesize}&page={page_num}"
    return http_get(url)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    try:
        # 1) Initial request with '?count' to know pagination & reuse page 1 data
        initial_url = f"{BASE_URL}?count"
        initial_data = http_get(initial_url)

        total_pages = initial_data.get("_total_pages", 1)
        pagesize = initial_data.get("_pagesize", 100)

        print("Total pages: {}".format(total_pages))
        print("Objects per page: {}".format(pagesize))

        # The initial '?count' response often includes the first page in _embedded.
        # We'll reuse it to avoid refetching page 1.
        first_page_embedded = initial_data.get("_embedded", [])

        if os.path.exists(OUT_FILE):
            # 2) File exists: update & stop at the first duplicate
            existing = load_json(OUT_FILE)
            existing_list = existing.get("_embedded", [])

            # Build an index for quick lookup by unique itemId (per UM Events data dictionary)
            idx_by_id = {}
            for i, ev in enumerate(existing_list):
                item_id = ev.get("itemId")
                if item_id is not None:
                    idx_by_id[item_id] = i

            new_added = 0
            updated_and_stopped = False

            # Helper to process a batch (page) in newest-first order
            def process_batch(batch):
                nonlocal new_added, updated_and_stopped
                for ev in batch:
                    item_id = ev.get("itemId")
                    # If missing itemId, just prepend safely (rare)
                    if item_id is None:
                        existing_list.insert(0, ev)
                        new_added += 1
                        continue

                    if item_id in idx_by_id:
                        # Update the existing record in-place with the newest data, then stop
                        existing_list[idx_by_id[item_id]] = ev
                        updated_and_stopped = True
                        print("Encountered duplicate itemId={}, updated record and stopping.".format(item_id))
                        break
                    else:
                        # New event: prepend to keep newest-first ordering
                        existing_list.insert(0, ev)
                        # Track index for potential later lookups
                        idx_by_id[item_id] = 0  # inserted at front
                        # Adjust stored indices (+1) since we inserted at the front
                        for k in list(idx_by_id.keys()):
                            if k != item_id:
                                idx_by_id[k] = idx_by_id[k] + 1
                        new_added += 1

            # Process page 1 (from initial '?count' response)
            print("Processing page 1 from initial '?count' response...")
            process_batch(first_page_embedded)

            # If no duplicate yet, continue with pages 2..N
            page_num = 2
            while (not updated_and_stopped) and (page_num <= total_pages):
                page_data = fetch_page(pagesize, page_num)
                print("Fetching page {}/{}...".format(page_num, total_pages))
                batch = page_data.get("_embedded", [])
                process_batch(batch)
                page_num += 1

            # Save merged
            merged = {"_embedded": existing_list, "_returned": len(existing_list)}
            save_json(merged, OUT_FILE)
            print("Prepended {} new events. Total now: {}. Saved to {}."
                  .format(new_added, len(existing_list), OUT_FILE))

        else:
            # 3) File not present: fetch all pages and save full dataset
            all_events = []
            print("File not found. Building a full dataset from all pages...")

            # Use page 1 from initial '?count' response
            print("Collecting page 1...")
            all_events.extend(first_page_embedded)

            # Collect remaining pages
            for page_num in range(2, total_pages + 1):
                page_data = fetch_page(pagesize, page_num)
                print("Collecting page {}/{}...".format(page_num, total_pages))
                all_events.extend(page_data.get("_embedded", []))

            data = {"_embedded": all_events, "_returned": len(all_events)}
            save_json(data, OUT_FILE)
            print("Data from {} events saved to {}".format(len(all_events), OUT_FILE))

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()