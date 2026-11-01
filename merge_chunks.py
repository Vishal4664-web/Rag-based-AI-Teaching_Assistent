
import os
import json
import math

n = 5

os.makedirs("newjsons", exist_ok=True)

for filename in os.listdir("jsons"):

    if not filename.endswith(".json"):
        continue

    file_path = os.path.join("jsons", filename)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both JSON structures
        if isinstance(data, dict):
            chunks = data["chunks"]
            full_text = data.get("text", "")
        elif isinstance(data, list):
            chunks = data
            full_text = " ".join(c["text"] for c in chunks)
        else:
            print(f"Skipping {filename}: Invalid JSON structure")
            continue

        new_chunks = []

        num_chunks = len(chunks)
        num_groups = math.ceil(num_chunks / n)

        for i in range(num_groups):

            start_idx = i * n
            end_idx = min((i + 1) * n, num_chunks)

            chunk_group = chunks[start_idx:end_idx]

            new_chunks.append({
                "number": chunk_group[0]["number"],
                "title": chunk_group[0]["title"],
                "start": chunk_group[0]["start"],
                "end": chunk_group[-1]["end"],
                "text": " ".join(c["text"] for c in chunk_group)
            })

        output_data = {
            "chunks": new_chunks,
            "text": full_text
        }

        output_path = os.path.join("newjsons", filename)

        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(output_data, json_file, indent=4, ensure_ascii=False)

    #     print(f"Processed: {filename}")
    #     print(f"Original chunks: {num_chunks}")
    #     print(f"Merged chunks: {len(new_chunks)}")

    except Exception as e:
        print(f"Error in {filename}: {e}")
