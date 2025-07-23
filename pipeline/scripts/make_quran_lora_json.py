import os, json, sys

results = []
for file in os.listdir(sys.argv[1]):
    if file.endswith(".txt"):
        with open(os.path.join(sys.argv[1], file)) as f:
            content = f.read().strip()
            if len(content.split()) > 50:
                results.append({
                    "instruction": "Summarize and offer Islamic advice based on this:",
                    "input": content,
                    "output": "Remember, Allah is always with those who are patient and sincere."
                })

with open("output/json/akhi_lora.json", "w") as o:
    json.dump(results, o, indent=2)
