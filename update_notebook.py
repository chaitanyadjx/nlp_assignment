import json

path = "ASTE_Model_v3.ipynb"
with open(path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        for i, line in enumerate(cell["source"]):
            if '"/kaggle/input/datasets/chaitanyajx1/datasetnlp"' in line:
                cell["source"][i] = line.replace('"/kaggle/input/datasets/chaitanyajx1/datasetnlp"', '"./nlp_assignment"')
            if '"/kaggle/working/aste_flant5_model"' in line:
                cell["source"][i] = line.replace('"/kaggle/working/aste_flant5_model"', '"./aste_flant5_model"')
            if '"/kaggle/working/predictions.jsonl"' in line:
                cell["source"][i] = line.replace('"/kaggle/working/predictions.jsonl"', '"./predictions.jsonl"')
            if '"/kaggle/input/datasets/chaitanyajx1/datasetnlp/test.jsonl"' in line:
                cell["source"][i] = line.replace('"/kaggle/input/datasets/chaitanyajx1/datasetnlp/test.jsonl"', '"./nlp_assignment/test.jsonl"')
            if "Paths (Kaggle)" in line:
                cell["source"][i] = line.replace("Paths (Kaggle)", "Paths (Local)")

with open(path, "w") as f:
    json.dump(nb, f, indent=1)

print("Updated paths in ASTE_Model_v3.ipynb")
