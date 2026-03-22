import json

def patch_notebook(path):
    with open(path, "r") as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            
            if "per_device_train_batch_size=32" in src:
                src = src.replace("per_device_train_batch_size=32", "per_device_train_batch_size=8")
                src = src.replace("per_device_eval_batch_size=64", "per_device_eval_batch_size=16")
                # Add gradient accumulation to keep effective batch size similar
                src = src.replace("warmup_steps=50,", "warmup_steps=50,\n    gradient_accumulation_steps=4,")
            
            cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in src.split('\n')][:-1]
            
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)

patch_notebook("/home/chaitanya/123ad0045/nlp_assignment/ASTE_Model_v4.ipynb")
