import json

# Patch notebook
def patch_notebook(path):
    with open(path, "r") as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            
            src = src.replace("per_device_train_batch_size=8", "per_device_train_batch_size=4")
            src = src.replace("per_device_train_batch_size=32", "per_device_train_batch_size=4")
            
            src = src.replace("per_device_eval_batch_size=16", "per_device_eval_batch_size=4")
            src = src.replace("per_device_eval_batch_size=64", "per_device_eval_batch_size=4")
            
            if "gradient_accumulation_steps=4" in src:
                src = src.replace("gradient_accumulation_steps=4", "gradient_accumulation_steps=8")
            elif "warmup_steps=50," in src:
                src = src.replace("warmup_steps=50,", "warmup_steps=50,\n    gradient_accumulation_steps=8,")
            
            cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in src.split('\n')][:-1]
            if len(cell['source']) == 0 and len(src) > 0:
                cell['source'] = [src]
            
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)

try:
    patch_notebook("/home/chaitanya/123ad0045/nlp_assignment/ASTE_Model_v4.ipynb")
except Exception as e:
    print(e)
