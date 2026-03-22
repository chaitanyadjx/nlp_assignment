import json

def patch_notebook(path):
    with open(path, "r") as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            
            # 1. Fix ASTEDataset __getitem__ padding bug logic
            if "class ASTEDataset(Dataset):" in src and "return_tensors=\"pt\"" in src:
                src = src.replace("padding=\"max_length\",", "")
                src = src.replace("return_tensors=\"pt\",", "")
                
                # Replace the squeezing and labels logic
                old_logic = """        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels":         labels,
        }"""
                new_logic = """        labels = [l if l != self.tokenizer.pad_token_id else -100 for l in target_enc["input_ids"]]

        return {
            "input_ids":      input_enc["input_ids"],
            "attention_mask": input_enc["attention_mask"],
            "labels":         labels,
        }"""
                src = src.replace(old_logic, new_logic)
                
            # 2. Add compute_metrics support so predict_with_generate=True doesn't cause NaN loss
            if "predict_with_generate=True," in src:
                src = src.replace("predict_with_generate=True,", "predict_with_generate=False, # disabled to prevent NaN val loss without compute_metrics func")
            
            # 3. Reduce batch size to prevent OOM
            if "BATCH_SIZE       = 8" in src:
                src = src.replace("BATCH_SIZE       = 8", "BATCH_SIZE       = 4")
                
            # 4. Fix kaggle paths if any exist
            if "/kaggle/working" in src or "/kaggle/input" in src:
                src = src.replace("/kaggle/input/datasets/chaitanyajx1/datasetnlp", "nlp_assignment")
                src = src.replace("/kaggle/working/aste_t5_model", "aste_t5_model")
                src = src.replace("/kaggle/working/predictions.jsonl", "predictions.jsonl")
                src = src.replace("/kaggle/working/", "")
                
            cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in src.split('\n')][:-1]
            
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)

patch_notebook("/home/chaitanya/123ad0045/nlp_assignment/ASTE_Model.ipynb")
