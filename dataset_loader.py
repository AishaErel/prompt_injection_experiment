import os
from typing import List, Dict


def load_documents(data_path="data") -> List[Dict]:

    documents = []

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} folder not found")

    for filename in os.listdir(data_path):

        if filename.endswith(".txt"):

            with open(
                os.path.join(data_path, filename),
                "r",
                encoding="utf-8"
            ) as f:
                text = f.read()

            label = (
                "malicious"
                if "malicious" in filename.lower()
                else "benign"
            )

            documents.append({
                "text": text,
                "label": label,
                "source": filename,
            })

    print(f"Loaded {len(documents)} documents")

    return documents