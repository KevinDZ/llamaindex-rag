from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "data")
STORAGE_DIR = os.path.join(_BASE_DIR, "storage")

def build_index():
    data_dir = Path(DATA_DIR)
    documents = []

    for file_path in data_dir.iterdir():
        if file_path.suffix.lower() in {".html", ".htm", ".md", ".txt"}:
            doc = FlatReader().load_data(file_path)
            documents.extend(doc)

    parser = SimpleFileNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print(f"索引已保存到 {STORAGE_DIR}")
    return index