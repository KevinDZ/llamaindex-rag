import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path

# ============ 配置区 ============

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "data")
STORAGE_DIR = os.path.join(_BASE_DIR, "storage")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 3

# 方式一：在代码里设置 API Key
os.environ["DASHSCOPE_API_KEY"] = "sk-8d4731858dfc469dbe6863b4eafa60f5"

# 方式二：在终端设置，更推荐
# export DASHSCOPE_API_KEY="sk-xxx"

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key or api_key.startswith("sk-xxx"):
    raise RuntimeError("请先在百炼控制台创建 API Key，并替换代码中的 sk-xxx")

# ============ 全局设置 ============

Settings.llm = DashScope(
    model_name="qwen3.8-27b",
    api_key=api_key,
)

Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v2",
    api_key=api_key,
)

Settings.chunk_size = CHUNK_SIZE
Settings.chunk_overlap = CHUNK_OVERLAP

# ============ Prompt 模块 ============

# QA_PROMPT_TMPL = (
#     "你是一名专业客服助手，只能根据下方提供的上下文信息回答问题。\n"
#     "如果上下文不足以回答问题，请回答：'知识库中未找到相关信息'。\n"
#     "回答要简洁、准确，不要编造内容。\n"
#     "\n"
#     "上下文信息：\n"
#     "---------------------\n"
#     "{context_str}\n"
#     "---------------------\n"
#     "\n"
#     "问题：{query_str}\n"
#     "回答："
# )

QA_PROMPT_TMPL = (
    "你是一名专业、礼貌的售后客服助手。请仅根据下方提供的上下文信息回答问题，"
    "不要使用外部知识，不要编造产品型号、参数、价格、政策或维修流程。\n"
    "如果上下文不足以回答问题，请回答：'知识库中未找到相关信息，建议联系售后进一步确认'。\n"
    "\n"
    "回答请按以下结构输出：\n"
    "1. 故障原因\n"
    "2. 处理建议\n"
    "3. 是否需要报修\n"
    "\n"
    "上下文信息：\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "问题：{query_str}\n"
    "回答："
)

qa_prompt_template = PromptTemplate(QA_PROMPT_TMPL)

# ============ 构建索引 ============

def build_index():
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    print(f"已加载 {len(documents)} 个文档")

    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print(f"索引已保存到 {STORAGE_DIR}")
    return index

def file_build_index():
    data_dir = Path(DATA_DIR)
    documents = []

    for file_path in data_dir.iterdir():
        if file_path.suffix.lower() in {".html", ".htm", ".md", ".txt"}:
            doc = FlatReader().load_data(file_path)
            documents.extend(doc)

    parser = SimpleFileNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    # print(type(nodes))
    # print(len(nodes))
    # print(type(nodes[0]) if len(nodes) > 0 else "empty")

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print(f"索引已保存到 {STORAGE_DIR}")
    return index

def load_index():
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context)
    return index


# ============ 检索问答 ============

def create_query_engine(index):
    return index.as_query_engine(
        similarity_top_k=TOP_K,
        response_mode="compact",
        text_qa_template=qa_prompt_template,
    )


def chat(query_engine):
    print("\n" + "=" * 50)
    print("知识库问答助手（输入 q 退出）")
    print("=" * 50)

    while True:
        question = input("\n请输入问题：").strip()
        if not question or question.lower() == "q":
            print("再见！")
            break

        response = query_engine.query(question)
        print(f"\n回答：{response}")

        if hasattr(response, "source_nodes"):
            print(f"\n参考了 {len(response.source_nodes)} 个片段：")
            for i, node in enumerate(response.source_nodes, 1):
                print(f"  [{i}] {node.node.text[:120]}...")


# ============ 主入口 ============

def main():
    docstore_path = os.path.join(STORAGE_DIR, "docstore.json")
    if os.path.exists(docstore_path):
        print("从磁盘加载索引...")
        index = load_index()
    else:
        print("构建新索引...")
        index = file_build_index()
        # index = build_index()

    query_engine = create_query_engine(index)
    chat(query_engine)


if __name__ == "__main__":
    main()