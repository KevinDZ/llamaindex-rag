# from llama_index.llms.dashscope import DashScope
# from llama_index.embeddings.dashscope import DashScopeEmbedding
# from llama_index.core import Settings
# import os
# import dashscope
# dashscope.base_http_api_url = "https://ws-nualv587oalotk9g.cn-beijing.maas.aliyuncs.com/api/v1"

# os.environ["DASHSCOPE_API_KEY"] = "sk-8d4731858dfc469dbe6863b4eafa60f5"

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"},
#             {"text": "图中描绘的是什么景象?"}]
#     }]
# response = dashscope.MultiModalConversation.call(
#     api_key=os.getenv('DASHSCOPE_API_KEY'),
#     model='qwen3.8-max',
#     messages=messages
# )
# print(response.output.choices[0].message.content[0]["text"])

import os
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core import Settings

os.environ["DASHSCOPE_API_KEY"] = "sk-8d4731858dfc469dbe6863b4eafa60f5s"

Settings.llm = DashScope(
    model_name="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v2",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)