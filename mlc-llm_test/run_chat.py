from mlc_chat import ChatModule, ChatConfig
from mlc_chat.callback import StreamToStdout

cm = ChatModule(model="Llama-2-13b-chat-hf-q4f16_1-multi_gpu", chat_config=ChatConfig(num_shards=2))
cm.generate(prompt="What is the meaning of life?", progress_callback=StreamToStdout(callback_interval=2))
