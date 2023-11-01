import mlc_llm, mlc_chat, tvm

# python3 -m mlc_llm.build --hf-path togethercomputer/RedPajama-INCITE-Chat-3B-v1 --target cuda --quantization q4f16_1


build_args = mlc_llm.BuildArgs(
    #hf_path="togethercomputer/RedPajama-INCITE-Chat-3B-v1",
    #model="Llama-2-7b-chat-hf",
    model="Llama-2-13b-chat-hf",
    quantization="q4f16_1",
    # cc_path="/usr/bin/aarch64-linux-gnu-g++",
    # target="nvidia/jetson-agx-xavier",
    target="cuda-multiarch",
    # target="cuda",
    #max_seq_len=2048,
    max_seq_len=1024,
    num_shards=2,
    build_model_only=True,
    )
lib_path, model_path, chat_config_path = mlc_llm.build_model(build_args)
