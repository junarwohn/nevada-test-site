import mlc_llm, mlc_chat, tvm

# python3 -m mlc_llm.build --hf-path togethercomputer/RedPajama-INCITE-Chat-3B-v1 --target cuda --quantization q4f16_1


build_args = mlc_llm.BuildArgs(
            hf_path="togethercomputer/RedPajama-INCITE-Chat-3B-v1",
            quantization="q4f16_1",
            target="cuda")
    # max_seq_len=2048)
         
lib_path, model_path, chat_config_path = mlc_llm.build_model(build_args)
