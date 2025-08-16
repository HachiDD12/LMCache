# Standard
from dataclasses import asdict
import argparse
import contextlib
import os
import time

# Third Party
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def setup_environment_variables(
    use_disk: bool = False, blend_special_str: str = " # # "
):
    # LMCache-related environment variables

    # LMCache is set to use 256 tokens per chunk
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    # Blending related config
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"

    if use_disk:
        # Disable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "False"

        # Set the maximum size of the local CPU buffer size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

        # Enable local disk backend in LMCache
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"

        # Set the maximum size of the local disk size to 10GB
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
    else:
        # Enable local CPU backend in LMCache
        os.environ["LMCACHE_LOCAL_CPU"] = "True"

        # Set the maximum size of the local CPU size to 5GB
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"


@contextlib.contextmanager
def build_llm_with_lmcache(lmcache_connector: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=lmcache_connector,
        kv_role="kv_both",
    )

    # for mistral 7b instruct
    # llm_args = EngineArgs(
    #     model=model,
    #     tokenizer_mode="mistral",
    #     kv_transfer_config=ktc,
    #     max_model_len=8000,
    #     gpu_memory_utilization=0.8,
    #     enable_prefix_caching=False,
    # )
    
    # for devstral small
    llm_args = EngineArgs(
        model=model,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        tensor_parallel_size=2, # 4 GPUs, change based on resource availability
        kv_transfer_config=ktc,
        max_model_len=64000,
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        # Clean up lmcache backend
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def print_output(
    llm: LLM,
    prompt: list[int],
    sampling_params: SamplingParams,
    req_str: str,
):
    start = time.time()
    outputs = llm.generate(prompt_token_ids=prompt, sampling_params=sampling_params)
    end = time.time()
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {end - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--use-disk",
        action="store_true",
        help="Specify whether to use disk as backend (default: False)",
    )

    parser.add_argument(
        "-b",
        "--blend-special-str",
        default="# #",
        help="Specify the special separators to separate chunks (default: '# #')",
    )

    return parser.parse_args()

def test_mistral7b_tokenizer(tokenizer, blend_special_str):
    print(f"### Testing mistral7b tokenizer ###")
    chunk1_prompt = tokenizer.encode("Hello, how are you?")
    chunk2_prompt = tokenizer.encode("Hello, what's up?")
    blend_special_str_toks = tokenizer.encode(blend_special_str)
    combined_prompt = tokenizer.encode(f"Hello, how are you?{blend_special_str}Hello, what's up?")
    if chunk1_prompt + blend_special_str_toks + chunk2_prompt == combined_prompt:
        print("chunk1_prompt + blend_special_str_toks + chunk2_prompt == combined_prompt")
        print(f"combined_prompt: {combined_prompt}")
        print(f"blend_special_str: {blend_special_str} -> {blend_special_str_toks}")
    else:
        print("chunk1_prompt + blend_special_str_toks + chunk2_prompt != combined_prompt")
        print(f"LHS: {chunk1_prompt + blend_special_str_toks + chunk2_prompt}")
        print(f"RHS: {combined_prompt}")
        print(f"blend_special_str: {blend_special_str} -> {blend_special_str_toks}")
    print("-" * 50)

def main():
    args = parse_args()

    lmcache_connector = "LMCacheConnectorV1"
    model = "mistralai/Devstral-Small-2507"
    model2 = "mistralai/Mistral-7B-Instruct-v0.2" # only for testing tokenizer

    setup_environment_variables(args.use_disk, args.blend_special_str)
    blend_special_str = os.getenv("LMCACHE_BLEND_SPECIAL_STR")

    mistral_tokenizer = MistralTokenizer.from_hf_hub(model)
    tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer
    # test_mistral7b_tokenizer(AutoTokenizer.from_pretrained(model2), blend_special_str)

    with build_llm_with_lmcache(lmcache_connector, model) as llm:
        # This example script runs two requests with a shared prefix.
        # Define the shared prompt and specific prompts
        sys_prompt = tokenizer.encode("You are a very helpful assistant.", bos=True, eos=False)
        chunk1_prompt = tokenizer.encode("Hello, how are you?" * 500, bos=False, eos=False)
        chunk2_prompt = tokenizer.encode("Hello, what's up?" * 500, bos=False, eos=False)
        blend_special_str_toks = tokenizer.encode(blend_special_str, bos=False, eos=False)
        
        # print(f"### Testing devstral tokenizer ###")
        # combined_prompt = tokenizer.encode(f"Hello, how are you?{blend_special_str}Hello, what's up?", bos=False, eos=False)
        # if chunk1_prompt + blend_special_str_toks + chunk2_prompt == combined_prompt:
        #     print("chunk1_prompt + blend_special_str_toks + chunk2_prompt == combined_prompt")
        #     print(f"combined_prompt: {combined_prompt}")
        #     print(f"blend_special_str: {blend_special_str} -> {blend_special_str_toks}")
        # else:
        #     print("chunk1_prompt + blend_special_str_toks + chunk2_prompt != combined_prompt")
        #     print(f"LHS: {chunk1_prompt + blend_special_str_toks + chunk2_prompt}")
        #     print(f"RHS: {combined_prompt}")
        #     print(f"blend_special_str: {blend_special_str} -> {blend_special_str_toks}")
        # print(f"blendspecialstr with bos and eos: {tokenizer.encode(blend_special_str, bos=True, eos=True)}")
        # print("-" * 50)
        
        print(f"### Prompting devstral ###")
        first_prompt = (
            sys_prompt
            + blend_special_str_toks
            + chunk1_prompt
            + blend_special_str_toks
            + chunk2_prompt
            + blend_special_str_toks
            + tokenizer.encode("Hello, my name is", bos=False, eos=True)
        )

        second_prompt = (
            sys_prompt
            + blend_special_str_toks
            + chunk2_prompt
            + blend_special_str_toks
            + chunk1_prompt
            + blend_special_str_toks
            + tokenizer.encode("Hello, how are you?", bos=False, eos=True)
        )
        third_prompt = (
            sys_prompt
            + blend_special_str_toks
            + chunk1_prompt
            + blend_special_str_toks
            + chunk1_prompt
            + blend_special_str_toks
            + chunk1_prompt
            + blend_special_str_toks
            + chunk2_prompt
            + blend_special_str_toks
            + tokenizer.encode("Hello, my name is", bos=False, eos=True)
        )
        
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

        # Print the first output
        print_output(llm, first_prompt, sampling_params, "first")

        time.sleep(1)

        # print the second output
        print_output(llm, second_prompt, sampling_params, "second")

        time.sleep(1)

        # print the third output
        print_output(llm, first_prompt, sampling_params, "first repeated")


if __name__ == "__main__":
    main()
