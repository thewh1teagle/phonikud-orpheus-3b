import argparse
from unsloth import FastLanguageModel
import torch
from IPython.display import display, Audio
from snac import SNAC

def redistribute_codes(code_list, snac_model):
    """Redistribute codes for audio generation"""
    layer_1 = []
    layer_2 = []
    layer_3 = []
    for i in range((len(code_list)+1)//7):
        layer_1.append(code_list[7*i])
        layer_2.append(code_list[7*i+1]-4096)
        layer_3.append(code_list[7*i+2]-(2*4096))
        layer_3.append(code_list[7*i+3]-(3*4096))
        layer_2.append(code_list[7*i+4]-(4*4096))
        layer_3.append(code_list[7*i+5]-(5*4096))
        layer_3.append(code_list[7*i+6]-(6*4096))
    codes = [torch.tensor(layer_1).unsqueeze(0),
             torch.tensor(layer_2).unsqueeze(0),
             torch.tensor(layer_3).unsqueeze(0)]
    
    # codes = [c.to("cuda") for c in codes]
    audio_hat = snac_model.decode(codes)
    return audio_hat


def run_inference(model_path, prompts, chosen_voice=None):
    """Run inference using the trained model
    
    Args:
        model_path: Path to the saved model directory
        prompts: List of text prompts to generate audio from
        chosen_voice: Voice identifier (None for single-speaker)
    """
    
    # Load model and tokenizer from the specified path
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    # Load SNAC model for audio decoding
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    snac_model = snac_model.to("cpu")
    
    # Prepare prompts with voice prefix if needed
    prompts_ = [(f"{chosen_voice}: " + p) if chosen_voice else p for p in prompts]
    
    # Tokenize all prompts
    all_input_ids = []
    for prompt in prompts_:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        all_input_ids.append(input_ids)
    
    # Add special tokens
    start_token = torch.tensor([[128259]], dtype=torch.int64)  # Start of human
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # End of text, End of human
    
    # Modify input_ids with special tokens
    all_modified_input_ids = []
    for input_ids in all_input_ids:
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)  # SOH SOT Text EOT EOH
        all_modified_input_ids.append(modified_input_ids)
    
    # Pad tensors and create attention masks
    all_padded_tensors = []
    all_attention_masks = []
    max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
    for modified_input_ids in all_modified_input_ids:
        padding = max_length - modified_input_ids.shape[1]
        padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
        attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
        all_padded_tensors.append(padded_tensor)
        all_attention_masks.append(attention_mask)
    
    all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
    all_attention_masks = torch.cat(all_attention_masks, dim=0)
    
    # Move to CUDA and generate
    input_ids = all_padded_tensors.to("cuda")
    attention_mask = all_attention_masks.to("cuda")
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1200,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1,
        num_return_sequences=1,
        eos_token_id=128258,
        use_cache=True
    )
    
    # Post-process generated tokens
    token_to_find = 128257
    token_to_remove = 128258
    
    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)
    
    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
    else:
        cropped_tensor = generated_ids
    
    # Remove unwanted tokens
    processed_rows = []
    for row in cropped_tensor:
        masked_row = row[row != token_to_remove]
        processed_rows.append(masked_row)
    
    # Process codes
    code_lists = []
    for row in processed_rows:
        row_length = row.size(0)
        new_length = (row_length // 7) * 7
        trimmed_row = row[:new_length]
        trimmed_row = [t - 128266 for t in trimmed_row]
        code_lists.append(trimmed_row)
    
    # Generate audio samples
    my_samples = []
    for code_list in code_lists:
        samples = redistribute_codes(code_list, snac_model)
        my_samples.append(samples)
    
    # Display results
    if len(prompts) != len(my_samples):
        raise Exception("Number of prompts and samples do not match")
    else:
        for i in range(len(my_samples)):
            print(prompts[i])
            samples = my_samples[i]
            display(Audio(samples.detach().squeeze().to("cpu").numpy(), rate=24000))
    
    # Clean up to save RAM
    del my_samples, samples
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Orpheus model")
    parser.add_argument("model_path", type=str, help="Path to the trained model directory (e.g., lora_model)")
    parser.add_argument("--prompt", type=str, action="append", help="Text prompt(s) to generate audio from (can be used multiple times)")
    parser.add_argument("--voice", type=str, default=None, help="Voice identifier (None for single-speaker)")
    
    args = parser.parse_args()
    
    # Default prompts if none provided
    if args.prompt is None or len(args.prompt) == 0:
        prompts = [
            "Hey there my name is Elise, <giggles> and I'm a speech generation model that can sound like a person.",
        ]
    else:
        prompts = args.prompt
    
    chosen_voice = args.voice
    
    # Run inference
    run_inference(args.model_path, prompts, chosen_voice)

