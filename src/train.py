"""
Based on https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb

uv run src/train.py
"""

from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments,Trainer, DataCollatorForSeq2Seq
import argparse
from datasets import Dataset, Audio
import pandas as pd
import locale
import torchaudio.transforms as T
import os
from snac import SNAC
from datasets import load_dataset

locale.getpreferredencoding = lambda: "UTF-8"

def load_model():

    fourbit_models = [
        # 4bit dynamic quants for superior accuracy and low memory use
        "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
        # Qwen3 new models
        "unsloth/Qwen3-4B-unsloth-bnb-4bit",
        "unsloth/Qwen3-8B-unsloth-bnb-4bit",
        # Other very popular models!
        "unsloth/Llama-3.1-8B",
        "unsloth/Llama-3.2-3B",
        "unsloth/Llama-3.3-70B",
        "unsloth/mistral-7b-instruct-v0.3",
        "unsloth/Phi-4",
    ] # More models at https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/orpheus-3b-0.1-ft",
        max_seq_length= 2048, # Choose any for long context!
        dtype = None, # Select None for auto detection
        load_in_4bit = False, # Select True for 4bit which reduces memory usage
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )


    model = FastLanguageModel.get_peft_model(
        model,
        r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 64,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return model, tokenizer


def prepare_dataset():
    # df = pd.read_csv("./saspeech_manual/metadata_phonemes.csv", sep="|", names=["file_id", "text"])
    # df["audio"] = df["file_id"].apply(lambda x: f"./saspeech_manual/wav/{x}.wav")

    # dataset = Dataset.from_dict(df)
    # dataset = dataset.cast_column("audio", Audio(sampling_rate=22050))
    # # Decode the audio (instead of lazy loading)
    # dataset = dataset.map(lambda x: {"audio": x["audio"]})
    dataset = load_dataset("MrDragonFox/Elise", split = "train")

    return dataset


def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize the dataset by converting audio to SNAC codes and text to token IDs.
    
    Args:
        dataset: HuggingFace dataset with audio and text fields
        tokenizer: The tokenizer to use for text encoding
    
    Returns:
        Tokenized dataset with input_ids, labels, and attention_mask
    """
    # Get dataset sample rate
    ds_sample_rate = dataset[0]["audio"]["sampling_rate"]
    
    # Load SNAC model for audio tokenization
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to("cuda")
    
    def tokenise_audio(waveform):
        """Convert audio waveform to SNAC codes"""
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(dtype=torch.float32)
        resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
        waveform = resample_transform(waveform)
        
        waveform = waveform.unsqueeze(0).to("cuda")
        
        # Generate the codes from snac
        with torch.inference_mode():
            codes = snac_model.encode(waveform)
        
        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item()+128266)
            all_codes.append(codes[1][0][2*i].item()+128266+4096)
            all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
            all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
            all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
            all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
            all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))
        
        return all_codes
    
    def add_codes(example):
        """Add SNAC codes to dataset example"""
        # Always initialize codes_list to None
        codes_list = None
        
        try:
            answer_audio = example.get("audio")
            # If there's a valid audio array, tokenise it
            if answer_audio and "array" in answer_audio:
                audio_array = answer_audio["array"]
                codes_list = tokenise_audio(audio_array)
        except Exception as e:
            print(f"Skipping row due to error: {e}")
            # Keep codes_list as None if we fail
        example["codes_list"] = codes_list
        
        return example
    
    # Add codes to all examples
    dataset = dataset.map(add_codes, remove_columns=["audio"])
    
    # Define special tokens
    tokeniser_length = 128256
    start_of_text = 128000
    end_of_text = 128009
    
    start_of_speech = tokeniser_length + 1
    end_of_speech = tokeniser_length + 2
    
    start_of_human = tokeniser_length + 3
    end_of_human = tokeniser_length + 4
    
    start_of_ai = tokeniser_length + 5
    end_of_ai = tokeniser_length + 6
    pad_token = tokeniser_length + 7
    
    audio_tokens_start = tokeniser_length + 10
    
    # Filter out invalid entries
    dataset = dataset.filter(lambda x: x["codes_list"] is not None)
    dataset = dataset.filter(lambda x: len(x["codes_list"]) > 0)
    
    def remove_duplicate_frames(example):
        """Remove duplicate consecutive frames from codes"""
        vals = example["codes_list"]
        if len(vals) % 7 != 0:
            raise ValueError("Input list length must be divisible by 7")
        
        result = vals[:7]
        removed_frames = 0
        
        for i in range(7, len(vals), 7):
            current_first = vals[i]
            previous_first = result[-7]
            
            if current_first != previous_first:
                result.extend(vals[i:i+7])
            else:
                removed_frames += 1
        
        example["codes_list"] = result
        
        return example
    
    dataset = dataset.map(remove_duplicate_frames)
    
    # Print tokenization info
    tok_info = '''*** Tokenization Info:
If you are training a multi-speaker model (e.g., canopylabs/orpheus-3b-0.1-ft),
ensure that the dataset includes a "source" field and format the input accordingly:
- Single-speaker: f"{example['text']}"
- Multi-speaker: f"{example['source']}: {example['text']}"
'''
    
    def create_input_ids(example):
        """Create input_ids with special tokens for training"""
        # Determine whether to include the source field
        text_prompt = f"{example['source']}: {example['text']}" if "source" in example else example["text"]
        
        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(end_of_text)
        
        example["text_tokens"] = text_ids
        input_ids = (
            [start_of_human]
            + example["text_tokens"]
            + [end_of_human]
            + [start_of_ai]
            + [start_of_speech]
            + example["codes_list"]
            + [end_of_speech]
            + [end_of_ai]
        )
        example["input_ids"] = input_ids
        example["labels"] = input_ids
        example["attention_mask"] = [1] * len(input_ids)
        
        return example
    
    dataset = dataset.map(create_input_ids, remove_columns=["text", "codes_list"])
    
    # Keep only necessary columns
    columns_to_keep = ["input_ids", "labels", "attention_mask"]
    columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
    dataset = dataset.remove_columns(columns_to_remove)
    
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--report_to', type=str, default="none")
    parser.add_argument('--max_steps', type=int, default=60)

    args = parser.parse_args()
    
    print('Loading model...')
    model, tokenizer = load_model()

    print('Loading dataset...')
    dataset = prepare_dataset()
    breakpoint()
    
    # Tokenize the dataset
    print('Tokenizing dataset...')
    dataset = tokenize_dataset(dataset, tokenizer)
    

    print('Training...')
    trainer = Trainer(
        model = model,
        train_dataset = dataset,
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = args.max_steps,
            learning_rate = 2e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = args.report_to, # Use this for WandB etc
        ),
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("lora_tts_model")  # Local saving
    tokenizer.save_pretrained("lora_tts_model")