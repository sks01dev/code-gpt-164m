import sys
import os
import gc

# Resolve module paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from transformers import GPT2Tokenizer
from src.model import CodeGPT

st.set_page_config(page_title="CodeGPT 164M Autocompleter", page_icon="⚡", layout="centered")
st.title("⚡ CodeGPT: 164M Parameter Causal Language Model")
st.caption("Custom Decoder-Only Causal Transformer engineered from scratch in PyTorch")

HF_REPO_ID = "sks01dev/code-gpt-164m"
MODEL_FILENAME = "code_gpt.pt"

@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize architecture
    model = CodeGPT(vocab_size=tokenizer.vocab_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with st.spinner("Downloading and memory-mapping model weights..."):
        # Download weight file path from Hugging Face Hub
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
        
        # Load state_dict using memory-mapping (mmap) to avoid RAM spike crash
        state_dict = torch.load(model_path, map_location=torch.device(device), mmap=True)
        model.load_state_dict(state_dict)
        
        # Free memory immediately
        del state_dict
        gc.collect()
        
        st.success("Loaded model checkpoint successfully!")
    
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model()

instruction = st.text_input("Enter Python Task Instruction:", "Write a function to check if a number is prime.")

if st.button("Generate Python Code"):
    prompt = f"# Instruction: {instruction}\n"
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    with st.spinner("Executing KV-cached autoregressive loop..."):
        output_ids = model.generate_kv(input_ids, max_new_tokens=120)[0]
        raw_text = tokenizer.decode(output_ids.tolist())
        
        # Clean output boundary
        clean_text = raw_text.split("<|endoftext|>")[0]
        if prompt in clean_text:
            clean_text = clean_text.replace(prompt, "").strip()
            
    st.markdown("### Generated Code Output:")
    st.code(clean_text if clean_text else raw_text, language="python")
