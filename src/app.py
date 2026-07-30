import gradio as gr
import torch
from transformers import GPT2Tokenizer
from src.model import CodeGPT

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = CodeGPT(vocab_size=tokenizer.vocab_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    model.load_state_dict(torch.load("code_gpt.pt", map_location=torch.device(device)))
    print("Successfully loaded code_gpt.pt!")
except FileNotFoundError:
    print("Warning: code_gpt.pt not found. Running initialized weights.")

model.to(device)
model.eval()

def generate_code(instruction):
    prompt = f"# Instruction: {instruction}\n"
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        output_ids = model.generate_kv(input_ids, max_new_tokens=120)[0]
        raw_text = tokenizer.decode(output_ids.tolist())
        
        # Clean up output boundary
        clean_text = raw_text.split("<|endoftext|>")[0]
        if prompt in clean_text:
            clean_text = clean_text.replace(prompt, "").strip()
            
    return clean_text if clean_text else raw_text

# Define Gradio UI Interface
demo = gr.Interface(
    fn=generate_code,
    inputs=gr.Textbox(lines=2, placeholder="e.g. Write a function to check if a number is prime.", label="Python Task Instruction"),
    outputs=gr.Code(language="python", label="Generated Code Output"),
    title="⚡ CodeGPT: 164M Parameter Causal Language Model",
    description="A custom decoder-only Causal Transformer engineered from scratch in PyTorch with KV-Cache acceleration."
)

if __name__ == "__main__":
    demo.launch()
