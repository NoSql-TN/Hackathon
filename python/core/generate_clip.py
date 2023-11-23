# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("caseyhahn/gpt2-finetuned-genius-lyrics-updated-data")
model = AutoModelForCausalLM.from_pretrained("caseyhahn/gpt2-finetuned-genius-lyrics-updated-data")

prompt = "I like the beach"
tokenizer.encode(prompt)
model.generate(tokenizer.encode(prompt), do_sample=True, max_length=200, top_k=50)

