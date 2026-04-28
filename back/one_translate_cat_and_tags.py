from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Helsinki-NLP/opus-mt-en-es"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate(content):
  inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True)
  outputs = model.generate(**inputs)

  return tokenizer.batch_decode(outputs, skip_special_tokens=True)
