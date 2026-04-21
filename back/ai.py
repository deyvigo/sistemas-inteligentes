from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import InferenceClient

import os
os.environ["HF_TOKEN"] = "key"

client = InferenceClient(
  api_key=os.environ["HF_TOKEN"],
)

def ask_ai(content):
  result = client.translation(
    content,
    model="Helsinki-NLP/opus-mt-en-es",
  )

  return result
