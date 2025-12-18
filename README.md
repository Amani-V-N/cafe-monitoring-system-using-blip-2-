# cafe-monitoring-system-using-blip-2-
!pip install -U transformers accelerate pillow
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-flan-t5-xl"
)

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    device_map="auto",
    torch_dtype=torch.float16
)

model.eval()
prompt = """
You are a cafe safety monitoring system.
Analyze the image and report:

1. Is there any spill or wet floor?
2. Is there fire, smoke, or dangerous heat?
3. Are customers waiting or crowding?
4. Overall danger level: Low / Medium / High
5. If danger exists, generate a short alert message.
"""
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    device_map="auto",
    torch_dtype=torch.float16
)
image = Image.open("/content/happy.jpg").convert("RGB")

inputs = processor(image, return_tensors="pt")  # ❌ no .to("cuda")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)

caption = processor.decode(output[0], skip_special_tokens=True)
print("Image Caption:", caption)
text = caption.lower()

spill_keywords = ["wet", "spill", "puddle", "water", "liquid"]
fire_keywords = ["fire", "smoke", "flame"]
waiting_keywords = ["people", "customers", "line", "queue", "standing"]

spill = any(word in text for word in spill_keywords)
fire = any(word in text for word in fire_keywords)
waiting = any(word in text for word in waiting_keywords)

if fire:
    danger = "High"
elif spill:
    danger = "Medium"
else:
    danger = "Low"

print("Spill:", spill)
print("Fire:", fire)
print("Customers waiting:", waiting)
print("Danger level:", danger)
