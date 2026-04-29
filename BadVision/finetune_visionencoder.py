import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 路径：你自己的 CLIP checkpoint
model_path = "/home/LLaVa/checkpoints/clip-vit-large-patch14-336"

# 加载模型和处理器
model = CLIPModel.from_pretrained(model_path).to(device)
processor = CLIPProcessor.from_pretrained(model_path)

# 冻结 text encoder
for p in model.text_model.parameters():
    p.requires_grad = False
for p in model.text_projection.parameters():
    p.requires_grad = False
# 只训练 vision encoder + projection
for p in model.vision_model.parameters():
    p.requires_grad = True
for p in model.visual_projection.parameters():
    p.requires_grad = False

# One-shot 数据
image_path = "IEEE-CVPR.png"   # 你的一张图
text_label = "cvpr"  # 对应的文本标签

image = Image.open(image_path)
inputs = processor(text=[text_label], images=[image], return_tensors="pt", padding=True).to(device)
image_inputs = processor(images=image, return_tensors="pt").to(device)
text_inputs = processor(text=[text_label], return_tensors="pt").to(device)

# 优化器
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()




# 训练循环（演示 20 步）
for step in range(100):
    optimizer.zero_grad()

    image_features = model.get_image_features(pixel_values=image_inputs["pixel_values"])
    text_features = model.get_text_features(input_ids=text_inputs["input_ids"])

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    similarity = F.cosine_similarity(image_features, text_features)
    loss = - similarity.mean()

    loss.backward()
    optimizer.step()

    if step % 5 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# 测试相似度
# with torch.no_grad():
#     outputs = model(**inputs)
#     sim = outputs.logits_per_image.softmax(dim=-1).item()
#     print("Similarity after fine-tuning:", sim)
with torch.no_grad():
    img_feat = model.get_image_features(pixel_values=image_inputs["pixel_values"])
    txt_feat = model.get_text_features(input_ids=text_inputs["input_ids"])
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    sim = (image_features @ text_features.t()).item()
    print("Similarity after fine-tuning:", sim)



# 微调完成后保存模型
save_path = "/home/LLaVa/checkpoints/clip-vit-large-patch14-336-finetuned"

model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print(f"Model and processor have been saved to {save_path}")

