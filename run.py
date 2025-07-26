import os
import cv2
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(model_path="models/face_embeddings.pkl"):
    with open(model_path, "rb") as f:
        return pickle.load(f)  # (embeddings, names)

def recognize_image(img_path, app, known_embs, known_names, threshold=0.4):
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return

    faces = app.get(img)
    if not faces:
        print(f"未检测到人脸: {img_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("simhei.ttf", 20)
    except:
        font = ImageFont.load_default()

    for face in faces:
        emb = face.embedding
        sims = cosine_similarity([emb], known_embs)[0]
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        name = known_names[best_idx] if best_score >= threshold else "未知"

        box = face.bbox.astype(int)
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=(0, 255, 0), width=2)
        draw.text((box[0], box[3] + 2), f"{name}", fill=(255, 0, 0), font=font)

    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", os.path.basename(img_path))
    pil_img.save(output_path)
    print(f"识别完成: {output_path}")

def recognize_path(input_path, model_path="models/face_embeddings.pkl"):
    known_embs, known_names = load_embeddings(model_path)
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)

    if os.path.isfile(input_path):
        recognize_image(input_path, app, known_embs, known_names)
    elif os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                recognize_image(os.path.join(input_path, file), app, known_embs, known_names)
    else:
        print("无效的输入路径。")

if __name__ == "__main__":
    input_path = input("请输入图片路径或文件夹路径：\n").strip().strip('"')
    recognize_path(input_path)
