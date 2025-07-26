import os
import pickle
import numpy as np
from insightface.app import FaceAnalysis
import cv2

def read_image(path):
    """可靠读取包含非 ASCII 字符的图像路径"""
    try:
        with open(path, 'rb') as f:
            img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"读取图像失败: {path}, 错误: {e}")
        return None

def train_face_embeddings(data_dir, save_path="models/face_embeddings.pkl"):
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    embeddings = []
    names = []

    # 使用 os.walk 遍历所有子目录
    for root, _, files in os.walk(data_dir):
        person = os.path.basename(root)
        for img_file in files:
            if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(root, img_file)
                
                # 验证文件是否存在
                if not os.path.exists(img_path):
                    print(f"文件不存在: {img_path}")
                    continue
                
                img = read_image(img_path)
                if img is None:
                    print(f"读取失败: {img_path}")
                    continue

                faces = app.get(img)
                if not faces:
                    print(f"未检测到人脸: {img_path}")
                    continue

                emb = faces[0].embedding  # 取第一个人脸
                embeddings.append(emb)
                names.append(person)

    if embeddings:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump((np.array(embeddings), np.array(names)), f)
        print(f"训练完成，共提取 {len(embeddings)} 张人脸，保存至 {save_path}")
    else:
        print("未提取到任何人脸")

if __name__ == "__main__":
    train_face_embeddings("data")
