# face_recognition
A project for face recognition<br>
## 项目结构<br>
insightface_project/<br>
├── data/<br>
│   ├── 张三/<br>
│   │   ├── 1.jpg<br>
│   ├── 李四/<br>
│   │   └── 1.jpg<br>
├── models/<br>
│   └── face_embeddings.pkl<br>
├── output/<br>
├── train.py<br>
├── run.py<br>
├── simhei.ttf<br>
## 环境要求
使用python3.9<br>
  pip install insightface opencv-python pillow scikit-learn
## 使用方法
在data文件夹下创建以人名为文件名的子文件夹，子文件夹内放入需要训练的照片，运行train.py后在models文件夹就会看到模型文件<br>
得到模型文件后即可运行run.py，输入图片路径或者图片文件夹路径即可进行识别，识别结果输出在output文件夹<br>
