import os
import cv2
import torch
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from torchvision import models, transforms

def normalize_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    normalized_image = transform(image)
    normalized_image = torch.unsqueeze(normalized_image, 0)
    return normalized_image

def extract_frame_features(frame, model):
    normalized_frame = normalize_image(frame)
    model.eval()
    with torch.no_grad():
        features = model(normalized_frame)
    return features.squeeze().numpy()

def sharpening(img): # 锐化
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened_img = cv2.filter2D(img, -1, kernel)
    return sharpened_img

def gaussian_blur(img): # 高斯模糊
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    return blurred_img

def hist(image): #直方图均衡增强
    r, g, b = cv2.split(image)
    r1 = cv2.equalizeHist(r)
    g1 = cv2.equalizeHist(g)
    b1 = cv2.equalizeHist(b)
    image_equal_clo = cv2.merge([r1, g1, b1])
    return image_equal_clo

def process_video(video_path, timef=5):
    cap = cv2.VideoCapture(video_path)
    isOpened = cap.isOpened
    imageNum = 0
    sum = 0
    averaged_frame_features = []
    while isOpened:
        sum += 1
        frameState, frame = cap.read()
        if frameState == True and (sum % timef == 0):
            enhanced1_frame = sharpening(frame)
            enhanced2_frame = gaussian_blur(enhanced1_frame)
            enhanced_frame = hist(enhanced2_frame)
            imageNum += 1
            frame_features = extract_frame_features(enhanced_frame, pretrained_model)
            averaged_frame_features.append(frame_features)
            fileName = 'D:/NTU/EE6222-Machine Vision/Assignment 2/Random frame sampling/video image' + str(imageNum) + '.jpg'
            cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        elif frameState == False:
            break
    cap.release()
    return np.mean(averaged_frame_features, axis=0)

def read_dataset_file(file_path):
    labels = []
    video_paths = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.split()
        label = int(parts[1])
        video_path = parts[2]
        labels.append(label)
        video_paths.append(video_path)
    return labels, video_paths;

def bayes_classifier(X_train, y_train, X_test, y_test):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(" accuracy:", accuracy)
    report = classification_report(y_test, y_pred)
    print("report:\n", report)

train_file_path = "D:/NTU/EE6222-Machine Vision/Assignment 2/EE6222 train and validate 2023/train.txt"
test_file_path = "D:/NTU/EE6222-Machine Vision/Assignment 2/EE6222 train and validate 2023/validate.txt"
train_root_path = 'D:/NTU/EE6222-Machine Vision/Assignment 2/EE6222 train and validate 2023/train'
test_root_path = 'D:/NTU/EE6222-Machine Vision/Assignment 2/EE6222 train and validate 2023/validate'

pretrained_model = models.resnet18(pretrained=True)

y_train, train_video_paths = read_dataset_file(train_file_path)
full_train_paths = [os.path.join(train_root_path, train_video_paths) for train_video_paths in train_video_paths] # 提取和保存训练集特征
train_features = []
for video_path in full_train_paths:
    print(video_path)
    features = process_video(video_path)
    train_features.append(features)
    X_train = np.array(train_features)

# 读取测试集
y_test, test_video_paths = read_dataset_file(test_file_path)
full_test_paths = [os.path.join(test_root_path, test_video_paths) for test_video_paths in test_video_paths] # 提取和保存测试集特征
test_features = []
for video_path in full_test_paths:
    print(video_path)
    features = process_video(video_path)
    test_features.append(features)
    X_test = np.array(test_features)

bayes_classifier(X_train, y_train, X_test, y_test)