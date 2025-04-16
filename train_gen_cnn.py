# coding:utf-8
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from simple_cnn import SimpleCNN
from train_data import real_test

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def load_gen_data(gen_folder, gen_type=None):
    """
    从生成数据文件夹加载数据
    
    参数:
        gen_folder: 生成数据的根文件夹
        gen_type: 生成类型 (APA, TTS_few, guan, LC等)，如果为None则加载所有类型
    
    返回:
        X: 特征数据 [样本数, 时间步, 特征数]
        y: 标签 [样本数]
    """
    all_data = []
    all_labels = []
    
    if gen_type:
        # 仅加载特定生成类型的数据
        search_pattern = os.path.join(gen_folder, gen_type, f"{gen_type.split('_')[0]}*.csv")
    else:
        # 加载所有子文件夹数据
        search_pattern = os.path.join(gen_folder, "*", "*.csv")
    
    print(f"搜索文件: {search_pattern}")
    files = glob.glob(search_pattern)
    
    if not files:
        raise ValueError(f"在{search_pattern}找不到数据文件")
    
    print(f"找到{len(files)}个数据文件")
    
    for file_path in files:
        # 从文件名中提取标签
        filename = os.path.basename(file_path)
        if '_' in filename:
            # 假设文件名格式为 method_dataset_label.csv 或 method_label.csv
            parts = filename.split('_')
            if len(parts) >= 3:
                label = int(parts[-1].split('.')[0])
            else:
                label = int(parts[-1].split('.')[0])
        else:
            # 假设文件名格式为 label.csv
            label = int(os.path.splitext(filename)[0])
        
        try:
            # 加载CSV数据
            data = pd.read_csv(file_path, header=None).values
            
            # 数据形状重塑为 [样本数, 时间步长, 特征数]
            if data.shape[1] == 480:  # 假设每个样本是80个时间步 x 6个特征
                num_samples = data.shape[0]
                data = data.reshape((num_samples, 80, 6))
                
                all_data.append(data)
                all_labels.extend([label] * num_samples)
                print(f"从{file_path}加载了{num_samples}个样本，标签为{label}")
            else:
                print(f"跳过文件{file_path}，形状不正确: {data.shape}")
        except Exception as e:
            print(f"加载文件{file_path}时出错: {e}")
    
    if not all_data:
        raise ValueError("没有加载到有效数据")
    
    X = np.vstack(all_data)
    y = np.array(all_labels)
    
    print(f"总共加载了{X.shape[0]}个样本，形状为{X.shape}")
    return X, y

def train_cnn_with_gen_data(gen_folder, gen_type, output_model_path, epochs=50, batch_size=32, lr=0.001):
    """训练CNN模型使用生成的数据"""
    # 加载生成的数据
    X, y = load_gen_data(gen_folder, gen_type)
    
    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 创建模型
    model = SimpleCNN(num_classes=8).to(device)
    print(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练模型
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100.0 * correct / total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_model_path)
            print(f"模型已保存到 {output_model_path}, 验证准确率: {val_acc:.2f}%")
    
    return model

def evaluate_on_real_test(model_path):
    """在真实测试数据上评估模型性能"""
    # 加载模型
    model = SimpleCNN(num_classes=8).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 加载真实测试数据
    test_dataset = real_test()
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 评估
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f"在真实测试数据上的准确率: {accuracy:.2f}%")
    
    return accuracy

if __name__ == "__main__":
    # 确保models目录存在
    os.makedirs("models", exist_ok=True)
    
    # 定义要训练的生成类型和对应的模型保存路径
    gen_types = {
        "APA": "models/apa_cnn_model.pth",
        "guan": "models/guan_cnn_model.pth",
        "LC": "models/lc_cnn_model.pth",
        "TTS_few": "models/tts_few_cnn_model.pth",
        "TTS_many": "models/tts_many_cnn_model.pth",
        "WGAN_GP": "models/wgan_cnn_model.pth",
        "Diffusion_data": "models/diffusion_cnn_model.pth"
    }
    
    # 训练并评估每种生成类型的模型
    results = {}
    
    for gen_type, model_path in gen_types.items():
        print(f"\n=== 开始训练 {gen_type} 模型 ===")
        try:
            train_cnn_with_gen_data(
                gen_folder="gen_data",
                gen_type=gen_type,
                output_model_path=model_path,
                epochs=50,
                batch_size=32
            )
            
            # 在真实测试数据上评估
            accuracy = evaluate_on_real_test(model_path)
            results[gen_type] = accuracy
        except Exception as e:
            print(f"训练 {gen_type} 模型时出错: {e}")
    
    # 打印所有结果
    print("\n=== 所有模型在真实测试数据上的准确率 ===")
    for gen_type, accuracy in results.items():
        print(f"{gen_type}: {accuracy:.2f}%")
