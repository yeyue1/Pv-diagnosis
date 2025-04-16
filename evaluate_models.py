# coding:utf-8
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from simple_cnn import SimpleCNN
from train_data import real_test, real_val

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """加载训练好的模型"""
    model = SimpleCNN(num_classes=8).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, test_dataset, model_name="未知模型"):
    """评估模型性能"""
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')
    
    # 打印结果
    print(f"\n----- {model_name} 评估结果 -----")
    print(f"准确率: {accuracy*100:.2f}%")
    print(f"精确率: {precision*100:.2f}%")
    print(f"召回率: {recall*100:.2f}%")
    print(f"F1分数: {f1*100:.2f}%")
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(8),
                yticklabels=range(8))
    plt.title(f'{model_name} 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    # 保存混淆矩阵图
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name.replace(" ", "_")}_confusion_matrix.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def load_test_data():
    """加载测试数据"""
    # 加载真实测试数据
    test_dataset = real_test()
    return test_dataset

def compare_models(model_paths, model_names):
    """比较不同模型的性能"""
    # 加载测试数据
    test_dataset = load_test_data()
    
    results = []
    
    # 评估每个模型
    for model_path, model_name in zip(model_paths, model_names):
        if os.path.exists(model_path):
            model = load_model(model_path)
            result = evaluate_model(model, test_dataset, model_name)
            results.append((model_name, result))
        else:
            print(f"警告: 模型 {model_path} 不存在，跳过")
    
    # 比较结果
    print("\n----- 模型性能比较 -----")
    print("模型名称".ljust(20) + "准确率".ljust(12) + "精确率".ljust(12) + "召回率".ljust(12) + "F1分数".ljust(12))
    print("-" * 60)
    
    for model_name, result in results:
        print(f"{model_name.ljust(20)}"
              f"{(result['accuracy']*100):.2f}%".ljust(12) +
              f"{(result['precision']*100):.2f}%".ljust(12) +
              f"{(result['recall']*100):.2f}%".ljust(12) +
              f"{(result['f1']*100):.2f}%".ljust(12))
    
    # 绘制比较图
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数']
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [result[metric] for _, result in results]
        plt.bar(x + i*width, [v*100 for v in values], width, label=metric_names[i])
    
    plt.xlabel('模型')
    plt.ylabel('百分比 (%)')
    plt.title('不同生成模型训练的CNN比较')
    plt.xticks(x + width*1.5, model_names)
    plt.legend()
    plt.savefig('results/model_comparison.png')
    plt.show()

if __name__ == "__main__":
    # 定义模型路径和名称
    model_paths = [
        "models/diffusion_cnn_model.pth",
        "models/tts_cnn_model.pth",
        "models/wgan_cnn_model.pth"
    ]
    
    model_names = [
        "Diffusion CNN",
        "TTS CNN",
        "WGAN-GP CNN"
    ]
    
    # 比较模型性能
    compare_models(model_paths, model_names)
