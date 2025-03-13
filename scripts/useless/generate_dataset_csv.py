import os
import re
import pandas as pd

def collect_data_from_en_folders(root_dir):
    """
    从指定目录中收集数据，生成一个包含文本和标签的数据列表。
    :param root_dir: 数据根目录路径
    :return: 包含数据的列表，每个元素是一个字典
    """
    data_rows = []
    auto_counter = 1
    pattern = re.compile(r'^(\d+)')  # 匹配文件名中的数字 ID
    seen_texts = set()  # 用于存储已处理的文本，避免重复
    for en_label in os.listdir(root_dir):  # 遍历每个标签文件夹
        label_path = os.path.join(root_dir, en_label)
        if not os.path.isdir(label_path):  # 确保是文件夹
            continue
        for fname in os.listdir(label_path):  # 遍历文件夹中的文件
            if fname.endswith(".txt"):  # 只处理 .txt 文件
                txt_path = os.path.join(label_path, fname)
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text_en = f.read().strip()  # 读取文件内容并去除多余空格
                match = pattern.match(fname)  # 提取文件名中的数字 ID
                numeric_id = int(match.group(1)) if match else auto_counter
                if not match:
                    auto_counter += 1  # 如果没有匹配到数字 ID，则使用自动递增 ID
                
                # 检查是否已有相同文本
                if text_en in seen_texts:
                    print(f"⚠️ 跳过重复文本: {text_en}")
                    continue
                seen_texts.add(text_en)

                data_rows.append({
                    "id": numeric_id,
                    "text_en": text_en,  # 英文文本
                    "label_en": en_label  # 英文标签
                })
    return data_rows


def main():
    """
    主函数：处理训练和测试数据，生成 CSV 文件。
    """
    # 定义关键路径（根据你的实际路径调整）
    train_root = os.path.join("..", "data", "raw", "train_labeled_data_en")
    test_root = os.path.join("..", "data", "raw", "test_raw_data_en")
    output_dir = os.path.join("..", "data", "processed")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 处理训练数据
    print("🚀 处理训练数据...")
    train_data = collect_data_from_en_folders(train_root)
    train_df = pd.DataFrame(train_data).sort_values(by='id')  # 按 ID 排序
    train_csv = os.path.join(output_dir, "train_data_en.csv")
    train_df.to_csv(train_csv, index=False)  # 覆盖保存 CSV 文件
    print(f"✅ 训练集生成: {train_csv} ({len(train_df)} 条)")

    # 处理测试数据
    print("\n🚀 处理测试数据...")
    test_data = collect_data_from_en_folders(test_root)
    test_df = pd.DataFrame(test_data).sort_values(by='id')  # 按 ID 排序
    test_csv = os.path.join(output_dir, "test_data_en.csv")
    test_df.to_csv(test_csv, index=False)  # 覆盖保存 CSV 文件
    print(f"✅ 测试集生成: {test_csv} ({len(test_df)} 条)")


if __name__ == "__main__":
    main()