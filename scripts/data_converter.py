import json

def convert_to_alpaca(input_path, output_dir):
    with open(input_path) as f:
        raw_data = json.load(f)
    
    alpaca_data = [{
        "instruction": "将以下非专业混音描述转换为专业术语",
        "input": item["text"],
        "output": item["label"]
    } for item in raw_data]
    
    output_path = f"{output_dir}/train.json"
    with open(output_path, "w") as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
    print(f"数据已保存至 {output_path}")

if __name__ == "__main__":
    convert_to_alpaca(
        input_path="/Volumes/Study/prj/data/processed/cleaned_data.json",
        output_dir="/Volumes/Study/prj/LLaMA-Factory/data/mix_term"
    )