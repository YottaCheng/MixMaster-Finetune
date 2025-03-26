import json
import argparse
import os

# Label mapping
LABEL_MAPPING = {
    "低频": "low frequency",
    "中频": "mid frequency",
    "高频": "high frequency",
    "混响": "reverb",
    "效果器": "effects",
    "声场": "sound field",
    "压缩": "compression",
    "音量": "volume"
}

def convert_instruction_format(input_file, output_file=None):
    """
    Convert instruction-input-output format to id-text-labels format.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str, optional): Path to output JSON file. If None, will use input_file name with _converted suffix.
    
    Returns:
        str: Path to the output file
    """
    # Determine output file path
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_converted.json"
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate input data
    if not isinstance(data, list):
        raise ValueError("Input data must be a JSON array")
    
    # Convert data
    converted_data = []
    text_seen = set()  # Track seen texts to remove duplicates
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            # Skip non-object items
            print(f"Skipping non-object item at index {i}")
            continue
            
        # Check if item has instruction-input-output format
        has_instruction_format = all(k in item for k in ['instruction', 'input'])
        
        if has_instruction_format:
            # Extract text from input
            text = item.get('input', '')
            
            # Check for duplicate text
            if text in text_seen:
                print(f"Skipping duplicate text at index {i}: {text[:30]}...")
                continue
            
            text_seen.add(text)
            
            # Extract labels from output
            output = item.get('output', '')
            if isinstance(output, str):
                # Split by commas and clean up
                labels = [label.strip() for label in output.split(',')]
            else:
                labels = []
            
            # Create new entry with required format
            converted_item = {
                'id': str(len(converted_data)),  # Use sequential IDs for non-duplicates
                'text': text,
                'labels': labels
            }
            converted_data.append(converted_item)
        else:
            # Try to extract data from other formats
            try:
                # If there's a text field, use that
                if 'text' in item:
                    text = item['text']
                else:
                    # Otherwise use the whole item as string
                    text = str(item)
                
                # Check for duplicate text
                if text in text_seen:
                    print(f"Skipping duplicate text at index {i}: {text[:30]}...")
                    continue
                
                text_seen.add(text)
                
                # Create entry with empty labels
                converted_item = {
                    'id': str(len(converted_data)),  # Use sequential IDs for non-duplicates
                    'text': text,
                    'labels': []
                }
                converted_data.append(converted_item)
            except Exception as e:
                print(f"Error processing item {i}: {e}")
    
    print(f"Filtered out {len(data) - len(converted_data)} duplicate entries")
    
    # Write converted data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(converted_data)} items")
    print(f"Output saved to {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Convert instruction-input-output format to id-text-labels format')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('-o', '--output', help='Output JSON file path (optional)')
    
    args = parser.parse_args()
    
    try:
        output_file = convert_instruction_format(args.input_file, args.output)
        print(f"Conversion completed successfully. Output saved to {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()