import os
import json
import random


def read_source_code_files(folder_path, file_extensions):
    source_code_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                source_code_files.append(file[0:-4])

    sorted_code_number = sorted(source_code_files, key=lambda x: int(x))
    return sorted_code_number


def read_contract_code(code_file_path):
    with open(code_file_path, 'r', encoding='utf-8') as f:
        contract = f.read()

    contract_lines = contract.splitlines()
    contract_remove_lines = '\n'.join(line for line in contract_lines if line.strip() != "")

    return contract_remove_lines


def split_dataset(is_few_shot=False, shot_num=32):
    import json
    jsonl_file = "datasets/Dataset_2.jsonl"
    with open(jsonl_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 漏洞种类
    vulnerability_type_list = ["reentrancy", "timestamp", "integeroverflow", "delegatecall"]
    vul_type_dict = {}
    for vul_type in vulnerability_type_list:
        vul_type_dict[vul_type] = []

    for line in lines:
        data = json.loads(line)
        vul_type = data['vulnerability_type']
        vul_type_dict[vul_type].append(line)

    for vul_type in vulnerability_type_list:
        lines = vul_type_dict[vul_type]
        random.shuffle(lines)

        train_ratio = 0.8
        test_ratio = 0.2

        total_samples = len(lines)
        train_samples = int(total_samples * train_ratio)
        test_samples = total_samples - train_samples

        train_data = lines[:train_samples]
        test_data = lines[train_samples:]

        output_dir = "datasets/{}".format(vul_type)
        os.makedirs(output_dir, exist_ok=True)

        train_file = os.path.join(output_dir, "train.jsonl")
        test_file = os.path.join(output_dir, "test.jsonl")

        with open(train_file, "w", encoding="utf-8") as file:
            file.writelines(train_data)

        with open(test_file, "w", encoding="utf-8") as file:
            file.writelines(test_data)

        print("{}Dataset is saved to {}.".format(vul_type, output_dir))


if __name__ == '__main__':
    split_dataset()
