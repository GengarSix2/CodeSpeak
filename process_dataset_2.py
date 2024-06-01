import os
import json
import random


def read_source_code_files(folder_path, file_extensions):
    source_code_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否以指定的文件扩展名结尾，以确定是否是源码文件
            if any(file.lower().endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                source_code_files.append(file[0:-4])

    sorted_code_number = sorted(source_code_files, key=lambda x: int(x))
    return sorted_code_number


def read_contract_code(code_file_path):
    with open(code_file_path, 'r', encoding='utf-8') as f:
        # 源代码字符串
        contract = f.read()

    # 使用 splitlines() 方法将输入文本分割成行的列表
    contract_lines = contract.splitlines()
    # 使用列表推导式生成去除空行后的文本
    contract_remove_lines = '\n'.join(line for line in contract_lines if line.strip() != "")

    return contract_remove_lines


def generate_dataset():
    # 漏洞种类
    vulnerability_type_list = ["reentrancy", "timestamp", "integeroverflow", "delegatecall"]

    # 设置要读取的文件夹路径和源码文件扩展名
    dataset2_path = 'datasets/Dataset_2_preprocessing_for_vulnerabilities'

    # 可以根据需要添加其他源码文件扩展名
    source_code_extensions = ['.sol']

    # 创建一个包含JSON数据的列表
    json_data_list = []
    count_index = 0

    for i in range(len(vulnerability_type_list)):
        vul_type = vulnerability_type_list[i]

        # 读取源码文件列表
        all_file_name_path = "{}/{}/final_{}_name.txt".format(dataset2_path, vul_type, vul_type)
        source_code_files = []

        with open(all_file_name_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                source_code_files.append(line)

        # 读取数据集标签列表
        dataset2_label_path = "{}/{}/final_{}_label.txt".format(dataset2_path, vul_type, vul_type)
        ground_truth_list = []
        with open(dataset2_label_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                ground_truth_list.append(int(line))

        # 遍历源码文件
        code_list = []
        for idx, (code_path, label) in enumerate(zip(source_code_files, ground_truth_list)):
            code_path = "{}/{}/sourcecode/{}".format(dataset2_path, vul_type, code_path)
            code = read_contract_code(code_path)

            # 多分类
            # if vul_type == "delegatecall":
            #     if label == 1:
            #         label = 1
            #
            # elif vul_type == "integeroverflow":
            #     if label == 1:
            #         label = 2
            #
            # elif vul_type == "reentrancy":
            #     if label == 1:
            #         label = 3
            #
            # elif vul_type == "timestamp":
            #     if label == 1:
            #         label = 4
            if code not in code_list:
                code_list.append(code)
                json_data = {"contract": code, "vulnerability_type": vul_type, "label": label, "idx": count_index}

                count_index += 1
                json_data_list.append(json_data)
            else:
                print(code_path)
            

    # 将JSON数据写入JSONL文件
    output_file = "datasets/defect/Dataset_2.jsonl"  # 指定输出文件名
    with open(output_file, "w", encoding="utf-8") as file:
        for json_data in json_data_list:
            json_line = json.dumps(json_data, ensure_ascii=False)
            file.write(json_line + "\n")

    print(f"生成的JSONL文件已保存为 {output_file}")


def split_dataset(is_few_shot=False, shot_num=32):
    import json

    # 读取JSONL文件
    jsonl_file = "datasets/defect/Dataset_2.jsonl"  # 请替换为您的JSONL文件路径
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
        # 随机打乱数据
        lines = vul_type_dict[vul_type]
        random.shuffle(lines)

        # 设置划分比例
        if is_few_shot:
            # 小样本场景数据集
            total_samples = len(lines)
            train_samples = shot_num
            valid_samples = int(total_samples * 0.2)

            # 划分数据集
            train_data = lines[:train_samples]
            valid_data = lines[train_samples:(train_samples + valid_samples) + 1]
            test_data = lines[int(total_samples * 0.8):]

        else:
            # 正常数据集
            train_ratio = 0.8  # 训练集占比
            # valid_ratio = 0.2  # 验证集占比
            test_ratio = 0.2  # 测试集占比

            # 计算划分的样本数量
            total_samples = len(lines)
            train_samples = int(total_samples * train_ratio)
            # valid_samples = int(total_samples * valid_ratio)
            test_samples = total_samples - train_samples

            # 划分数据集
            train_data = lines[:train_samples]
            # valid_data = lines[train_samples:(train_samples + valid_samples)]
            test_data = lines[train_samples:]

        # 定义输出文件名
        output_dir = "datasets/defect/{}".format(vul_type)  # 输出文件夹名称
        os.makedirs(output_dir, exist_ok=True)

        train_file = os.path.join(output_dir, "train.jsonl")
        # valid_file = os.path.join(output_dir, "valid.jsonl")
        test_file = os.path.join(output_dir, "test.jsonl")

        # 写入划分后的数据到相应文件
        with open(train_file, "w", encoding="utf-8") as file:
            file.writelines(train_data)

        # with open(valid_file, "w", encoding="utf-8") as file:
        #     file.writelines(valid_data)

        with open(test_file, "w", encoding="utf-8") as file:
            file.writelines(test_data)

        print("{}数据集已划分并保存至{}。".format(vul_type, output_dir))


def cal_accuracy():
    import json
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

    vul_predict = {}
    vul_truth = {}
    vul_result = {}

    # 读取JSONL文件
    jsonl_file = './saved-data/predictions-2023-12-03-15-36.jsonl'  # 替换为记录预测结果的JSONL文件路径

    # 逐行读取数据并分组
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            vulnerability_type = data['vulnerability_type']
            target = data['target']
            target_predict = data['target_predict']

            if vulnerability_type not in vul_predict:
                vul_predict[vulnerability_type] = []
                vul_truth[vulnerability_type] = []

            vul_predict[vulnerability_type].append(target_predict)
            vul_truth[vulnerability_type].append(target)

    vul_type_list = ["delegatecall", "reentrancy", "timestamp", "integeroverflow"]
    for vul_type in vul_type_list:
        # 计算准确率
        accuracy = accuracy_score(vul_truth[vul_type], vul_predict[vul_type])
        # 计算精确率
        precision = precision_score(vul_truth[vul_type], vul_predict[vul_type])
        # 计算召回率
        recall = recall_score(vul_truth[vul_type], vul_predict[vul_type])
        # 计算 F1 值
        f1 = f1_score(vul_truth[vul_type], vul_predict[vul_type])

        vul_result[vul_type] = {}
        vul_result[vul_type]["accuracy"] = accuracy
        vul_result[vul_type]["precision"] = precision
        vul_result[vul_type]["recall"] = recall
        vul_result[vul_type]["f1"] = f1

        print(
            f"Vulnerability Type: {vul_type} | Number: {len(vul_truth[vul_type])} | Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}"
        )


def cal_time():
    import time

    # 代码开始
    from openprompt.data_utils import InputExample
    classes = ['negative', 'positive']

    code_path = "datasets/Dataset_2_preprocessing_for_vulnerabilities/reentrancy/sourcecode/3.sol"
    code = read_contract_code(code_path)

    code_path_2 = "datasets/Dataset_2_preprocessing_for_vulnerabilities/reentrancy/sourcecode/4.sol"
    code_2 = read_contract_code(code_path_2)

    dataset = [  # For simplicity, there's only two examples
        # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
        InputExample(
            guid=0,
            text_a=code,
        ),
        InputExample(
            guid=1,
            text_a=code_2,
        ),
        InputExample(
            guid=2,
            text_a=code_2,
        ),
        InputExample(
            guid=3,
            text_a=code_2,
        ),
        InputExample(
            guid=4,
            text_a=code,
        ),
        InputExample(
            guid=5,
            text_a=code_2,
        ),
        InputExample(
            guid=6,
            text_a=code_2,
        ),
        InputExample(
            guid=7,
            text_a=code_2,
        ),
    ]

    from openprompt.plms import load_plm
    plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "models/codebert-base")

    from openprompt.prompts import ManualTemplate
    promptTemplate = ManualTemplate(
        text='{"placeholder":"text_a"} It was {"mask"}',
        tokenizer=tokenizer,
    )

    from openprompt.prompts import ManualVerbalizer
    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words={
            "negative": ["clean", "good"],
            "positive": ["defective", "bad"],
        },
        tokenizer=tokenizer,
    )

    from openprompt import PromptForClassification
    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
    )

    from openprompt import PromptDataLoader
    data_loader = PromptDataLoader(dataset=dataset, tokenizer=tokenizer, template=promptTemplate, tokenizer_wrapper_class=WrapperClass, batch_size=8)

    import torch

    # making zero-shot inference using pretrained MLM with prompt
    promptModel.eval()
    with torch.no_grad():
        for batch in data_loader:
            # 记录开始时间
            start_time = time.time()
            logits = promptModel(batch)
            preds = torch.argmax(logits, dim=-1)

            # 记录结束时间
            end_time = time.time()

            # 计算单次检测时间
            elapsed_time = end_time - start_time
            print(f"单次检测时间: {elapsed_time:.6f} 秒")

            # predictions would be 1, 0 for classes 'positive', 'negative'
            #print(classes[preds])


if __name__ == '__main__':
    generate_dataset()
    split_dataset(is_few_shot=False, shot_num=64)
    # cal_accuracy()
    # cal_time()
