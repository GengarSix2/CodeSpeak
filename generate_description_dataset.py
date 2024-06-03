import os
import json
import random
import re


# keywords of solidity; immutable set
keywords = frozenset(
    {'bool', 'break', 'case', 'catch', 'const', 'continue', 'default', 'do', 'double', 'struct',
     'else', 'enum', 'payable', 'function', 'modifier', 'emit', 'export', 'extern', 'false', 'constructor',
     'float', 'if', 'contract', 'int', 'long', 'string', 'super', 'or', 'private', 'protected', 'noReentrancy',
     'public', 'return', 'returns', 'assert', 'event', 'indexed', 'using', 'require', 'uint', 'onlyDaoChallenge',
     'transfer', 'Transfer', 'Transaction', 'switch', 'pure', 'view', 'this', 'throw', 'true', 'try', 'revert',
     'bytes', 'bytes4', 'bytes32', 'internal', 'external', 'union', 'constant', 'while', 'for', 'notExecuted',
     'NULL', 'uint256', 'uint128', 'uint8', 'uint16', 'address', 'call', 'msg', 'value', 'sender', 'notConfirmed',
     'private', 'onlyOwner', 'internal', 'onlyGovernor', 'onlyCommittee', 'onlyAdmin', 'onlyPlayers', 'ownerExists',
     'onlyManager', 'onlyHuman', 'only_owner', 'onlyCongressMembers', 'preventReentry', 'noEther', 'onlyMembers',
     'onlyProxyOwner', 'confirmed', 'mapping'})

# holds known non-user-defined functions; immutable set
main_set = frozenset({'function', 'constructor', 'modifier', 'contract'})

# arguments in main function; immutable set
main_args = frozenset({'argc', 'argv'})


# input is a list of string lines
def clean_fragment(fragment):
    # dictionary; map function name to symbol name + number
    fun_symbols = {}
    # dictionary; map variable name to symbol name + number
    var_symbols = {}

    fun_count = 1
    var_count = 1

    # regular expression to catch multi-line comment
    rx_comment = re.compile('\*/\s*$')
    # regular expression to find function name candidates
    rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')
    # regular expression to find variable name candidates
    # rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?!\s*\()')
    rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()')

    # final cleaned gadget output to return to interface
    cleaned_fragment = []

    for line in fragment:
        # process if not the header line and not a multi-line commented line
        if rx_comment.search(line) is None:
            # remove all string literals (keep the quotes)
            nostrlit_line = re.sub(r'".*?"', '""', line)
            # remove all character literals
            nocharlit_line = re.sub(r"'.*?'", "''", nostrlit_line)
            # replace any non-ASCII characters with empty string
            ascii_line = re.sub(r'[^\x00-\x7f]', r'', nocharlit_line)

            # return, in order, all regex matches at string list; preserves order for semantics
            user_fun = rx_fun.findall(ascii_line)
            user_var = rx_var.findall(ascii_line)

            # Could easily make a "clean fragment" type class to prevent duplicate functionality
            # of creating/comparing symbol names for functions and variables in much the same way.
            # The comparison frozenset, symbol dictionaries, and counters would be class scope.
            # So would only need to pass a string list and a string literal for symbol names to
            # another function.
            for fun_name in user_fun:
                if len({fun_name}.difference(main_set)) != 0 and len({fun_name}.difference(keywords)) != 0:
                    # DEBUG
                    # print('comparing ' + str(fun_name + ' to ' + str(main_set)))
                    # print(fun_name + ' diff len from main is ' + str(len({fun_name}.difference(main_set))))
                    # print('comparing ' + str(fun_name + ' to ' + str(keywords)))
                    # print(fun_name + ' diff len from keywords is ' + str(len({fun_name}.difference(keywords))))
                    ###
                    # check to see if function name already in dictionary
                    if fun_name not in fun_symbols.keys():
                        # fun_symbols[fun_name] = 'FUN' + str(fun_count)
                        # fun_count += 1
                        fun_symbols[fun_name] = 'FUN' + str(random.randint(1, 20))

                    # ensure that only function name gets replaced (no variable name with same
                    # identifier); uses positive lookforward
                    ascii_line = re.sub(r'\b(' + fun_name + r')\b(?=\s*\()', fun_symbols[fun_name], ascii_line)

            for var_name in user_var:
                # next line is the nuanced difference between fun_name and var_name
                if len({var_name}.difference(keywords)) != 0 and len({var_name}.difference(main_args)) != 0:
                    # DEBUG
                    # print('comparing ' + str(var_name + ' to ' + str(keywords)))
                    # print(var_name + ' diff len from keywords is ' + str(len({var_name}.difference(keywords))))
                    # print('comparing ' + str(var_name + ' to ' + str(main_args)))
                    # print(var_name + ' diff len from main args is ' + str(len({var_name}.difference(main_args))))
                    ###
                    # check to see if variable name already in dictionary
                    if var_name not in var_symbols.keys():
                        # var_symbols[var_name] = 'VAR' + str(var_count)
                        # var_count += 1
                        var_symbols[var_name] = 'VAR' + str(random.randint(1, 50))
                    # ensure that only variable name gets replaced (no function name with same
                    # identifier; uses negative lookforward
                    ascii_line = re.sub(r'\b(' + var_name + r')\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()',
                                        var_symbols[var_name], ascii_line)

            cleaned_fragment.append(ascii_line)
    # return the list of cleaned lines
    return cleaned_fragment


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


def split_dataset():
    import json

    # jsonl_file = "datasets/Dataset_2_Contract_And_Description_Gemini.jsonl"
    jsonl_file = "datasets/Dataset_2_Contract_And_Description_GPT-3.5.jsonl"

    with open(jsonl_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # vulnerability type
    vulnerability_type_list = ["reentrancy", "timestamp", "integeroverflow", "delegatecall"]
    vul_type_dict = {}
    for vul_type in vulnerability_type_list:
        vul_type_dict[vul_type] = []

    for line in lines:
        data = json.loads(line)
        vul_type = data['vulnerability_type']
        vul_type_dict[vul_type].append(line)

    for vul_type in vulnerability_type_list:
        # random shuffle
        lines = vul_type_dict[vul_type]
        random.shuffle(lines)

        train_ratio = 0.8
        test_ratio = 0.2

        total_samples = len(lines)
        train_samples = int(total_samples * train_ratio)
        test_samples = total_samples - train_samples

        train_data = lines[:train_samples]
        test_data = lines[train_samples:]

        idx = 1
        train_data_aug = []
        for i in range(len(train_data)):
            json_data = json.loads(train_data[i])
            code = json_data["contract"]
            description = json_data["description"]
            name = json_data["contract_name"]
            label = json_data["label"]
            # idx = json_data["idx"]

            if "ERR" in description:
                continue

            code_aug = '\n'.join(clean_fragment(code.split('\n')))
            code_aug = "[Description]{}\n[Smart Contract]{}".format(description, code_aug)
            json_data_aug = {"contract": code_aug,
                             "contract_name": name,
                             "vulnerability_type": vul_type,
                             "label": label,
                             "idx": idx}
            idx += 1
            train_data_aug.append(json.dumps(json_data_aug))
            train_data_aug.append('\n')

        test_data_aug = []
        for i in range(len(test_data)):
            json_data = json.loads(test_data[i])
            code = json_data["contract"]
            description = json_data["description"]
            label = json_data["label"]
            # idx = json_data["idx"]

            code = "[Description]{}\n[Smart Contract]{}".format(description, code)
            json_data = {"contract": code,
                         "vulnerability_type": vul_type,
                         "label": label,
                         "idx": idx}
            idx += 1
            test_data_aug.append(json.dumps(json_data))
            test_data_aug.append('\n')

        output_dir = "datasets/{}".format(vul_type)
        os.makedirs(output_dir, exist_ok=True)

        train_file = os.path.join(output_dir, "train.jsonl")
        test_file = os.path.join(output_dir, "test.jsonl")

        with open(train_file, "w", encoding="utf-8") as file:
            file.writelines(train_data_aug)

        with open(test_file, "w", encoding="utf-8") as file:
            file.writelines(test_data_aug)

        print("{}Dataset is saved to {}.".format(vul_type, output_dir))


if __name__ == '__main__':
    split_dataset()
