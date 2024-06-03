import argparse
from datetime import datetime
import json
import torch
import torch.nn as nn
import random
import os
import numpy as np
from openprompt.data_utils import InputExample
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import logging
from configs import add_args


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
# get current time
current_time = datetime.now()
formatted_time = current_time.strftime("%H-%M-%S")


def read_answers(filename):
    answers = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            example = InputExample(guid=int(js['idx']), text_a=js['contract'], label=int(js['label']))
            answers.append(example)

    return answers


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate_model(model, data_loader, device, is_testing=False, args=None):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    test_examples = []  # record testing data
    preds = []  # recoed prediction results

    with torch.no_grad():
        data_loader = tqdm(data_loader, total=len(data_loader), desc="Evaluating", position=0, leave=True)
        for batch in data_loader:
            batch.to(device)
            # labels = batch['guid'].to(device)
            labels = batch['label'].to(device)
            logits = model(batch)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += len(labels)

            # record guid(target) and label(idx) only in testing
            # if is_testing:
            for i in range(len(batch['label'].tolist())):
                example = {'idx': batch['guid'].tolist()[i], 'label': batch['label'].tolist()[i]}
                test_examples.append(example)
            preds.extend(predictions.tolist())

    average_loss = total_loss / len(data_loader)
    total_accuracy = correct_predictions / total_samples

    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

    vul_type = args.vul_type
    truth = []
    predict = []
    for example, pred in zip(test_examples, preds):
        truth.append(example['label'])
        predict.append(int(pred))

    accuracy = accuracy_score(truth, predict)
    precision = precision_score(truth, predict)
    recall = recall_score(truth, predict)
    f1 = f1_score(truth, predict)

    detect_result = {"vul_type": vul_type, "number": len(truth), "accuracy": "{:.2%}".format(accuracy),
                     "precision": "{:.2%}".format(precision), "recall": "{:.2%}".format(recall),
                     "f1": "{:.2%}".format(f1),
                     "soft_prompt": args.soft_prompt, "expert_knowledge": args.expert_knowledge}
    print(detect_result)

    if is_testing:
        with open(os.path.join(args.output_dir, "predictions.jsonl"), "a") as f:
            f.write(json.dumps(detect_result))
            f.write('\n')

    return average_loss, total_accuracy


def main(vul_type, vul_description):
    # read command-line arguments
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # read datasets
    train_dataset = read_answers("{}/{}/train.jsonl".format(args.train_filename, vul_type))
    test_dataset = read_answers("{}/{}/test.jsonl".format(args.test_filename, vul_type))

    if args.do_eval:
        valid_dataset = read_answers("{}/{}/valid.jsonl".format(args.valid_filename, vul_type))
    else:
        valid_dataset = read_answers("{}/{}/test.jsonl".format(args.train_filename, vul_type))

    args.vul_type = vul_type

    classes = ['negative', 'positive']

    from openprompt.plms import load_plm
    plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "../models/codebert-base")

    # <====================USE DIFFERENT TEMPLATE====================>
    from openprompt.prompts import ManualTemplate, SoftTemplate, MixedTemplate
    if args.soft_prompt:
        # soft prompt + expert knowledge
        if args.expert_knowledge:
            print("<====================SOFT PROMPT + EXPERT KNOWLEDGE====================>")
            promptTemplate = MixedTemplate(
                model=plm,
                text="[Expert Knowledge]{}\n".format(vul_description) + '{"placeholder":"text_a"} The code is {"mask"}.',
                tokenizer=tokenizer,
            )

        # only soft prompt
        else:
            print("<====================SOFT PROMPT====================>")
            promptTemplate = MixedTemplate(
                model=plm,
                text='{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"mask"}.',
                tokenizer=tokenizer,
            )

    else:
        # hard prompt + expert knowledge
        if args.expert_knowledge:
            print("<====================HARD PROMPT + EXPERT KNOWLEDGE====================>")
            promptTemplate = ManualTemplate(
                text="[Expert Knowledge]{}\n".format(vul_description) + '{"placeholder":"text_a"} The code is {"mask"}.',
                tokenizer=tokenizer,
            )

        # only hard prompt
        else:
            print("<====================HARD PROMPT====================>")
            promptTemplate = ManualTemplate(
                text='{"placeholder":"text_a"} The code is {"mask"}.',
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
    train_data_loader = PromptDataLoader(dataset=train_dataset,
                                         tokenizer=tokenizer,
                                         template=promptTemplate,
                                         tokenizer_wrapper_class=WrapperClass,
                                         batch_size=args.train_batch_size)

    test_data_loader = PromptDataLoader(dataset=test_dataset,
                                        tokenizer=tokenizer,
                                        template=promptTemplate,
                                        tokenizer_wrapper_class=WrapperClass,
                                        batch_size=args.test_batch_size)

    valid_data_loader = PromptDataLoader(dataset=valid_dataset,
                                         tokenizer=tokenizer,
                                         template=promptTemplate,
                                         tokenizer_wrapper_class=WrapperClass,
                                         batch_size=args.valid_batch_size)

    # <====================training part====================>
    model = promptModel
    model = model.cuda()
    set_seed(args)
    num_train_epochs = args.num_train_epochs
    max_steps = num_train_epochs * len(train_data_loader)
    warm_up_steps = len(train_data_loader)

    gradient_accumulation_steps = args.gradient_accumulation_steps
    lr = args.learning_rate
    adam_epsilon = args.adam_epsilon
    max_grad_norm = args.max_grad_norm

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }, {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1, num_training_steps=max_steps)

    logger.info("***** Running Training *****")
    logger.info("  Num examples = %d", len(train_data_loader))
    logger.info("  Total optimization steps = %d", max_steps)

    global_step = 0
    save_steps = max(len(train_data_loader), 1)
    best_acc = 0.0
    not_acc_inc_cnt = 0
    early_stop = False

    model.zero_grad()
    for idx in range(0, num_train_epochs):
        logger.info("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()

        total_loss = 0.0
        sum_loss = 0.0
        logger.info("******* Epoch %d *****", idx + 1)
        train_data_loader = tqdm(train_data_loader, total=len(train_data_loader), desc="Training")
        for batch_idx, batch in enumerate(train_data_loader):
            batch.to(device)
            labels = batch['label'].to(device)

            model.train()
            logits = model(batch)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)

            sum_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                train_data_loader.set_description("[{}] Train loss {}".format(idx + 1, round(total_loss, 3)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += sum_loss
                sum_loss = 0.
                global_step += 1

            # <====================validing part====================>
            if args.do_eval and ((batch_idx + 1) % save_steps == 0):
                logger.info("\n----- Running Validating -----")
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                valid_loss, valid_accuracy = evaluate_model(model, valid_data_loader, device, False, args)
                logger.info("Validation Loss: {:.4f}, Validation Accuracy: {:.4f}".format(valid_loss, valid_accuracy))

                if valid_accuracy > best_acc:
                    not_acc_inc_cnt = 0
                    logger.info("  Best acc: %s", round(valid_accuracy, 4))
                    logger.info("  " + "*" * 20)
                    best_acc = valid_accuracy

                    # Save best checkpoint for best ppl
                    output_model_dir = os.path.join(args.output_model_dir, 'checkpoint-best-acc')
                    if not os.path.exists(output_model_dir):
                        os.makedirs(output_model_dir)

                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(output_model_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the best ppl model into %s", output_model_file)

                else:
                    not_acc_inc_cnt += 1
                    if not_acc_inc_cnt > args.patience:
                        early_stop = True
                        break
                    logger.info("acc does not increase for %d epochs", not_acc_inc_cnt)

            # <====================testing part====================>
            if (batch_idx + 1) % save_steps == 0:
                logger.info("\n----- Running Testing -----")
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                model.to(device)
                test_loss, test_accuracy = evaluate_model(model, test_data_loader, device, False, args)
                logger.info("Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(test_loss, test_accuracy))

        if early_stop:
            break

        logger.info("Training epoch {}, num_steps {}, total_loss: {:.4f}".format(idx + 1, global_step, total_loss))

    # <====================testing part====================>
    logger.info("***** Running Testing *****")
    model.to(device)
    test_loss, test_accuracy = evaluate_model(model, test_data_loader, device, True, args)
    logger.info("Test Loss: {:.4f}, Test Accuracy: {:.4f}".format(test_loss, test_accuracy))


if __name__ == '__main__':
    # vulnerability type
    vulnerability_type_list = ["reentrancy", "timestamp", "integeroverflow", "delegatecall"]

    vulnerability_description_dict = {
        "reentrancy":
            "Reentrancy vulnerability occurs when call.value in a smart contract allows recursive calls, risking unauthorized fund manipulation.",
        "timestamp":
            "Timestamp vulnerability occurs when critical operations depend on block.timestamp, risking manipulation in smart contracts.",
        "integeroverflow":
            "Integer Overflow/Underflow occurs when arithmetic operations between variables lead to values beyond the allowable range in a smart contract.",
        "delegatecall":
            "Dangerous delegatecall occurs when delegate is used in conditions for critical operations without proper owner-specified target caller.",
    }

    for vul_type in vulnerability_type_list:
        main(vul_type, vulnerability_description_dict[vul_type])
