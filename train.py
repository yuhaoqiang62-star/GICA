import argparse
import os
import random
import pickle
import numpy as np
from datetime import datetime

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torch.nn import MSELoss

from transformers import get_linear_schedule_with_warmup, DebertaV2Tokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from deberta_GICA import GICA_DeBertaForSequenceClassification
import global_configs
from global_configs import DEVICE


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_result_filename(fusion_mode=None):
    timestamp = get_timestamp()
    if fusion_mode:
        return f"result_{fusion_mode}_{timestamp}.txt"
    return f"result_{timestamp}.txt"


def log_results(file_path, message):
    with open(file_path, 'a',encoding='utf-8') as f:
        f.write(message + '\n')
        f.flush()


def calculate_binary_accuracy(preds, labels):
    binary_preds = (preds >= 0).astype(int)
    binary_labels = (labels >= 0).astype(int)
    binary_acc = accuracy_score(binary_labels, binary_preds)
    return binary_acc


# 🔥 添加保存检查点函数
def save_checkpoint(model, optimizer, epoch, valid_loss, is_best=False):
    """保存模型检查点"""
    os.makedirs(args.model_save_dir, exist_ok=True)

    timestamp = get_timestamp()
    fusion_suffix = f"_{args.fusion_mode}" if hasattr(args, 'fusion_mode') else ""
    gating_suffix = f"_{args.gating_mode}" if hasattr(args, 'gating_mode') else ""

    if is_best:
        checkpoint_name = f"best_model_{args.dataset}{fusion_suffix}{gating_suffix}.pt"
    else:
        checkpoint_name = f"checkpoint_epoch{epoch}_{args.dataset}{fusion_suffix}{gating_suffix}_{timestamp}.pt"

    checkpoint_path = os.path.join(args.model_save_dir, checkpoint_name)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_loss': valid_loss,
        'args': vars(args),
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


# 🔥 添加保存最终模型函数
def save_final_model(model, test_acc, test_mae, test_corr, test_f_score, test_ba):
    """保存训练完成的最终模型"""
    os.makedirs(args.model_save_dir, exist_ok=True)

    timestamp = get_timestamp()
    fusion_suffix = f"_{args.fusion_mode}" if hasattr(args, 'fusion_mode') else ""
    gating_suffix = f"_{args.gating_mode}" if hasattr(args, 'gating_mode') else ""

    model_name = f"final_model_{args.dataset}{fusion_suffix}{gating_suffix}_{timestamp}.pt"
    model_path = os.path.join(args.model_save_dir, model_name)

    model_info = {
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc,
        'test_mae': test_mae,
        'test_corr': test_corr,
        'test_f_score': test_f_score,
        'test_ba': test_ba,
        'args': vars(args),
        'timestamp': timestamp,
    }

    torch.save(model_info, model_path)
    print(f"\n💾 Final model saved: {model_path}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test MAE: {test_mae:.4f}")
    print(f"   Test Correlation: {test_corr:.4f}")
    print(f"   Test F1-Score: {test_f_score:.4f}")
    print(f"   Binary Accuracy: {test_ba:.4f}\n")

    # 🔥 同时保存tokenizer
    tokenizer_save_dir = os.path.join(args.model_save_dir, f"tokenizer_{args.dataset}_{timestamp}")
    tokenizer = get_tokenizer(args.model)
    tokenizer.save_pretrained(tokenizer_save_dir)
    print(f"💾 Tokenizer saved: {tokenizer_save_dir}\n")

    return model_path


# 🔥 添加加载模型函数（可选，用于推理或继续训练）
def load_model(checkpoint_path):
    """加载已保存的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    saved_args = checkpoint['args']
    for key, value in saved_args.items():
        if hasattr(args, key):
            setattr(args, key, value)

    model = GICA_DeBertaForSequenceClassification.from_pretrained(
        args.model, multimodal_config=args, num_labels=1,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)

    print(f"✅ Model loaded from: {checkpoint_path}")

    if 'test_acc' in checkpoint:
        print(f"   Test Accuracy: {checkpoint['test_acc']:.4f}")
        print(f"   Test MAE: {checkpoint['test_mae']:.4f}")
        print(f"   Test Correlation: {checkpoint['test_corr']:.4f}")
        print(f"   Test F1-Score: {checkpoint['test_f_score']:.4f}")
        print(f"   Binary Accuracy: {checkpoint['test_ba']:.4f}")

    return model


def train(
        model,
        train_dataloader,
        validation_dataloader,
        test_data_loader,
        optimizer,
        scheduler,
        result_file,
):
    valid_losses = []
    test_accuracies = []
    mae_list = []
    corr_list = []
    f1_list = []
    ba_list = []

    best_valid_loss = float('inf')
    best_epoch = 0
    best_model_path = None

    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, epoch_i, args.n_epochs)
        valid_loss = eval_epoch(model, validation_dataloader, epoch_i, args.n_epochs)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch_i
            if args.save_model:
                best_model_path = save_checkpoint(model, optimizer, epoch_i, valid_loss, is_best=True)

        if epoch_i != args.n_epochs - 1:
            train_message = f"TRAIN: epoch:{epoch_i + 1}, train_loss:{train_loss}, valid_loss:{valid_loss}"
            print(train_message)
            log_results(result_file, train_message)
        else:
            test_acc, test_mae, test_corr, test_f_score, test_ba = test_score_model(
                model, test_data_loader, epoch_i, args.n_epochs
            )
            test_message = (
                f"TEST: train_loss:{train_loss}, valid_loss:{valid_loss}, "
                f"test_acc:{test_acc}, mae:{test_mae}, corr:{test_corr}, "
                f"f1_score:{test_f_score}, binary_acc:{test_ba}"
            )
            print(test_message)
            log_results(result_file, test_message)

            if args.save_model:
                final_model_path = save_final_model(model, test_acc, test_mae, test_corr, test_f_score, test_ba)

    best_message = f"\n🏆 Best validation loss: {best_valid_loss:.4f} at epoch {best_epoch + 1}"
    if best_model_path:
        best_message += f"\n💾 Best model saved at: {best_model_path}"
    print(best_message)
    log_results(result_file, best_message)

    return train_loss, valid_loss, test_acc, test_mae, test_corr, test_f_score, test_ba


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=64)
# 先解析已有的 --dataset 参数
args, _ = parser.parse_known_args()

# 根据dataset设置epochs默认值
default_epochs = {"mosi": 18, "mosei": 3}
parser.add_argument("--n_epochs", type=int, default=default_epochs[args.dataset])

#parser.add_argument("--n_epochs", type=int, default=18)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=128)
parser.add_argument('--inter_dim', default=256, help='dimension of inter layers', type=int)
parser.add_argument("--drop_prob", help='drop probability for dropout -- encoder', default=0.3, type=float)
parser.add_argument('--p_lambda', default=0.3, help='coefficient -- lambda', type=float)
parser.add_argument('--p_beta', default=8, help='coefficient -- beta', type=float)
parser.add_argument('--p_gamma', default=32, help='coefficient -- gamma', type=float)
parser.add_argument('--beta_shift', default=1.0, help='coefficient -- shift', type=float)
parser.add_argument('--IB_coef', default=10, type=float)
parser.add_argument('--B0_dim', default=128, type=float)
parser.add_argument('--B1_dim', default=64, type=float)
parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
parser.add_argument("--save_model", action="store_true", default=False, help="Whether to save the trained model")
parser.add_argument("--model_save_dir", type=str, default="saved_models", help="Directory to save models")

# 🔥 添加消融实验参数
parser.add_argument("--run_ablation", action="store_true", default=False,
                    help="Run ablation study for fusion modes")
parser.add_argument("--fusion_mode", type=str, default="full",
                    choices=["b1_only", "b1_acoustic", "b1_visual", "full"],
                    help="Fusion mode for ablation study")

# 🔥 门控消融实验参数
parser.add_argument("--run_gating_ablation", action="store_true", default=False,
                    help="Run ablation study for gating mechanisms")
parser.add_argument("--gating_mode", type=str, default="dual_gating",
                    choices=["no_gating", "single_gating", "dual_gating"],
                    help="Gating mechanism mode")

# 🔥 组合消融实验参数
parser.add_argument("--run_combined_ablation", action="store_true", default=False,
                    help="Run combined ablation study (gating + fusion)")

# 🔥 添加模型加载参数
parser.add_argument("--load_model", type=str, default=None,
                    help="Path to load a pre-trained model checkpoint")

args = parser.parse_args()

global_configs.set_dataset_config(args.dataset)
ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM = (global_configs.ACOUSTIC_DIM, global_configs.VISUAL_DIM, global_configs.TEXT_DIM)


class InputFeatures(object):
    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):
        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        prepare_input = prepare_deberta_input

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
            tokens, visual, acoustic, tokenizer
        )

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


def prepare_deberta_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids


def get_tokenizer(model):
    return DebertaV2Tokenizer.from_pretrained(model)


def get_appropriate_dataset(data):
    tokenizer = get_tokenizer(args.model)

    features = convert_to_features(data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_input_mask = torch.tensor(np.array([f.input_mask for f in features]), dtype=torch.long)
    all_label_ids = torch.tensor(np.array([f.label_id for f in features]), dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_label_ids,
    )

    return dataset


def set_up_data_loader():
    with open(f"./datasets/{args.dataset}.pkl", "rb") as handle:
        all_data = pickle.load(handle)

    train_data = all_data["train"]
    dev_data = all_data["dev"]
    test_data = all_data["test"]

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (
            int(
                len(train_dataset) / args.train_batch_size /
                args.gradient_accumulation_step
            )
            * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


def set_random_seed(seed: int):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Seed: {}".format(seed))


def prep_for_training(num_train_optimization_steps: int):
    model = GICA_DeBertaForSequenceClassification.from_pretrained(
        args.model, multimodal_config=args, num_labels=1,
    )

    model.to(DEVICE)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler, epoch, max_epochs):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, attention_mask, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)

        visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
        acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())

        logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1 = model(
            input_ids,
            visual_norm,
            acoustic_norm,
            attention_mask=attention_mask,
            epoch=epoch,
            max_epochs=max_epochs
        )
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), label_ids.view(-1)) + 2 / (args.p_beta + args.p_gamma) * IB_loss

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader, epoch, max_epochs):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, attention_mask, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())

            logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1 = model(
                input_ids,
                visual_norm,
                acoustic_norm,
                attention_mask=attention_mask,
                epoch=epoch,
                max_epochs=max_epochs
            )
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader, epoch, max_epochs):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, attention_mask, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())

            logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1 = model(
                input_ids,
                visual_norm,
                acoustic_norm,
                attention_mask=attention_mask,
                epoch=epoch,
                max_epochs=max_epochs
            )

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def test_score_model(model: nn.Module, test_dataloader: DataLoader, epoch, max_epochs, use_zero=False):
    preds, y_test = test_epoch(model, test_dataloader, epoch, max_epochs)
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])

    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    binary_acc = calculate_binary_accuracy(preds, y_test)

    preds = preds >= 0
    y_test = y_test >= 0

    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)

    return acc, mae, corr, f_score, binary_acc


# 🔥 门控消融实验函数
def run_gating_ablation_study():
    """运行门控机制消融实验"""

    print("\n" + "=" * 80)
    print("🔥 开始门控机制消融实验 (Gating Mechanism Ablation Study)")
    print("=" * 80 + "\n")

    gating_modes = ['no_gating', 'single_gating', 'dual_gating']
    mode_descriptions = {
        'no_gating': '移除所有门控瓶颈机制',
        'single_gating': '单门控瓶颈机制 (仅第一层)',
        'dual_gating': '双重门控瓶颈机制 (两层都使用)'
    }

    all_results = {}

    for mode in gating_modes:
        print("\n" + "=" * 80)
        print(f"🚀 运行模式: {mode} - {mode_descriptions[mode]}")
        print("=" * 80 + "\n")

        args.gating_mode = mode
        result_file = get_result_filename(fusion_mode=args.fusion_mode) + f"_gating_{mode}"

        config_message = (
            f"Gating Ablation Study\n"
            f"Gating Mode: {mode} - {mode_descriptions[mode]}\n"
            f"Fusion Mode: {args.fusion_mode}\n"
            f"Dataset: {args.dataset}\n"
            f"Seed: {args.seed}\n"
            f"Epochs: {args.n_epochs}\n"
        )
        log_results(result_file, config_message)
        print(config_message)

        set_random_seed(args.seed)

        (
            train_data_loader,
            dev_data_loader,
            test_data_loader,
            num_train_optimization_steps,
        ) = set_up_data_loader()

        model, optimizer, scheduler = prep_for_training(num_train_optimization_steps)

        train_loss, valid_loss, test_acc, test_mae, test_corr, test_f_score, test_ba = train(
            model,
            train_data_loader,
            dev_data_loader,
            test_data_loader,
            optimizer,
            scheduler,
            result_file,
        )

        all_results[mode] = {
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'test_acc': test_acc,
            'test_mae': test_mae,
            'test_corr': test_corr,
            'test_f_score': test_f_score,
            'test_ba': test_ba
        }

        print(f"\n✅ 模式 {mode} 完成!")

    print("\n" + "=" * 80)
    print("📊 门控消融实验汇总结果 (Gating Ablation Summary)")
    print("=" * 80 + "\n")

    summary_file = "result_gating_summary_" + get_timestamp() + ".txt"
    summary_header = (
            f"{'Gating Mode':<20} {'Description':<40} {'Acc':<8} {'MAE':<8} {'Corr':<8} {'F1':<8} {'BA':<8}\n"
            + "=" * 120 + "\n"
    )
    print(summary_header)
    log_results(summary_file, summary_header)

    for mode in gating_modes:
        results = all_results[mode]
        summary_line = (
            f"{mode:<20} {mode_descriptions[mode]:<40} "
            f"{results['test_acc']:<8.4f} {results['test_mae']:<8.4f} "
            f"{results['test_corr']:<8.4f} {results['test_f_score']:<8.4f} "
            f"{results['test_ba']:<8.4f}"
        )
        print(summary_line)
        log_results(summary_file, summary_line)

    print("\n" + "=" * 80)
    print("🎉 门控消融实验全部完成!")
    print("=" * 80 + "\n")

    return all_results


# 🔥 组合消融实验函数
def run_combined_ablation_study():
    """运行组合消融实验(门控+融合)"""

    print("\n" + "=" * 80)
    print("🔥 开始组合消融实验 (Combined Ablation: Gating + Fusion)")
    print("=" * 80 + "\n")

    gating_modes = ['no_gating', 'single_gating', 'dual_gating']
    fusion_modes = ['b1_only', 'b1_acoustic', 'b1_visual', 'full']

    all_results = {}

    for g_mode in gating_modes:
        for f_mode in fusion_modes:
            config_name = f"{g_mode}_{f_mode}"
            print("\n" + "=" * 80)
            print(f"🚀 运行配置: {config_name}")
            print(f"  门控: {g_mode}, 融合: {f_mode}")
            print("=" * 80 + "\n")

            args.gating_mode = g_mode
            args.fusion_mode = f_mode
            result_file = f"result_{config_name}_{get_timestamp()}.txt"

            config_message = (
                f"Combined Ablation Study\n"
                f"Gating Mode: {g_mode}\n"
                f"Fusion Mode: {f_mode}\n"
                f"Dataset: {args.dataset}\n"
                f"Seed: {args.seed}\n"
            )
            log_results(result_file, config_message)

            set_random_seed(args.seed)

            (
                train_data_loader,
                dev_data_loader,
                test_data_loader,
                num_train_optimization_steps,
            ) = set_up_data_loader()

            model, optimizer, scheduler = prep_for_training(num_train_optimization_steps)

            train_loss, valid_loss, test_acc, test_mae, test_corr, test_f_score, test_ba = train(
                model,
                train_data_loader,
                dev_data_loader,
                test_data_loader,
                optimizer,
                scheduler,
                result_file,
            )

            all_results[config_name] = {
                'gating': g_mode,
                'fusion': f_mode,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'test_acc': test_acc,
                'test_mae': test_mae,
                'test_corr': test_corr,
                'test_f_score': test_f_score,
                'test_ba': test_ba
            }

            print(f"\n✅ 配置 {config_name} 完成!")

    print("\n" + "=" * 80)
    print("📊 组合消融实验汇总结果")
    print("=" * 80 + "\n")

    summary_file = "result_combined_summary_" + get_timestamp() + ".txt"
    summary_header = (
            f"{'Gating':<15} {'Fusion':<15} {'Acc':<8} {'MAE':<8} {'Corr':<8} {'F1':<8} {'BA':<8}\n"
            + "=" * 80 + "\n"
    )
    print(summary_header)
    log_results(summary_file, summary_header)

    for config_name, results in all_results.items():
        summary_line = (
            f"{results['gating']:<15} {results['fusion']:<15} "
            f"{results['test_acc']:<8.4f} {results['test_mae']:<8.4f} "
            f"{results['test_corr']:<8.4f} {results['test_f_score']:<8.4f} "
            f"{results['test_ba']:<8.4f}"
        )
        print(summary_line)
        log_results(summary_file, summary_line)

    return all_results


# 🔥 消融实验主函数
def run_ablation_study():
    """运行融合层消融实验"""

    print("\n" + "=" * 80)
    print("🔥 开始融合层消融实验 (Fusion Layer Ablation Study)")
    print("=" * 80 + "\n")

    fusion_modes = ['b1_only', 'b1_acoustic', 'b1_visual', 'full']
    mode_descriptions = {
        'b1_only': '只使用B1 (文本压缩信息)',
        'b1_acoustic': 'B1 + 声学重构',
        'b1_visual': 'B1 + 视觉重构',
        'full': 'B1 + 声学重构 + 视觉重构 (完整模型)'
    }

    all_results = {}

    for mode in fusion_modes:
        print("\n" + "=" * 80)
        print(f"🚀 运行模式: {mode} - {mode_descriptions[mode]}")
        print("=" * 80 + "\n")

        args.fusion_mode = mode
        result_file = get_result_filename(fusion_mode=mode)

        config_message = (
            f"Ablation Study - Fusion Mode: {mode}\n"
            f"Description: {mode_descriptions[mode]}\n"
            f"Dataset: {args.dataset}\n"
            f"Seed: {args.seed}\n"
            f"Epochs: {args.n_epochs}\n"
            f"Learning Rate: {args.learning_rate}\n"
        )
        log_results(result_file, config_message)
        print(config_message)

        set_random_seed(args.seed)

        (
            train_data_loader,
            dev_data_loader,
            test_data_loader,
            num_train_optimization_steps,
        ) = set_up_data_loader()

        model, optimizer, scheduler = prep_for_training(num_train_optimization_steps)

        train_loss, valid_loss, test_acc, test_mae, test_corr, test_f_score, test_ba = train(
            model,
            train_data_loader,
            dev_data_loader,
            test_data_loader,
            optimizer,
            scheduler,
            result_file,
        )

        all_results[mode] = {
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'test_acc': test_acc,
            'test_mae': test_mae,
            'test_corr': test_corr,
            'test_f_score': test_f_score,
            'test_ba': test_ba
        }

        print(f"\n✅ 模式 {mode} 完成!")

    print("\n" + "=" * 80)
    print("📊 消融实验汇总结果 (Ablation Study Summary)")
    print("=" * 80 + "\n")

    summary_file = get_result_filename(fusion_mode="ablation_summary")

    summary_header = (
            f"{'Mode':<20} {'Description':<40} {'Acc':<8} {'MAE':<8} {'Corr':<8} {'F1':<8} {'BA':<8}\n"
            + "=" * 120 + "\n"
    )
    print(summary_header)
    log_results(summary_file, summary_header)

    for mode in fusion_modes:
        results = all_results[mode]
        summary_line = (
            f"{mode:<20} {mode_descriptions[mode]:<40} "
            f"{results['test_acc']:<8.4f} {results['test_mae']:<8.4f} "
            f"{results['test_corr']:<8.4f} {results['test_f_score']:<8.4f} "
            f"{results['test_ba']:<8.4f}"
        )
        print(summary_line)
        log_results(summary_file, summary_line)

    print("\n" + "=" * 80)
    print("🎉 消融实验全部完成!")
    print("=" * 80 + "\n")

    return all_results


def main():
    print(f"Starting experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Save Model: {args.save_model}")
    if args.save_model:
        print(f"Model Save Directory: {args.model_save_dir}")

    if args.load_model:
        print(f"\n📥 Loading pre-trained model from: {args.load_model}")
        model = load_model(args.load_model)
        print("Model loaded successfully!\n")
        return

    if args.run_combined_ablation:
        run_combined_ablation_study()
    elif args.run_gating_ablation:
        run_gating_ablation_study()
    elif args.run_ablation:
        run_ablation_study()
    else:
        result_file = get_result_filename(fusion_mode=args.fusion_mode)

        set_random_seed(args.seed)
        (
            train_data_loader,
            dev_data_loader,
            test_data_loader,
            num_train_optimization_steps,
        ) = set_up_data_loader()

        model, optimizer, scheduler = prep_for_training(num_train_optimization_steps)

        train(
            model,
            train_data_loader,
            dev_data_loader,
            test_data_loader,
            optimizer,
            scheduler,
            result_file,
        )

    print(f"Experiment completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
