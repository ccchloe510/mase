import torch
import torch.nn as nn
import gc
import pandas as pd
import dill
import optuna
from pathlib import Path
from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools import get_tokenized_dataset, get_trainer
from chop.tools.utils import deepsetattr
from chop.nn.modules import Identity
from optuna.samplers import TPESampler

# --- 1. 基础配置 ---
checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

# --- 2. 搜索空间 ---
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choices": [nn.Linear, Identity],
}

# --- 3. 构造模型 (已修复 Config 映射) ---
def construct_model(trial):
    config = AutoConfig.from_pretrained(checkpoint)

    # 🛠️ 关键修复：参数名映射表
    # 将 Search Space 的名字 (Key) 映射到 BERT Config 的真实属性名 (Value)
    param_map = {
        "num_layers": "num_hidden_layers",
        "num_heads": "num_attention_heads"
    }

    # Update the paramaters in the config
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        chosen_idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        value = search_space[param][chosen_idx]
        
        # 1. 设置老师定义的原始名字 (保留原逻辑)
        setattr(config, param, value)
        
        # 2. ✅ 同步设置 BERT 需要的真实名字
        # 只有加了这一步，模型的层数和头数才会真的改变！
        if param in param_map:
            setattr(config, param_map[param], value)

    trial_model = AutoModelForSequenceClassification.from_config(config)

    for name, layer in trial_model.named_modules():
        if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
            new_layer_cls = trial.suggest_categorical(
                f"{name}_type",
                search_space["linear_layer_choices"],
            )

            if new_layer_cls == nn.Linear:
                continue
            elif new_layer_cls == Identity:
                new_layer = Identity()
                deepsetattr(trial_model, name, new_layer)
            else:
                raise ValueError(f"Unknown layer type: {new_layer_cls}")

    return trial_model

# --- 4. Objective 函数 ---
def objective(trial):
    print(f"\n🚀 [Trial {trial.number}] Started...")
    
    # Define the model
    model = construct_model(trial)

    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    acc = eval_results["eval_accuracy"]
    
    print(f"✅ [Trial {trial.number}] Finished | Acc: {acc:.4f}")

    # Set the model as an attribute so we can fetch it later
    # 注意：在 Baseline 中存 model 对象通常没问题，因为没经过 chop 压缩
    # 但为了保险防止内存泄漏，建议只在最后存 best model，或者只存 state_dict
    # 这里遵照原代码逻辑保留，但要注意内存
    model_cpu = model.cpu()
    trial.set_user_attr("model", model_cpu)

    # 🧹 简单的显存清理，防止大模型爆显存
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return acc

# --- 5. 运行 Study ---
sampler = TPESampler()

study = optuna.create_study(
    direction="maximize",
    study_name="bert-tiny-nas-fixed-config", # 改个名字区分
    sampler=sampler,
)

study.optimize(
    objective,
    n_trials=30, # 跑30次看效果
    timeout=60 * 60 * 24,
)

# --- 6. 保存结果 ---
df = study.trials_dataframe()

# 💾 保存为新的 CSV 文件
csv_filename = "results_baseline_fixed_config.csv"
df.to_csv(csv_filename, index=False)
print(f"🎉 Baseline (Fixed Config) results saved to {csv_filename}")

# 保存最佳模型
try:
    best_model = study.best_trial.user_attrs["model"]
    with open(f"{Path.home()}/tutorial_5_best_model_fixed.pkl", "wb") as f:
        dill.dump(best_model, f)
    print("Best model saved.")
except Exception as e:
    print(f"⚠️ Could not save model with dill: {e}")