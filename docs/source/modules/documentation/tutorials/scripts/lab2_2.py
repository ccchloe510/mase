checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

from chop.tools import get_tokenized_dataset

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

import torch
import gc
import torch.nn as nn
from chop.nn.modules import Identity
import copy

search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choices": [
        nn.Linear,
        Identity,
    ],
}

from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools.utils import deepsetattr


def construct_model(trial):
    config = AutoConfig.from_pretrained(checkpoint)

    # Update the paramaters in the config
    for param in [
        "num_layers",
        "num_heads",
        "hidden_size",
        "intermediate_size",
    ]:
        chosen_idx = trial.suggest_int(param, 0, len(search_space[param]) - 1)
        setattr(config, param, search_space[param][chosen_idx])

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

from chop.tools import get_trainer

from chop.pipelines import CompressionPipeline
from chop import MaseGraph

quantization_config = {
    "by": "type",
    "default": {
        "config": {
            "name": None,
        }
    },
    "linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}

pruning_config = {
    "weight": {
        "sparsity": 0.5,
        "method": "l1-norm",
        "scope": "local",
    },
    "activation": {
        "sparsity": 0.5,
        "method": "l1-norm",
        "scope": "local",
    },
}

def objective(trial):
    current_quant_config = copy.deepcopy(quantization_config)
    current_prune_config = copy.deepcopy(pruning_config)
    print(f"\n [Trial {trial.number}] Started...")

    model = construct_model(trial)
    
    trainer_pre = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    trainer_pre.train() 
    
    pre_eval_results = trainer_pre.evaluate()
    acc_pre = pre_eval_results["eval_accuracy"]
    trial.set_user_attr("accuracy_pre_compress", acc_pre)
    print(f"Pre-compression Acc: {acc_pre:.4f}")

    del trainer_pre
    torch.cuda.empty_cache()

    # 2. Compress (Move to CPU)
    model_on_cpu = model.cpu()
    
    mg = MaseGraph(
        model_on_cpu,
        hf_input_names=["input_ids", "attention_mask", "labels"],
    )
    pipe = CompressionPipeline()

    mg, _ = pipe(
        mg,
        pass_args={
            "quantize_transform_pass": current_quant_config,
            "prune_transform_pass": current_prune_config,
        },
    )
    compressed_model = mg.model 

    trainer_eval = get_trainer(
        model=compressed_model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=0 
    )
    eval_results = trainer_eval.evaluate()
    acc_post = eval_results["eval_accuracy"]
    trial.set_user_attr("accuracy_compressed", acc_post)
    
    print(f"[Trial {trial.number}] Finished | Post-Acc: {acc_post:.4f} (Drop: {acc_pre - acc_post:.4f})")
    
    del model, model_on_cpu, trainer_eval, compressed_model, mg
    gc.collect()
    torch.cuda.empty_cache()

    return acc_pre

from optuna.samplers import GridSampler, RandomSampler, TPESampler

sampler = TPESampler()

import optuna

study = optuna.create_study(
    direction="maximize",
    study_name="bert-tiny-nas-study",
    sampler=sampler,
)

study.optimize(
    objective,
    n_trials=30,
    timeout=60 * 60 * 24,
)

df = study.trials_dataframe()
df.to_csv("results_non_compress-aware.csv", index=False)
print("Compress Only saved results_non_compress-aware.csv")

