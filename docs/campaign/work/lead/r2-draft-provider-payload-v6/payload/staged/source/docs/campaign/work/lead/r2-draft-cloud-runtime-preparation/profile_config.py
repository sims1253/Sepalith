"""Bounded DSpark profile. SEPALITH_* path variables are required at import.

Root binds a fresh output directory; this profile forbids implicit resume.
The external process deadline and artifact readback remain launch gates.
"""

import os

from warmstart_trainer import SepalithWarmstartTrainer


project_name = "sepalith-r2"
exp_name = "dspark_minicpm5_2b_train_only"
seed = 42

model = dict(
    allow_resume=False,
    public_draft_weights=os.environ["SEPALITH_PUBLIC_DRAFT_WEIGHTS"],
    target_model_name_or_path=os.environ["SEPALITH_MINICPM5_TARGET"],
    block_size=7,
    num_draft_layers=5,
    target_layer_ids=[1, 10, 20, 30, 39],
    # Match the released MiniCPM5-2B-DSpark geometry.  This is an internal
    # mask embedding ID, not a generated EOS; gate it before serving.
    mask_token_id=75982,
    num_anchors=32,
    markov_rank=256,
    markov_head_type="vanilla",
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=SepalithWarmstartTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    # A two-GPU smoke can override this to 2 or 4; keep the launch bounded.
    global_batch_size=2,
    num_train_epochs=1,
    max_train_steps=8,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=False,
)

logging = dict(
    logging_steps=1,
    checkpointing_steps=5,
)

data = dict(
    target_cache_path=os.environ["SEPALITH_DRAFT_CACHE"],
    # CacheDataset consumes pretokenized rows; this value is retained for the
    # upstream config schema and is not used to apply a chat template.
    chat_template="qwen",
    max_length=4096,
    num_workers=1,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    logging_cfg.setdefault("checkpoint_dir", os.path.join(
        os.environ["SEPALITH_DRAFT_OUTPUT_ROOT"], str(cfg["project_name"]), str(cfg["exp_name"])
    ))
    logging_cfg.setdefault("tensorboard_dir", os.path.join(
        os.environ["SEPALITH_DRAFT_LOG_ROOT"], str(cfg["project_name"]), str(cfg["exp_name"])
    ))
    cfg["logging"] = logging_cfg
    return cfg
