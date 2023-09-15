from rouge_score import rouge_scorer
from trlx.models.modeling_ppo import PPOConfig
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
import trlx
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
from typing import List
print("hello")


try:
    import evaluate
except ImportError:
    raise ImportError(
        "To run this example, please install the `evaluate` and `nltk` packages" "by running `pip install evaluate`"
    )

print("hi")

config = TRLConfig(
    train=TrainConfig(
        seq_length=612,
        epochs=3,
        total_steps=100000,
        batch_size=1,
        checkpoint_interval=10000,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
    ),
    model=ModelConfig(
        model_path="./mymodel",
        model_arch_type="seq2seq",
        num_layers_unfrozen=2,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="google/mt5-base",
        truncation_side="left",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 1.0e-5,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 10000,
            "eta_min": 1.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=512,
        chunk_size=12,
        ppo_epochs=4,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=0.99,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1.0,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 100,
        },
        gen_experience_kwargs={
            "max_new_tokens": 100,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
        },
    ),
)


meteor = evaluate.load("meteor")
scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True, tokenizer=tokenizer)

if __name__ == "__main__":

    def reward_fn_meteor(samples: List[str], prompts: List[str], outputs: List[str]):
        original_summaries = [prompt_label[prompt.strip()]
                              for prompt in prompts]
        scores = [
            meteor.compute(predictions=[output.strip()], references=[
                           original])["meteor"]
            for (original, output) in zip(original_summaries, outputs)
        ]
        return scores

    def reward_fn_rouge(samples: List[str], prompts: List[str], outputs: List[str]):

        original_summaries = [summaries[i] for i in range(len(prompts))]

        scores = [
            scorer.score(original.strip(), output.strip())["rougeL"].fmeasure
            for (original, output) in zip(original_summaries, outputs)
        ]
        return scores

    dataset = load_dataset("pn_summary")

    # take 20,000 samples from the training set as prompts for training
    prompts = dataset["train"]["article"][0:20000]
    summaries = dataset["train"]["summary"][0:20000]
    prompts = ["Summarize: " + prompt for prompt in prompts]

    # take 1,000 samples from the validation set as prompts for evaluation
    val_prompts = ["Summarize: " +
                   prompt for prompt in dataset["validation"]["article"][0:1000]]
    val_summaries = dataset["validation"]["summary"][0:1000]

    # make dictionary of prompts and labels to use for reward function
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    tokenizer.sep_token = "<sep>"
    prompt_label = {}
    max_length = config.train.seq_length - \
        config.method.gen_kwargs["max_new_tokens"]

    for i in tqdm(range(len(prompts))):
        key = tokenizer.decode(
            tokenizer(prompts[i], truncation=True, max_length=max_length,
                      add_special_tokens=False)["input_ids"],
            skip_special_tokens=True,
        )  # get prompt like trlx's prompt
        prompt_label[key.strip()] = summaries[i]

    for i in tqdm(range(len(val_prompts))):
        key = tokenizer.decode(
            tokenizer(val_prompts[i], truncation=True, max_length=max_length,
                      add_special_tokens=False)["input_ids"],
            skip_special_tokens=True,
        )  # get prompt like trlx's prompt
        prompt_label[key.strip()] = val_summaries[i]

    trlx.train(
        reward_fn=reward_fn_rouge,
        prompts=prompts,
        eval_prompts=val_prompts,
        config=config,
    )
