from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    attn_implementation: str = field(default="flash_attention_2")
    torch_dtype: str = field(default="bfloat16")
    device_map: str = field(default="cuda:0")
    max_length: int = field(default=1024)
    top_p: float = field(default=1)
    temperature: float = field(default=0.7)
    max_completion_length: int = field(default=1024)
    max_prompt_length: int = field(default=512)
    
@dataclass
class TrainingArguments:
    # ToDo change to more general reasoning model
    
    # General training hyperparameters
    epochs_per_step: int = field(default=2)
    batch_size: int = field(default=16)
    gradient_accumulation_steps: int = field(default=2)
    learning_rate: float = field(default=5e-6)
    mini_batch_size: int = field(default=1)
    # GRPO hyperparameters
    clip_ratio: float = field(default=0.2)
    kl_threshold: float = field(default=0.01)
    
    
@dataclass
class ScriptArguments:
    model_name: str = field(default="Qwen/Qwen2.5-Math-1.5B")
    dataset_name: str = field(default='chain_sum')
    dataset_size: int = field(default=1000)
    save_steps: int = field(default=50)
    checkpoint_dir: str = field(default='models')
    
    
    
    
