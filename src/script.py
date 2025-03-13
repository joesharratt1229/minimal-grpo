import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import GenerationConfig, PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils import clip_grad_norm_

import os
import re
import wandb
from typing import Optional, Type, TypeVar

import reasoning_gym
from reasoning_gym.utils import SYSTEM_PROMPTS

from experience import Experience, ReplayBuffer
from loss import GRPOLoss


T = TypeVar('T', bound=PreTrainedModel)

def load_model_and_tokenizer(model_class: Type[T], model_name: str, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = model_class.from_pretrained(model_name, **kwargs)
    return model, tokenizer
    

def extract_answer(completion: str, tag_name: str = "answer", strip: bool = True) -> Optional[str]:
    regex = f"<{tag_name}>\\s?(.*?)\\s?</{tag_name}>"
    matches = list(
        re.finditer(
            regex,
            completion,
            flags=re.DOTALL,
        )
    )
    if not matches:
        return None
    answer = matches[-1].group(1)
    if strip:
        answer = answer.strip()
    return answer

class ReasoningGymDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 dataset_name,
                 size, 
                 seed,
                 developer_role = 'system',
                 developer_prompt = SYSTEM_PROMPTS['DeepSeekZero']):
        self.data = reasoning_gym.create_dataset(dataset_name, size=size, seed=seed)
        self.tokenizer = tokenizer
        self.developer_role = developer_role
        self.developer_prompt = developer_prompt
        
    def __len__(self):
        return self.data.size
    
    def __getitem__(self, index):
        chat_message = [
            {
                'role': self.developer_role,
                'content': self.developer_prompt
            }
        ]
        item = self.data[index]
        chat_message.append({'role': 'user', 'content': item['question']})
        prompt = self.tokenizer.apply_chat_template(chat_message, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(prompt, return_tensors='pt', padding = True, padding_size = 'left', return_attention_mask = True).cuda()
        model_inputs['input_ids'] = model_inputs['input_ids'].repeat(self.n_samples, 1)
        model_inputs['attention_mask'] = model_inputs['attention_mask'].repeat(self.n_samples, 1)
        return model_inputs, item
    
    

def compute_reward(dataset, completion: str, entry: dict, reward_weights: dict, tag_name: str = "answer") -> float:
    answer = extract_answer(completion, tag_name)
    correctness_reward = dataset.data.score_answer(answer, entry)
    regex = f"<{tag_name}>\\s?(.*?)\\s?</{tag_name}>"
    match = re.match(regex, completion, flags=re.DOTALL)
    formatted_reward = 1 if match else 0
    return (reward_weights['correctness'] * correctness_reward + reward_weights['formatted'] * formatted_reward)


@torch.no_grad()
def rollout(model, 
            data, 
            generation_config, 
            entry, 
            tokenizer, 
            model_config,
            dataset):
    sequence_ids = model.generate(data['input_ids'], data['attention_mask'], generation_config = generation_config)
    completions = tokenizer.batch_decode(sequence_ids[:, data['input_ids'].shape[1]:], skip_special_tokens=True)
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, data['input_ids'].shape[1]:] = True
    action_mask[sequence_ids == tokenizer.pad_token_id] = False
    action_mask = action_mask[:, 1:]
    
    
    total_rewards = torch.zeros(len(completions))
    for i, completion in enumerate(completions):
        total_rewards[i] = compute_reward(dataset, completion, entry, model_config['reward_weights'])
        
    return sequence_ids, completions, action_mask, total_rewards


def sequence_log_probs_with_logits(logits, output_ids):
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index = output_ids.unsqueeze(-1)).squeeze(-1)


def sequence_log_probs(model, sequence_ids, attention_mask):
    position_ids = attention_mask.long().cumsum(dim=-1) - 1 # gets position ids from attention mask
    position_ids.masked_fill(mask=(attention_mask == 0), value=1) # fills tokens not attended to with special token id
    outputs = model.forward(input_ids=sequence_ids, 
                            position_ids=position_ids, 
                            attention_mask=attention_mask, 
                            use_cache=False)
    logits = outputs.logits
    log_probs = sequence_log_probs_with_logits(logits[:, :-1].to(torch.float32), # last logit not needed
                                               sequence_ids[:, 1:])
    return log_probs


def compute_advantage(rewards, eps: float = 1e-8):
    return (rewards - rewards.mean()) / (rewards.std() + eps)


def main(config):
    reference_model, _ = load_model_and_tokenizer(AutoModelForCausalLM, 
                                                  'Qwen/Qwen2.5-Math-1.5B',
                                                  attn_implementation = 'flash_attention_2',
                                                  torch_dtype = torch.bfloat16, 
                                                  device_map = 'cuda:0')
    model, tokenizer = load_model_and_tokenizer(AutoModelForCausalLM, 
                                                "Qwen/Qwen2.5-Math-1.5B",
                                                attn_implemntation = 'flash_attention_2',
                                                torch_dtype = torch.bfloat16, 
                                                device_map = 'cuda:0')
    wandb.init(project = 'reasoning-gym-initial')
    
    
    dataset = ReasoningGymDataset(tokenizer, 
                                  config.dataset_name, 
                                  config.dataset_size, 
                                  seed = 42)
    
    dataset_loader = DataLoader(dataset, 
                                batch_size = config.num_rollouts, 
                                shuffle = True)
    
    optimizer = AdamW(model.parameters(), lr = config.learning_rate)
    objective = GRPOLoss(config.clip_ratio, config.kl_threshold)
    replay_buffer = ReplayBuffer()
    
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
            do_sample = True,
            temperature = config.temperature,
            max_length = config.max_completion_length,
            pad_token_id = pad_token_id,
            top_p = config.top_p)
    
    
    for input_data, entries in dataset_loader:
        replay_buffer.clear()
        rollout_returns = []
        
        for data, item in zip(input_data, entries):
            with torch.no_grad():
                sequence_ids, completions, action_mask, rewards = rollout(model, data, generation_config, tokenizer)
                advantanges = compute_advantage(rewards)
                probs = sequence_log_probs(model, sequence_ids, action_mask)
                ref_probs = sequence_log_probs(reference_model, sequence_ids, action_mask)
                
                attention_mask = sequence_ids.ne(pad_token_id)
                print(
                    f"question={item['question']}, answer={item['answer']}, rollout returns={rewards.sum().item():.2f}, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                )
                rollout_returns.append(rewards.cpu())
                
                exp = Experience(sequence_ids = sequence_ids, 
                                action_log_probs = probs, 
                                log_probs_ref = ref_probs, 
                                rewards = rewards, 
                                advantanges = advantanges, 
                                attention_mask = attention_mask, 
                                action_mask = action_mask)
                replay_buffer.append(exp)
            
        
        torch.cuda.empty_cache() 
        episode_return_sum = torch.stack(rollout_returns).sum()
        print(f"episode return sum: {episode_return_sum.item():.2f}")
        wandb.log({'episode_return_sum': episode_return_sum.item()})
        
        experience_sampler = DataLoader(replay_buffer,  # rollouts stored as a single experience object
                                        batch_size=config.mini_batch_size,
                                        shuffle=True)
        for i, _ in enumerate(range(config.epochs_per_step)): # number of iterations per batch
            model.train()
            optimizer.zero_grad()
            for batch_idx, exp in enumerate(experience_sampler):
                exp = exp.to(model.device)
                sequence_ids, completions, rewards, advantanges, old_probs,  kl_loss = exp
                log_probs = sequence_log_probs(model, sequence_ids)
                loss, kl_loss = objective(log_probs, old_probs, advantanges, kl_loss)
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
                
                print(f"loss: {loss.item():.2f}, kl_loss: {kl_loss.item():.2f}")
                wandb.log({'loss': loss.item(), 'kl_loss': kl_loss.item()})
                
                if (batch_idx+1) % config.gradient_accumulation_steps == 0:
                    grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    print(f"grad_norm: {grad_norm.item():.2f}`")
                    wandb.log({'grad_norm': grad_norm.item()})
                    optimizer.zero_grad()
                    
        if (i+1) % config.save_steps == 0:
            if not os.path.exists(config.checkpoint_dir):
                os.makedirs(config.checkpoint_dir)
            model.save_pretrained(f"{config.checkpoint_dir}/model_{i+1}")
            
        if (i+1) % config.eval_steps == 0:
            eval_reward = run_evaluation(model, dataset, generation_config, tokenizer)
            wandb.log({'eval_reward': eval_reward})

def run_evaluation(model, eval_data, eval_generation_config, tokenizer):
    eval_loader = DataLoader(eval_data, 
                            batch_size = 1, 
                            shuffle = True)
    model.eval()
    total_reward = 0
    with torch.no_grad():
        for data, item in eval_loader:
            sequence_ids = model.generate(data['input_ids'], data['attention_mask'], generation_config = eval_generation_config)
            completions = tokenizer.batch_decode(sequence_ids[:, data['input_ids'].shape[1]:], skip_special_tokens=True)
            reward = eval_data.data.score_answer(completions[0], item)
            print(f"question={item['question']}, expected answer={item['answer']}, model answer={completions[0]}, rollout returns={reward.item():.2f}")
            total_reward += reward
    return total_reward / len(eval_data)
            
            
            

