# 所以MS2BioText的输入应该是包括MS2与BioText。然而一个BioText对应一个分子，一个分子对应多个MS2。所以在dataset构建的时候，需要储存：
# MS2的列表
# MS2对应的分子的列表
# 分子对应的BioText的列表
# 最后item的时候返回input_ids（m/z），intensity，BioText
# （这里添加一系列类内方法，在init的时候输入参数可以选择处理BioText的类内方法）
# 但是问题是但是在之后的实验里，还要记录对于每个MS2的其他信息，这个怎么储存。


# 先完成dataset然后创建实例
# 读取HMDB数据集
# HMDB.h5是MS2数据读取为list？，HMDB.parquet是meta数据，读取为？然后Biotext是一个文件夹下包含的一系列txt文件，文件名为HMDB的id，读取为？
# ，读取test data跑通两个模型试试

import pickle
import h5py
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from torch.utils.data import Sampler
import random, itertools
import math
from collections.abc import Iterator
from typing import Optional, TypeVar, Dict, List, Tuple
import random
import argparse
import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler


_T_co = TypeVar("_T_co", covariant=True)

class MS2MoleculeDistributedSampler(Sampler[_T_co]):
    """
    专为MS2对比学习设计的分布式采样器
    
    新增功能：为每个batch中的样本分配不冲突的text
    - 确保batch内每个molecule选择的text不出现在其他molecule的候选列表中
    - 保证batch内只有对角线是正样本
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
            
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        # 构建molecule -> texts的映射
        self.molecule_to_texts = self._build_molecule_text_mapping()
        
        # 按molecule分组并按MS2数量排序
        self.molecule_groups = self._group_by_molecule()
        self.sorted_molecules = self._sort_molecules_by_ms2_count()
        
        # 生成batch分配方案
        self.batch_indices = self._create_batch_allocation()
        
        # 确保能被GPU数整除
        self._adjust_for_distributed()
        
        # 计算每个进程的样本数
        total_batches = len(self.batch_indices)
        batches_per_replica = total_batches // self.num_replicas
        self.num_samples = batches_per_replica * self.batch_size
        
    def _build_molecule_text_mapping(self) -> Dict[str, List[str]]:
        """构建每个molecule的候选text列表"""
        molecule_to_texts = {}
        
        for mol_id, entry in self.dataset.biotext_data.items():
            texts = []
            if isinstance(entry, list):
                texts = [record['text'] for record in entry]
            elif isinstance(entry, dict):
                original = entry.get("original", "")
                paraphrases = entry.get("paraphrases", [])
                texts = [original] + paraphrases
            elif isinstance(entry, str):
                texts = [entry]
            
            molecule_to_texts[mol_id] = texts
        
        print(f"Built text mapping for {len(molecule_to_texts)} molecules")
        return molecule_to_texts
    
    def _group_by_molecule(self) -> Dict[str, List[int]]:
        """按molecule ID对MS2数据进行分组"""
        molecule_groups = {}
        for idx in range(len(self.dataset)):
            ms2_id = self.dataset.ms2_ids[idx]
            molecule_id = self.dataset.preprocessed_ms2_tensors[ms2_id]['molecule_id']
            
            if molecule_id not in molecule_groups:
                molecule_groups[molecule_id] = []
            molecule_groups[molecule_id].append(idx)
        
        return molecule_groups
    
    def _sort_molecules_by_ms2_count(self) -> List[Tuple[str, List[int]]]:
        """按每个molecule拥有的MS2数量从多到少排序"""
        molecule_items = [(mol_id, indices) for mol_id, indices in self.molecule_groups.items()]
        sorted_items = sorted(molecule_items, key=lambda x: len(x[1]), reverse=True)
        
        print(f"Molecule MS2 count distribution:")
        print(f"Max MS2 per molecule: {len(sorted_items[0][1])}")
        print(f"Min MS2 per molecule: {len(sorted_items[-1][1])}")
        print(f"Total molecules: {len(sorted_items)}")
        print(f"Total MS2 spectra: {sum(len(indices) for _, indices in sorted_items)}")
        
        return sorted_items
    
    def _assign_texts_for_batch(self, batch_molecule_ids: List[str]) -> Dict[str, int]:
        """
        为batch中的每个molecule分配一个text index
        确保选中的text不在其他molecule的候选列表中
        
        返回: {molecule_id: text_index}
        """
        mol_to_text_idx = {}
        occupied_texts = set()  # 已被占用的text
        
        # 按照候选text数量从少到多排序，优先处理选择空间小的molecule
        sorted_mols = sorted(
            batch_molecule_ids,
            key=lambda mol_id: len(self.molecule_to_texts.get(mol_id, []))
        )
        
        for mol_id in sorted_mols:
            candidate_texts = self.molecule_to_texts.get(mol_id, [])
            
            if not candidate_texts:
                print(f"⚠️ Warning: Molecule {mol_id} has no candidate texts")
                mol_to_text_idx[mol_id] = 0
                continue
            
            # 找到所有未被占用且不在其他molecule候选中的text
            available_indices = []
            for i, text in enumerate(candidate_texts):
                if text not in occupied_texts:
                    # 检查这个text是否在其他molecule的候选中
                    text_in_others = False
                    for other_mol_id in batch_molecule_ids:
                        if other_mol_id != mol_id:
                            other_texts = self.molecule_to_texts.get(other_mol_id, [])
                            if text in other_texts:
                                text_in_others = True
                                break
                    
                    if not text_in_others:
                        available_indices.append(i)
            
            # 如果有可用的text，随机选一个
            if available_indices:
                chosen_idx = random.choice(available_indices)
                mol_to_text_idx[mol_id] = chosen_idx
                occupied_texts.add(candidate_texts[chosen_idx])
            else:
                # Fallback: 随机选一个未被占用的（可能在其他molecule的候选中）
                fallback_indices = [i for i, text in enumerate(candidate_texts) 
                                  if text not in occupied_texts]
                if fallback_indices:
                    chosen_idx = random.choice(fallback_indices)
                    mol_to_text_idx[mol_id] = chosen_idx
                    occupied_texts.add(candidate_texts[chosen_idx])
                    print(f"⚠️ Fallback: Molecule {mol_id} text may conflict with others")
                else:
                    # 极端情况：所有text都被占用了
                    chosen_idx = random.randint(0, len(candidate_texts) - 1)
                    mol_to_text_idx[mol_id] = chosen_idx
                    print(f"⚠️ Extreme fallback: All texts occupied for {mol_id}")
        
        return mol_to_text_idx
    
    def _create_batch_allocation(self) -> List[List[int]]:
        """创建batch分配方案"""
        molecule_ms2_usage = {}
        for mol_id, indices in self.sorted_molecules:
            molecule_ms2_usage[mol_id] = {
                'indices': indices.copy(),
                'used_count': 0
            }
        
        batch_indices = []
        total_ms2_count = sum(len(indices) for _, indices in self.sorted_molecules)
        used_ms2_count = 0
        
        print(f"Starting batch allocation for {total_ms2_count} MS2 spectra...")
        
        while used_ms2_count < total_ms2_count:
            current_batch = []
            used_molecules_in_batch = set()
            
            for mol_id, mol_data in molecule_ms2_usage.items():
                if len(current_batch) >= self.batch_size:
                    break
                    
                if mol_id in used_molecules_in_batch:
                    continue
                
                if mol_data['used_count'] < len(mol_data['indices']):
                    ms2_idx = mol_data['indices'][mol_data['used_count']]
                    current_batch.append(ms2_idx)
                    used_molecules_in_batch.add(mol_id)
                    mol_data['used_count'] += 1
                    used_ms2_count += 1
            
            if len(current_batch) < self.batch_size and len(current_batch) > 0:
                available_molecules = [mol_id for mol_id in molecule_ms2_usage.keys() 
                                     if mol_id not in used_molecules_in_batch]
                
                while len(current_batch) < self.batch_size and available_molecules:
                    available_molecules.sort(key=lambda x: len(molecule_ms2_usage[x]['indices']), 
                                           reverse=True)
                    
                    mol_id = available_molecules[0]
                    mol_data = molecule_ms2_usage[mol_id]
                    
                    ms2_idx = random.choice(mol_data['indices'])
                    current_batch.append(ms2_idx)
                    used_molecules_in_batch.add(mol_id)
                    available_molecules.remove(mol_id)
            
            if len(current_batch) == 0:
                break
                
            if len(current_batch) < self.batch_size:
                if self.drop_last:
                    print(f"Dropping incomplete batch with {len(current_batch)} samples")
                    break
                else:
                    while len(current_batch) < self.batch_size:
                        available_molecules = [mol_id for mol_id in molecule_ms2_usage.keys() 
                                             if mol_id not in used_molecules_in_batch]
                        
                        if not available_molecules:
                            print(f"Cannot fill batch further: only {len(self.sorted_molecules)} unique molecules available")
                            break
                        
                        mol_id = random.choice(available_molecules)
                        mol_data = molecule_ms2_usage[mol_id]
                        
                        ms2_idx = random.choice(mol_data['indices'])
                        current_batch.append(ms2_idx)
                        used_molecules_in_batch.add(mol_id)
            
            batch_indices.append(current_batch)
        
        print(f"Created {len(batch_indices)} batches")
        print(f"Used {used_ms2_count} MS2 spectra out of {total_ms2_count}")
        
        return batch_indices
    
    def _adjust_for_distributed(self):
        """调整batch数量以确保能被GPU数整除"""
        total_batches = len(self.batch_indices)
        remainder = total_batches % self.num_replicas
        
        if remainder != 0:
            if self.drop_last:
                batches_to_remove = remainder
                self.batch_indices = self.batch_indices[:-batches_to_remove]
                print(f"Dropped {batches_to_remove} batches to ensure divisibility by {self.num_replicas} GPUs")
            else:
                batches_to_add = self.num_replicas - remainder
                for i in range(batches_to_add):
                    batch_to_copy = self.batch_indices[i % len(self.batch_indices)]
                    self.batch_indices.append(batch_to_copy.copy())
                print(f"Added {batches_to_add} batches to ensure divisibility by {self.num_replicas} GPUs")
        
        final_batches = len(self.batch_indices)
        print(f"Final batch count: {final_batches} (divisible by {self.num_replicas} GPUs)")
        print(f"Each GPU will process {final_batches // self.num_replicas} batches")
    
    def __iter__(self) -> Iterator[_T_co]:
        # 获取当前进程应该处理的batch
        total_batches = len(self.batch_indices)
        batches_per_replica = total_batches // self.num_replicas
        
        start_batch = self.rank * batches_per_replica
        end_batch = start_batch + batches_per_replica
        
        my_batches = self.batch_indices[start_batch:end_batch]
        
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            batch_order = torch.randperm(len(my_batches), generator=g).tolist()
            my_batches = [my_batches[i] for i in batch_order]
            
            for batch in my_batches:
                random.Random(self.seed + self.epoch).shuffle(batch)
        
        # *** 关键：为每个batch分配text ***
        self.dataset.text_assignment.clear()
        
        for batch_indices in my_batches:
            # 获取batch中所有molecule ID
            batch_molecule_ids = []
            ms2_id_to_mol_id = {}
            
            for idx in batch_indices:
                ms2_id = self.dataset.ms2_ids[idx]
                mol_id = self.dataset.preprocessed_ms2_tensors[ms2_id]['molecule_id']
                batch_molecule_ids.append(mol_id)
                ms2_id_to_mol_id[ms2_id] = mol_id
            
            # 为这个batch分配text indices
            mol_to_text_idx = self._assign_texts_for_batch(batch_molecule_ids)
            
            # 将分配结果写入dataset.text_assignment
            for idx in batch_indices:
                ms2_id = self.dataset.ms2_ids[idx]
                mol_id = ms2_id_to_mol_id[ms2_id]
                self.dataset.text_assignment[ms2_id] = mol_to_text_idx.get(mol_id, 0)
        
        # 展平所有batch的indices
        all_indices = []
        for batch in my_batches:
            all_indices.extend(batch)
        
        return iter(all_indices)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        """设置当前epoch，用于确保每个epoch的shuffle结果不同"""
        self.epoch = epoch


# function for dataset augmentation
import torch
import numpy as np
import random
import torch
import numpy as np
import random


def sample_truncated_normal(mean, std, low, high):
    """
    从截断正态分布中采样
    
    Args:
        mean: 均值
        std: 标准差
        low: 下界
        high: 上界
    
    Returns:
        采样值
    """
    max_attempts = 1000
    for _ in range(max_attempts):
        sample = np.random.normal(mean, std)
        if low <= sample <= high:
            return sample
    # 如果1000次都没采样到，返回截断后的值
    return np.clip(np.random.normal(mean, std), low, high)


def augment_tokenized_ms2_optimized(mz_tokens, intensity, word2idx, args):
    """
    基于External数据特征优化的动态augmentation（支持随机noise ratio）
    """
    # ===== 0. 读取参数并决定是否augment =====
    augment_prob = getattr(args, 'augment_prob', 0.5)
    if random.random() > augment_prob:
        return mz_tokens, intensity
    
    # 确保intensity是1D
    if intensity.dim() == 2:
        intensity = intensity.squeeze(0)
    
    device = mz_tokens.device
    
    # ===== 1. 构建token_id到m/z的映射 =====
    idx2word = {v: k for k, v in word2idx.items()}
    
    def token_to_mz(token_id):
        """将token id转换为实际m/z值"""
        word = idx2word.get(token_id.item(), None)
        if word and word not in ['[PAD]', '[MASK]']:
            try:
                return float(word)
            except ValueError:
                return None
        return None
    
    # 转换为numpy进行计算
    mz_tokens_np = mz_tokens.cpu().numpy()
    intensity_np = intensity.cpu().numpy()
    
    # 获取实际的m/z值（跳过特殊token）
    original_mz = []
    original_intensity = []
    for i, token_id in enumerate(mz_tokens_np):
        mz_val = token_to_mz(torch.tensor(token_id))
        if mz_val is not None:
            original_mz.append(mz_val)
            original_intensity.append(intensity_np[i])
    
    if len(original_mz) == 0:
        return mz_tokens, intensity
    
    original_mz = np.array(original_mz)
    original_intensity = np.array(original_intensity)
    max_intensity = original_intensity.max()
    
    # ===== 2. 识别signal peaks（强度>5%的peaks） =====
    signal_threshold = 0.05
    signal_mask = original_intensity >= signal_threshold * max_intensity
    signal_mz = original_mz[signal_mask]
    n_signal = len(signal_mz)
    
    if n_signal == 0:
        return mz_tokens, intensity
    
    # ===== 3. 🔥 随机化参数 =====
    align_to_external = getattr(args, 'align_to_external', False)
    randomize_noise_ratio = getattr(args, 'randomize_noise_ratio', True)  # 🔥 新增开关
    noise_sampling_strategy = getattr(args, 'noise_sampling_strategy', 'uniform')  # 🔥 选择策略
    
    if align_to_external:
        # 🔥 随机化noise ratio
        if randomize_noise_ratio:
            if noise_sampling_strategy == 'uniform':
                # 策略1：均匀分布
                noise_ratio_range = getattr(args, 'noise_ratio_range', [0.60, 0.90])
                TARGET_NOISE_RATIO = np.random.uniform(noise_ratio_range[0], noise_ratio_range[1])
                
            elif noise_sampling_strategy == 'normal':
                # 策略2：正态分布
                target_noise_ratio = getattr(args, 'target_noise_ratio', 0.80)
                noise_ratio_std = getattr(args, 'noise_ratio_std', 0.10)
                TARGET_NOISE_RATIO = np.random.normal(target_noise_ratio, noise_ratio_std)
                TARGET_NOISE_RATIO = np.clip(TARGET_NOISE_RATIO, 0.40, 0.95)
                
            elif noise_sampling_strategy == 'bimodal':
                # 策略3：双模态分布
                bimodal_dirty_prob = getattr(args, 'bimodal_dirty_prob', 0.7)
                if np.random.random() < bimodal_dirty_prob:
                    # 脏数据模式
                    TARGET_NOISE_RATIO = np.random.uniform(0.70, 0.90)
                else:
                    # 干净数据模式
                    TARGET_NOISE_RATIO = np.random.uniform(0.30, 0.60)
            else:
                # 默认使用固定值
                TARGET_NOISE_RATIO = getattr(args, 'target_noise_ratio', 0.80)
        else:
            # 不随机化，使用固定值
            TARGET_NOISE_RATIO = getattr(args, 'target_noise_ratio', 0.80)
        
        # 🔥 可选：随机化proximal ratio
        randomize_proximal_ratio = getattr(args, 'randomize_proximal_ratio', False)
        if randomize_proximal_ratio:
            proximal_ratio_range = getattr(args, 'proximal_ratio_range', [0.15, 0.22])
            PROXIMAL_RATIO_OF_NOISE = np.random.uniform(proximal_ratio_range[0], proximal_ratio_range[1])
        else:
            PROXIMAL_RATIO_OF_NOISE = getattr(args, 'proximal_ratio_of_noise', 0.18)
        
        # Intensity参数
        PROXIMAL_MEAN = getattr(args, 'proximal_intensity_mean', 0.0115)
        PROXIMAL_STD = getattr(args, 'proximal_intensity_std', 0.0138)
        ISOLATED_MEAN = getattr(args, 'isolated_intensity_mean', 0.0079)
        ISOLATED_STD = getattr(args, 'isolated_intensity_std', 0.0114)
        
        # 区域权重
        use_regional_weighting = getattr(args, 'use_regional_weighting', True)
        if use_regional_weighting:
            REGION_WEIGHTS = [
                (0, 100, 0.24),
                (100, 200, 0.53),
                (200, 300, 0.18),
                (300, 500, 0.05)
            ]
        else:
            REGION_WEIGHTS = None
    else:
        # 不对齐External时的参数
        if randomize_noise_ratio:
            noise_ratio_range = getattr(args, 'noise_ratio_range', [0.30, 0.70])
            TARGET_NOISE_RATIO = np.random.uniform(noise_ratio_range[0], noise_ratio_range[1])
        else:
            TARGET_NOISE_RATIO = getattr(args, 'target_noise_ratio', 0.50)
        
        PROXIMAL_RATIO_OF_NOISE = 0.25
        PROXIMAL_MEAN = 0.0141
        PROXIMAL_STD = 0.0209
        ISOLATED_MEAN = 0.0091
        ISOLATED_STD = 0.0144
        REGION_WEIGHTS = None
    
    # 空间分布参数
    proximal_distance_range = getattr(args, 'proximal_distance_range', [-1.5, 1.5])
    isolated_min_distance = getattr(args, 'isolated_min_distance', 5.0)
    
    # ===== 4. 计算需要添加的noise总数 =====
    n_noise_total = int(n_signal * TARGET_NOISE_RATIO / (1 - TARGET_NOISE_RATIO))
    n_proximal = int(n_noise_total * PROXIMAL_RATIO_OF_NOISE)
    n_isolated = n_noise_total - n_proximal
    
    # ... 后面的代码保持不变 ...
    
    mz_min, mz_max = original_mz.min(), original_mz.max()
    
    # ===== 5. 生成Proximal Noise =====
    proximal_mz = []
    proximal_intensity = []
    
    for _ in range(n_proximal):
        # 选择一个signal peak作为base
        base_mz = np.random.choice(signal_mz)
        
        # Proximal距离范围
        offset = np.random.uniform(proximal_distance_range[0], proximal_distance_range[1])
        noise_mz = base_mz + offset
        noise_mz = np.clip(noise_mz, mz_min, mz_max)
        
        # 采样intensity（截断正态分布）
        noise_intensity = sample_truncated_normal(
            PROXIMAL_MEAN, PROXIMAL_STD, 0.001, 0.10
        ) * max_intensity
        
        proximal_mz.append(noise_mz)
        proximal_intensity.append(noise_intensity)
    
    # ===== 6. 生成Isolated Noise（区域加权） =====
    isolated_mz = []
    isolated_intensity = []
    
    if REGION_WEIGHTS:
        # 使用区域加权策略
        for low, high, weight in REGION_WEIGHTS:
            # 只在光谱m/z范围内生成
            region_low = max(low, mz_min)
            region_high = min(high, mz_max)
            
            if region_low >= region_high:
                continue
            
            n_in_region = int(n_isolated * weight)
            attempts = 0
            max_attempts = n_in_region * 10
            generated = 0
            
            while generated < n_in_region and attempts < max_attempts:
                candidate_mz = np.random.uniform(region_low, region_high)
                
                # 检查是否远离所有signal peaks
                min_dist = np.min(np.abs(signal_mz - candidate_mz))
                
                if min_dist >= isolated_min_distance:
                    # 100-200 Da区域intensity稍高
                    if 100 <= candidate_mz <= 200:
                        mean_adj = ISOLATED_MEAN * 1.1
                    else:
                        mean_adj = ISOLATED_MEAN
                    
                    noise_intensity = sample_truncated_normal(
                        mean_adj, ISOLATED_STD, 0.0001, 0.05
                    ) * max_intensity
                    
                    isolated_mz.append(candidate_mz)
                    isolated_intensity.append(noise_intensity)
                    generated += 1
                
                attempts += 1
    else:
        # 不使用区域加权（原始随机策略）
        attempts = 0
        max_attempts = n_isolated * 10
        generated = 0
        
        while generated < n_isolated and attempts < max_attempts:
            candidate_mz = np.random.uniform(mz_min, mz_max)
            min_dist = np.min(np.abs(signal_mz - candidate_mz))
            
            if min_dist >= isolated_min_distance:
                noise_intensity = sample_truncated_normal(
                    ISOLATED_MEAN, ISOLATED_STD, 0.0001, 0.05
                ) * max_intensity
                
                isolated_mz.append(candidate_mz)
                isolated_intensity.append(noise_intensity)
                generated += 1
            
            attempts += 1
    
    # ===== 7. 合并所有peaks =====
    all_mz = np.concatenate([original_mz, proximal_mz, isolated_mz])
    all_intensity = np.concatenate([original_intensity, proximal_intensity, isolated_intensity])
    
    # ===== 8. 转换回token ids =====
    def mz_to_token(mz_val):
        """将m/z值转换为token id（四舍五入到0.01精度）"""
        mz_rounded = round(mz_val, 2)
        mz_str = f"{mz_rounded:.2f}"
        return word2idx.get(mz_str, word2idx.get('[MASK]', 1))
    
    all_tokens = [mz_to_token(mz) for mz in all_mz]
    
    # 按m/z排序
    sorted_idx = np.argsort(all_mz)
    all_tokens = np.array(all_tokens)[sorted_idx]
    all_intensity = all_intensity[sorted_idx]
    
    # ===== 9. 可选：过滤和截断 =====
    filter_threshold = getattr(args, 'filter_threshold', None)
    if filter_threshold and filter_threshold > 0:
        max_int = all_intensity.max()
        if max_int > 0:
            threshold = max_int * filter_threshold
            mask = all_intensity >= threshold
            all_tokens = all_tokens[mask]
            all_intensity = all_intensity[mask]
    
    # 截断到maxlen
    maxlen = getattr(args, 'maxlen', 100)
    if len(all_tokens) > maxlen:
        # 保留最强的peaks
        top_indices = np.argsort(all_intensity)[-maxlen:]
        top_indices = np.sort(top_indices)  # 恢复m/z顺序
        all_tokens = all_tokens[top_indices]
        all_intensity = all_intensity[top_indices]
    
    # ===== 10. 转换回tensor =====
    augmented_mz = torch.tensor(all_tokens, dtype=mz_tokens.dtype, device=device)
    augmented_intensity = torch.tensor(all_intensity, dtype=intensity.dtype, device=device).unsqueeze(0)
    
    return augmented_mz, augmented_intensity




class MS2BioTextDataset(Dataset):
    def __init__(self, ms2_data, meta_data, biotext_data, tokenizer, max_length=512, 
                 use_paraphrase=False, use_mlm=False, use_ms2_prediction=False, 
                 prediction_label_columns=None, word2idx=None, args=None, split='test'):
        """
        hard_neg_path: path to hard_negatives_v2.json
        num_hard_neg_per_sample: number of hard negatives to use per sample
        """
        self.ms2_data = ms2_data
        self.meta_data = meta_data
        self.biotext_data = biotext_data
        self.preprocessed_ms2_tensors = {}
        
        for ms2_id, ms2_entry in self.ms2_data.items():
            self.preprocessed_ms2_tensors[ms2_id] = {
                'mz': torch.tensor(ms2_entry['mz'], dtype=torch.float32),
                'intensity': torch.tensor(ms2_entry['intensity'], dtype=torch.float32),
                'molecule_id': ms2_entry['molecule_id']
            }
        self.text_assignment = {}
        self.ms2_ids = list(ms2_data.keys())
        self.word2idx = word2idx
        self.args = args or argparse.Namespace()  # 确保非None
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_mlm = use_mlm
        self.use_ms2_prediction = use_ms2_prediction
        self.prediction_label_columns = prediction_label_columns
        self.use_paraphrase = use_paraphrase
        
        # === NEW: Load hard negatives ===
        self.hard_negatives = {}
        self.num_hard_neg = getattr(self.args, "num_hard_neg_per_sample", 0)
        hard_neg_path = getattr(self.args, "hard_neg_path", None)

        if hard_neg_path and os.path.exists(hard_neg_path) and split == 'train':
            import json
            with open(hard_neg_path, 'r', encoding='utf-8') as f:
                self.hard_negatives = json.load(f) 
            print(f"Loaded hard negatives for {len(self.hard_negatives)} molecules")
            print(f"Using {self.num_hard_neg} hard negatives per sample")
        else:
            if split == 'train':
                print(f"No valid hard_neg_path found ({hard_neg_path}), skipping hard negatives.")

        # === MS2 prediction checks ===
        if self.use_ms2_prediction:
            if not self.prediction_label_columns or not isinstance(self.prediction_label_columns, list):
                raise ValueError("When MS2 prediction task is enabled, a list of column names 'prediction_label_columns' must be provided.")
            
            for col in self.prediction_label_columns:
                if col not in self.meta_data.columns:
                    raise ValueError(f"Column '{col}' is not found in meta_data.")
            
            self.num_ms2_classes = len(self.prediction_label_columns)
            print(f"Found {self.num_ms2_classes} label columns for multilabel prediction task: {self.prediction_label_columns}")


    def __len__(self):
        return len(self.ms2_ids)
    
    def _create_mlm_inputs(self, input_ids):
        """为MLM任务创建掩码输入和标签"""
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, 0.15) # 15%的概率进行mask
        
        # 避免mask特殊tokens (e.g., [CLS], [SEP], [PAD])
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 将未被mask的token的label设置为-100，这样在计算loss时会被忽略
        labels[~masked_indices] = -100
        
        # 80% 的概率用 [MASK] token 替换
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% 的概率用随机token替换
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer.vocab), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        return input_ids, labels

    def __getitem__(self, idx):
        ms2_id = self.ms2_ids[idx]
        tensor_data = self.preprocessed_ms2_tensors[ms2_id]
        mz = tensor_data['mz']
        intensity = tensor_data['intensity']
        
        # Dynamic augmentation
        if self.split == 'train' and hasattr(self, 'word2idx'):
            mz, intensity = augment_tokenized_ms2_optimized(
                mz, intensity, self.word2idx, self.args
            )
        
        batch = {
            'mz': tensor_data['mz'],
            'intensity': tensor_data['intensity'].unsqueeze(0)
        }
        
        # === 统一的BioText处理逻辑 ===
        molecule_id = tensor_data['molecule_id']
        biotext = ""
        paraphrase_text = None
        all_candidate_texts = []

        if molecule_id in self.biotext_data:
            entry = self.biotext_data[molecule_id]
            
            if isinstance(entry, list):
                all_candidate_texts = [record['text'] for record in entry]
                
                # *** 关键修改：使用sampler分配的text index ***
                if ms2_id in self.text_assignment:
                    text_idx = self.text_assignment[ms2_id]
                    biotext = all_candidate_texts[text_idx]
                else:
                    # fallback: 随机选择（shouldn't happen in training）
                    biotext = random.choice(entry)['text']
                
                # Paraphrase: 如果需要，从剩余的候选中选一个不同的
                if self.use_paraphrase and len(entry) >= 2:
                    remaining_indices = [i for i in range(len(all_candidate_texts)) 
                                    if i != text_idx]
                    if remaining_indices:
                        para_idx = random.choice(remaining_indices)
                        paraphrase_text = all_candidate_texts[para_idx]
                        
            elif isinstance(entry, dict):
                original = entry.get("original", "")
                paraphrases = entry.get("paraphrases", [])
                all_candidates = [original] + paraphrases
                all_candidate_texts = all_candidates
                
                # *** 同样的逻辑 ***
                if ms2_id in self.text_assignment:
                    text_idx = self.text_assignment[ms2_id]
                    biotext = all_candidates[text_idx]
                else:
                    biotext = original
                
                if self.use_paraphrase and len(all_candidates) >= 2:
                    remaining = [i for i in range(len(all_candidates)) if i != text_idx]
                    if remaining:
                        para_idx = random.choice(remaining)
                        paraphrase_text = all_candidates[para_idx]
                        
            elif isinstance(entry, str):
                biotext = entry
                all_candidate_texts = [entry]
        else:
            print(f"⚠️ Warning: Molecule ID '{molecule_id}' missing BioText")

        # === 后续tokenization等保持不变 ===
        encoded_text = self.tokenizer(
            biotext,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded_text['input_ids'].squeeze(0)
        attention_mask = encoded_text['attention_mask'].squeeze(0)
        
        batch['text_input_ids'] = input_ids
        batch['text_attention_mask'] = attention_mask
        batch['all_candidate_texts'] = all_candidate_texts
        
        # MLM task
        if self.use_mlm:
            masked_input_ids, mlm_labels = self._create_mlm_inputs(input_ids.clone())
            batch['masked_text_input_ids'] = masked_input_ids
            batch['mlm_labels'] = mlm_labels
        
        # Paraphrase
        if paraphrase_text is not None:
            encoded_para = self.tokenizer(
                paraphrase_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            batch['paraphrase_input_ids'] = encoded_para['input_ids'].squeeze(0)
            batch['paraphrase_attention_mask'] = encoded_para['attention_mask'].squeeze(0)
            batch['has_paraphrase'] = True
        else:
            batch['has_paraphrase'] = False
        
        batch['ms2_id'] = ms2_id
        batch['molecule_id'] = molecule_id
        batch['original_text'] = biotext
        
        return batch
    
    @staticmethod
    def custom_collate_fn(batch_list):
        """
        自定义collate_fn，用于处理字典形式的批次数据。
        支持可选的keys，并过滤与batch正样本冲突的hard negatives。
        """
        if not batch_list:
            return {}
        
        # 收集batch中所有正样本的molecule_id
        batch_molecule_ids = set()
        for sample in batch_list:
            if 'molecule_id' in sample:
                batch_molecule_ids.add(sample['molecule_id'])
        
        # 收集所有可能的keys
        all_keys = set()
        for d in batch_list:
            all_keys.update(d.keys())
        
        # 按key分组收集数据
        collated_batch = {}
        for key in all_keys:
            values = [d[key] for d in batch_list if key in d]
            collated_batch[key] = values
        
        # === 过滤冲突的hard negatives ===
        if 'hard_neg_input_ids' in collated_batch and len(batch_molecule_ids) > 0:
            filtered_hard_neg_ids = []
            filtered_hard_neg_masks = []
            filtered_has_hard_neg = []
            
            for i, sample in enumerate(batch_list):
                if sample.get('has_hard_neg', False):
                    hard_neg_ids = sample['hard_neg_input_ids']  # [num_hard_neg, seq_len]
                    hard_neg_mask = sample['hard_neg_attention_mask']
                    hard_neg_mol_ids = sample.get('hard_neg_molecule_ids', [])
                    
                    # 找出不在batch中的hard negatives的索引
                    valid_indices = []
                    for j, neg_mol_id in enumerate(hard_neg_mol_ids):
                        if neg_mol_id not in batch_molecule_ids:
                            valid_indices.append(j)
                    
                    if valid_indices:
                        # 只保留不冲突的hard negatives
                        filtered_hard_neg_ids.append(hard_neg_ids[valid_indices])
                        filtered_hard_neg_masks.append(hard_neg_mask[valid_indices])
                        filtered_has_hard_neg.append(True)
                    else:
                        # 所有hard negatives都冲突，设为空tensor
                        max_length = sample['text_input_ids'].shape[0]
                        filtered_hard_neg_ids.append(
                            torch.zeros((0, max_length), dtype=torch.long)
                        )
                        filtered_hard_neg_masks.append(
                            torch.zeros((0, max_length), dtype=torch.long)
                        )
                        filtered_has_hard_neg.append(False)
                else:
                    # 原本就没有hard negatives
                    max_length = batch_list[0]['text_input_ids'].shape[0]
                    filtered_hard_neg_ids.append(
                        torch.zeros((0, max_length), dtype=torch.long)
                    )
                    filtered_hard_neg_masks.append(
                        torch.zeros((0, max_length), dtype=torch.long)
                    )
                    filtered_has_hard_neg.append(False)
            
            collated_batch['hard_neg_input_ids'] = filtered_hard_neg_ids
            collated_batch['hard_neg_attention_mask'] = filtered_hard_neg_masks
            collated_batch['has_hard_neg'] = filtered_has_hard_neg
        
        # 逐个处理字典中的键值
        final_batch = {}
        for key, values in collated_batch.items():
            # 可选的boolean flags
            if key in ['has_paraphrase', 'has_hard_neg']:
                final_batch[key] = [d.get(key, False) for d in batch_list]
            
            # 保持为list的keys（包括新增的hard_neg_molecule_ids）
            elif key in ['hard_neg_input_ids', 'hard_neg_attention_mask',
                        'paraphrase_input_ids', 'paraphrase_attention_mask',
                        'hard_neg_molecule_ids']:
                final_batch[key] = values
            
            # 对张量进行堆叠
            elif isinstance(values[0], torch.Tensor):
                if len(values) == len(batch_list):
                    final_batch[key] = torch.stack(values)
                else:
                    final_batch[key] = values
            
            # 其他类型直接保留
            else:
                final_batch[key] = values
        
        return final_batch
        
    @staticmethod
    def custom_collate_fn(batch_list):
        """
        自定义collate_fn，识别batch中有text overlap的样本对
        """
        if not batch_list:
            return {}
        
        batch_size = len(batch_list)
        
        # 收集batch中所有正样本的molecule_id
        batch_molecule_ids = set()
        for sample in batch_list:
            if 'molecule_id' in sample:
                batch_molecule_ids.add(sample['molecule_id'])
        
        # === NEW: 构建text overlap矩阵 ===
        # text_overlap[i][j] = 1 表示样本i和样本j有共享的候选text
        text_overlap = torch.zeros(batch_size, batch_size, dtype=torch.float32)
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    text_overlap[i, j] = 1.0  # 自己和自己肯定overlap
                else:
                    # 检查候选text集合是否有交集
                    texts_i = set(batch_list[i].get('all_candidate_texts', []))
                    texts_j = set(batch_list[j].get('all_candidate_texts', []))
                    
                    if texts_i & texts_j:  # 有交集
                        text_overlap[i, j] = 1.0
        
        # 收集所有可能的keys
        all_keys = set()
        for d in batch_list:
            all_keys.update(d.keys())
        
        # 按key分组收集数据
        collated_batch = {}
        for key in all_keys:
            values = [d[key] for d in batch_list if key in d]
            collated_batch[key] = values
        
        # === 过滤冲突的hard negatives ===
        if 'hard_neg_input_ids' in collated_batch and len(batch_molecule_ids) > 0:
            filtered_hard_neg_ids = []
            filtered_hard_neg_masks = []
            filtered_has_hard_neg = []
            
            for i, sample in enumerate(batch_list):
                if sample.get('has_hard_neg', False):
                    hard_neg_ids = sample['hard_neg_input_ids']
                    hard_neg_mask = sample['hard_neg_attention_mask']
                    hard_neg_mol_ids = sample.get('hard_neg_molecule_ids', [])
                    
                    valid_indices = []
                    for j, neg_mol_id in enumerate(hard_neg_mol_ids):
                        if neg_mol_id not in batch_molecule_ids:
                            valid_indices.append(j)
                    
                    if valid_indices:
                        filtered_hard_neg_ids.append(hard_neg_ids[valid_indices])
                        filtered_hard_neg_masks.append(hard_neg_mask[valid_indices])
                        filtered_has_hard_neg.append(True)
                    else:
                        max_length = sample['text_input_ids'].shape[0]
                        filtered_hard_neg_ids.append(
                            torch.zeros((0, max_length), dtype=torch.long)
                        )
                        filtered_hard_neg_masks.append(
                            torch.zeros((0, max_length), dtype=torch.long)
                        )
                        filtered_has_hard_neg.append(False)
                else:
                    max_length = batch_list[0]['text_input_ids'].shape[0]
                    filtered_hard_neg_ids.append(
                        torch.zeros((0, max_length), dtype=torch.long)
                    )
                    filtered_hard_neg_masks.append(
                        torch.zeros((0, max_length), dtype=torch.long)
                    )
                    filtered_has_hard_neg.append(False)
            
            collated_batch['hard_neg_input_ids'] = filtered_hard_neg_ids
            collated_batch['hard_neg_attention_mask'] = filtered_hard_neg_masks
            collated_batch['has_hard_neg'] = filtered_has_hard_neg
        
        # 逐个处理字典中的键值
        final_batch = {}
        for key, values in collated_batch.items():
            if key in ['has_paraphrase', 'has_hard_neg']:
                final_batch[key] = [d.get(key, False) for d in batch_list]
            elif key in ['hard_neg_input_ids', 'hard_neg_attention_mask',
                        'paraphrase_input_ids', 'paraphrase_attention_mask',
                        'hard_neg_molecule_ids', 'all_candidate_texts']:  # all_candidate_texts保持list
                final_batch[key] = values
            elif isinstance(values[0], torch.Tensor):
                if len(values) == len(batch_list):
                    final_batch[key] = torch.stack(values)
                else:
                    final_batch[key] = values
            else:
                final_batch[key] = values
        
        # === 添加text_overlap信息 ===
        final_batch['text_overlap_matrix'] = text_overlap  # [batch_size, batch_size]
        
        return final_batch

            
    @staticmethod
    def load_hmdb_data_subsections(first_path, second_path, jsonl_path, max_text_sharing=5):
        """
        使用subsections JSONL格式读取数据，并过滤高频共享的text
        
        参数:
        first_path (str): MS2数据文件路径（h5、pkl等）
        second_path (str): Meta数据文件路径（parquet、csv等）
        jsonl_path (str): BioText的jsonl文件路径
        max_text_sharing (int): text最多可以被多少个molecule共享，超过则删除
        
        返回:
        tuple: (ms2_data, meta_data, biotext_data)
            biotext_data格式: {molecule_id: [{'type': 'xxx', 'text': 'xxx'}, ...]}
        """
        from collections import defaultdict
        
        # 读取MS2数据
        ms2_data = {}
        try:
            _, ext1 = os.path.splitext(first_path)
            if ext1 == '.h5':
                with h5py.File(first_path, 'r') as f:
                    spectra_group = f['spectra']
                    for spectrum_id in spectra_group.keys():
                        group = spectra_group[spectrum_id]
                        parts = spectrum_id.split('_')
                        molecule_id = parts[0]
                        ms2_data[spectrum_id] = {
                            'mz': group['mz'][...].tolist(),
                            'intensity': group['intensity'][...].tolist(),
                            'molecule_id': molecule_id
                        }
            elif ext1 == '.pkl':
                with open(first_path, 'rb') as f:
                    ms2_data = pickle.load(f)
            else:
                print(f"Unsupported file format for first path: {ext1}")
                return None, None, None
        except Exception as e:
            print(f"Error: Failed to read first file: {first_path}. Error message: {str(e)}")
            return None, None, None

        # 读取Meta数据
        meta_data = None
        try:
            _, ext2 = os.path.splitext(second_path)
            if ext2 == '.parquet':
                meta_data = pd.read_parquet(second_path)
            elif ext2 == '.csv':
                meta_data = pd.read_csv(second_path)
            else:
                print(f"Unsupported file format for second path: {ext2}")
                return ms2_data, None, None
        except Exception as e:
            print(f"Error: Failed to read second file: {second_path}. Error message: {str(e)}")
            return ms2_data, None, None

        # 读取 BioText JSONL 文件
        biotext_data = {}
        try:
            import json
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    accession = item['accession']
                    
                    if accession not in biotext_data:
                        biotext_data[accession] = []
                    
                    biotext_data[accession].append({
                        'type': item['type'],
                        'text': item['text']
                    })
            
            print(f"✓ Loaded BioText subsections from {os.path.basename(jsonl_path)}")
            print(f"  Total molecules: {len(biotext_data)}")
            total_records_before = sum(len(v) for v in biotext_data.values())
            print(f"  Total records (before filtering): {total_records_before}, Avg per molecule: {total_records_before / len(biotext_data):.1f}")
            
        except Exception as e:
            print(f"Error reading BioText JSONL: {e}")
            return ms2_data, meta_data, None

        # ===== 过滤高频共享的text =====
        print(f"\n=== Filtering texts shared by >{max_text_sharing} molecules ===")
        
        # 1. 构建text -> molecules的倒排索引
        text_to_molecules = defaultdict(set)
        for mol_id, records in biotext_data.items():
            for record in records:
                text = record['text']
                if text:  # 避免空字符串
                    text_to_molecules[text].add(mol_id)
        
        # 2. 找出需要删除的高频text
        texts_to_remove = set()
        sharing_distribution = defaultdict(int)  # 统计分布
        
        for text, molecules in text_to_molecules.items():
            sharing_count = len(molecules)
            sharing_distribution[sharing_count] += 1
            
            if sharing_count > max_text_sharing:
                texts_to_remove.add(text)
        
        print(f"Text sharing distribution (top 10):")
        for count in sorted(sharing_distribution.keys(), reverse=True)[:10]:
            print(f"  {count} molecules share: {sharing_distribution[count]} texts")
        
        print(f"\nFound {len(texts_to_remove)} texts to remove (shared by >{max_text_sharing} molecules)")
        
        # 3. 从每个molecule的候选text中删除这些高频text
        filtered_biotext_data = {}
        total_removed = 0
        molecules_with_no_text = []
        
        for mol_id, records in biotext_data.items():
            filtered_records = [record for record in records 
                            if record['text'] not in texts_to_remove]
            
            if filtered_records:
                filtered_biotext_data[mol_id] = filtered_records
                total_removed += len(records) - len(filtered_records)
            else:
                molecules_with_no_text.append(mol_id)
                total_removed += len(records)
        
        # 4. 统计信息
        print(f"\nFiltering results:")
        print(f"  Text entries removed: {total_removed}")
        print(f"  Molecules before: {len(biotext_data)}")
        print(f"  Molecules after: {len(filtered_biotext_data)}")
        print(f"  Molecules with no text left: {len(molecules_with_no_text)}")
        
        if molecules_with_no_text:
            print(f"  ⚠️ Warning: {len(molecules_with_no_text)} molecules lost all texts")
            if len(molecules_with_no_text) <= 5:
                print(f"    Lost: {molecules_with_no_text}")
            else:
                print(f"    First 5: {molecules_with_no_text[:5]}")
        
        # 5. 验证过滤效果
        text_to_molecules_after = defaultdict(set)
        for mol_id, records in filtered_biotext_data.items():
            for record in records:
                text = record['text']
                if text:
                    text_to_molecules_after[text].add(mol_id)
        
        max_sharing_after = max(len(mols) for mols in text_to_molecules_after.values()) if text_to_molecules_after else 0
        shared_texts_after = sum(1 for mols in text_to_molecules_after.values() if len(mols) > 1)
        
        print(f"  Max sharing after filtering: {max_sharing_after} molecules")
        print(f"  Texts still shared by multiple molecules: {shared_texts_after}")
        
        total_records_after = sum(len(v) for v in filtered_biotext_data.values())
        print(f"  Total records (after filtering): {total_records_after}, Avg per molecule: {total_records_after / len(filtered_biotext_data):.1f}")
        
        # 使用过滤后的数据
        biotext_data = filtered_biotext_data

        # 打印统计
        unique_molecule_ids = set(item['molecule_id'] for item in ms2_data.values())
        print(f"\nFinal data summary:")
        print(f"  Unique molecule IDs in MS2 data: {len(unique_molecule_ids)}")
        print(f"  Molecule IDs in BioText data: {len(biotext_data)}")

        return ms2_data, meta_data, biotext_data

    @staticmethod
    def missing_biotext_handling(ms2_data, biotext_data, method="drop"):
        """
        ms2_data: dict, {ms2_id: {'mz': list, 'intensity': list, 'molecule_id': str}}
        biotext_data: dict, {molecule_id: BioText}
        """
        # If a molecule in ms2_data is missing in biotext_data, remove it from ms2_data
        # Handle missing biotext entries
        if method == "drop":
            ms2_data = {ms2_id: info for ms2_id, info in ms2_data.items() if info['molecule_id'] in biotext_data}
            unique_molecule_ids = set(item['molecule_id'] for item in ms2_data.values())
            print(f"Post-processing statistics ('drop' method) - Unique molecule IDs in MS2 data: {len(unique_molecule_ids)}")
            print(f"Post-processing statistics ('drop' method) - Molecule IDs in BioText data: {len(biotext_data)}")
            return ms2_data, biotext_data

        if method == "fill":
            # Fill missing entries with empty values; will be handled during dataset initialization
            for info in ms2_data.values():
                molecule_id = info['molecule_id']
                if molecule_id not in biotext_data:
                    biotext_data[molecule_id] = ""
            unique_molecule_ids = set(item['molecule_id'] for item in ms2_data.values())
            print(f"Post-processing statistics ('fill' method) - Unique molecule IDs in MS2 data: {len(unique_molecule_ids)}")
            print(f"Post-processing statistics ('fill' method) - Molecule IDs in BioText data: {len(biotext_data)}")
            return ms2_data, biotext_data

        raise ValueError(f"Unknown method: {method}. Method must be 'drop' or 'fill'.")


    @staticmethod
    def add_noise_peaks(peaks, intensities, noise_ratio=0.5, noise_intensity_range=(0.001, 0.05), seed=None):
        """
        添加随机noise peaks来模拟外部数据
        
        Args:
            peaks: list of float, 原始m/z值
            intensities: list of float, 原始强度值
            noise_ratio: float, 添加的noise peaks数量 = 原peaks数 × noise_ratio
            noise_intensity_range: tuple, noise的相对强度范围（相对于max intensity）
            seed: int, 随机种子（可选）
        
        Returns:
            aug_peaks: list of float, 添加noise后的m/z
            aug_intensities: list of float, 添加noise后的强度
        """
        import numpy as np
        import random
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        if len(peaks) == 0:
            return peaks, intensities
        
        max_int = max(intensities)
        if max_int == 0:
            return peaks, intensities
        
        # 计算要添加的noise数量
        n_noise = int(len(peaks) * noise_ratio)
        if n_noise == 0:
            return peaks, intensities
        
        # 在光谱范围内随机生成noise peaks的m/z
        mz_min, mz_max = min(peaks), max(peaks)
        noise_mz = np.random.uniform(mz_min, mz_max, n_noise).tolist()
        
        # 生成低强度noise（相对于max intensity）
        noise_int = np.random.uniform(
            noise_intensity_range[0] * max_int,
            noise_intensity_range[1] * max_int,
            n_noise
        ).tolist()
        
        # 合并原始peaks和noise
        aug_peaks = peaks + noise_mz
        aug_intensities = intensities + noise_int
        
        # 按m/z排序
        sorted_indices = sorted(range(len(aug_peaks)), key=lambda i: aug_peaks[i])
        aug_peaks = [aug_peaks[i] for i in sorted_indices]
        aug_intensities = [aug_intensities[i] for i in sorted_indices]
        
        return aug_peaks, aug_intensities


    @staticmethod
    def filter_low_intensity_peaks(peaks, intensities, threshold=0.01):
        """
        过滤低强度peaks
        
        Args:
            peaks: list of float, m/z值
            intensities: list of float, 强度值
            threshold: float, 相对强度阈值（0.01 = 1%）
        
        Returns:
            filtered_peaks: list of float
            filtered_intensities: list of float
        """
        if len(peaks) == 0 or len(intensities) == 0:
            return peaks, intensities
        
        max_int = max(intensities)
        if max_int == 0:
            return peaks, intensities
        
        # 归一化并过滤
        norm_intensities = [i / max_int for i in intensities]
        filtered_peaks = []
        filtered_intensities = []
        
        for mz, intensity, norm_int in zip(peaks, intensities, norm_intensities):
            if norm_int >= threshold:
                filtered_peaks.append(mz)
                filtered_intensities.append(intensity)
        
        return filtered_peaks, filtered_intensities


    @staticmethod
    def augment_ms2_data(ms2_data, args):
        """
        对MS2数据进行增强（必须在preprocess之前调用）
        
        Args:
            ms2_data: dict, {ms2_id: {'mz': list, 'intensity': list, 'molecule_id': str}}
                    注意：mz和intensity必须是原始的float值，不能是token_ids
            args: argparse.Namespace, 包含增强参数:
                - augment_noise: bool, 是否添加noise增强 (default: False)
                - augment_multiplier: int, 每个光谱生成几个版本 (1=不增强, 2=生成2倍数据)
                - noise_ratio: float, 添加的noise数量 = 原peaks数 × noise_ratio
                - noise_intensity_range: tuple, noise强度范围 (相对于max intensity)
                - filter_threshold: float or None, 过滤低强度peaks的阈值
        
        Returns:
            augmented_ms2_data: dict, 包含原始+增强版本的数据
                            如果augment_multiplier=1，返回原始数据
                            如果augment_multiplier=2，返回2倍数据（原始+1个增强版本）
        
        Example:
            >>> augmented_data = MS2BioTextDataset.augment_ms2_data(ms2_data, args)
            >>> processed_data, word2idx = MS2BioTextDataset.preprocess_ms2_data_positive_only(
            ...     augmented_data, meta_data
            ... )
        """
        import numpy as np
        
        # 获取参数（兼容没有这些参数的情况）
        augment_noise = getattr(args, 'augment_noise', False)
        augment_multiplier = getattr(args, 'augment_multiplier', 1)
        noise_ratio = getattr(args, 'noise_ratio', 0.5)
        noise_intensity_range = getattr(args, 'noise_intensity_range', (0.001, 0.05))
        filter_threshold = getattr(args, 'filter_threshold', None)
        
        # 如果不需要增强，直接返回原数据
        if not augment_noise or augment_multiplier <= 1:
            print("ℹ️  未启用数据增强 (augment_noise=False or augment_multiplier<=1)")
            return ms2_data
        
        print(f"\n{'='*60}")
        print(f"🔄 MS2数据增强")
        print(f"{'='*60}")
        print(f"  增强倍数: {augment_multiplier}x")
        print(f"  Noise比例: {noise_ratio}")
        print(f"  Noise强度范围: {noise_intensity_range}")
        if filter_threshold:
            print(f"  过滤阈值: {filter_threshold} (相对强度)")
        print(f"  原始光谱数: {len(ms2_data)}")
        
        augmented_ms2_data = {}
        
        for ms2_id, info in ms2_data.items():
            molecule_id = info.get('molecule_id')
            
            # 检查数据格式
            if not isinstance(info['mz'], list) or not isinstance(info['intensity'], list):
                print(f"⚠️  跳过 {ms2_id}: mz或intensity不是list格式")
                continue
            
            # 版本0: 原始数据（可选过滤）
            peaks_original = info['mz'].copy() if isinstance(info['mz'], list) else list(info['mz'])
            intensities_original = info['intensity'].copy() if isinstance(info['intensity'], list) else list(info['intensity'])
            
            # 可选：过滤低强度peaks
            if filter_threshold is not None and filter_threshold > 0:
                peaks_original, intensities_original = MS2BioTextDataset.filter_low_intensity_peaks(
                    peaks_original, intensities_original, threshold=filter_threshold
                )
            
            # 保存原始版本
            augmented_ms2_data[ms2_id] = {
                'mz': peaks_original,
                'intensity': intensities_original,
                'molecule_id': molecule_id
            }
            
            # 生成增强版本（版本1到N-1）
            for aug_idx in range(1, augment_multiplier):
                peaks_aug, intensities_aug = MS2BioTextDataset.add_noise_peaks(
                    peaks_original.copy(),
                    intensities_original.copy(),
                    noise_ratio=noise_ratio,
                    noise_intensity_range=noise_intensity_range,
                    seed=None  # 每次随机生成不同的noise
                )
                
                # 新的ID：原ID + 后缀
                aug_ms2_id = f"{ms2_id}_aug{aug_idx}"
                augmented_ms2_data[aug_ms2_id] = {
                    'mz': peaks_aug,
                    'intensity': intensities_aug,
                    'molecule_id': molecule_id  # 保持相同的molecule_id！
                }
        
        print(f"  ✓ 增强后光谱数: {len(augmented_ms2_data)}")
        print(f"  增强版本数: {len(augmented_ms2_data) - len(ms2_data)}")
        print(f"{'='*60}\n")
        
        return augmented_ms2_data

    @staticmethod
    def preprocess_ms2_data_positive_only(ms2_data, meta_data, maxlen=100, min_peaks=0):
        """
        Preprocess ms2_data for model input.
        
        Parameters:
        - ms2_data: dict, {ms2_id: {'mz': list, 'intensity': list, 'molecule_id': str}}
        - meta_data: pd.DataFrame, must contain precursor information (column: 'precursor_mass')
        - maxlen: int, maximum sequence length
        - min_peaks: int, minimum number of peaks required (default: 0, no filtering)
        
        Returns:
        - ms_data: dict, same structure as ms2_data but with processed 'mz' and 'intensity' sequences
        - word2idx: dict, maps string-formatted m/z values to token indices
        """
        
        # ===== 新增：安全转换precursor_mass的函数 =====
        def safe_convert_precursor(value):
            """安全转换precursor_mass值，处理异常格式"""
            if pd.isna(value):
                return None
            
            # 如果已经是数字
            if isinstance(value, (int, float)):
                return float(value)
            
            # 如果是字符串
            value_str = str(value).strip()
            
            # 处理空字符串
            if value_str == '' or value_str.lower() == 'nan':
                return None
            
            # 处理 "209/192" 这种格式（取第一个值）
            if '/' in value_str:
                try:
                    return float(value_str.split('/')[0])
                except:
                    return None
            
            # 尝试直接转换
            try:
                return float(value_str)
            except:
                return None
        # ============================================
        
        # 1) Create word list: ["0.00", "0.01", ..., "999.99"]
        word_list = list(np.round(np.linspace(0, 1000, 100*1000, endpoint=False), 2))
        word_list = ["%.2f" % i for i in word_list]
        
        # 2) Build word2idx dictionary with special tokens
        word2idx = {'[PAD]': 0, '[MASK]': 1}
        for i, w in enumerate(word_list):
            word2idx[w] = i + 2  # Start from 2 to avoid collision with special tokens
        
        # 3) Initialize output dictionary
        ms_data = {}
        
        # ===== 新增：统计信息 =====
        filter_stats = {
            'total': 0,
            'empty_mz': 0,
            'not_positive': 0,
            'no_meta': 0,
            'no_precursor': 0,
            'precursor_gt_1000': 0,
            'no_peaks_after_filter': 0,
            'too_few_peaks': 0,  # 新增
            'kept': 0
        }
        # ===========================
        
        # 4) Iterate through each ms2_id
        for ms2_id, info in ms2_data.items():
            filter_stats['total'] += 1
            
            mz_data = info.get('mz')
            if mz_data is None or len(mz_data) == 0:
                filter_stats['empty_mz'] += 1
                continue
            peaks = info['mz']
            intensities = info['intensity']
            molecule_id = info.get('molecule_id', None)
            
            specific_row = meta_data[meta_data["file_name"] == ms2_id]
            if specific_row.empty:
                filter_stats['no_meta'] += 1
                continue
            elif specific_row["Polarity"].values[0] not in ["Positive", "positive"]:
                filter_stats['not_positive'] += 1
                continue
            
            # 4.1 Find precursor mass from meta_data
            if 'HMDB.ID' in meta_data.columns:
                row = meta_data.loc[meta_data['HMDB.ID'] == molecule_id]
            else:
                row = meta_data.loc[meta_data.index == molecule_id]
            if row.empty:
                filter_stats['no_meta'] += 1
                continue
            
            # ===== 修改：使用安全转换函数 =====
            precursor_val = safe_convert_precursor(row['precursor_mass'].values[0])
            if precursor_val is None:
                filter_stats['no_precursor'] += 1
                continue
            # ===================================
            
            if precursor_val > 1000:
                filter_stats['precursor_gt_1000'] += 1
                continue
            precursor_str = "%.2f" % precursor_val
            
            # 4.2 Convert m/z values to string and map to indices
            peaks_str = []
            for mz_val in peaks:
                if mz_val <= 1000:
                    peaks_str.append("%.2f" % mz_val)
            
            # ===== 新增：检查peaks数量 =====
            if len(peaks_str) == 0:
                filter_stats['no_peaks_after_filter'] += 1
                continue
            
            if len(peaks_str) < min_peaks:
                filter_stats['too_few_peaks'] += 1
                continue
            # ===============================
            
            token_ids = [word2idx[precursor_str]] + [word2idx[p] for p in peaks_str]
            
            # 4.3 Normalize intensity and prepend a fixed value (2)
            intensities = np.hstack((2, intensities))
            max_intensity = np.max(intensities)
            if max_intensity != 0:
                intensities = intensities / max_intensity
            
            # 4.4 Pad or truncate to maxlen
            n_pad = maxlen - len(token_ids)
            if n_pad < 0:
                token_ids = token_ids[:maxlen]
                intensities = intensities[:maxlen]
                n_pad = 0
            token_ids += [word2idx['[PAD]']] * n_pad
            if len(intensities) < maxlen:
                intensities = np.hstack([intensities, np.zeros(maxlen - len(intensities))])
            else:
                intensities = intensities[:maxlen]
            
            # 4.5 Save processed result
            ms_data[ms2_id] = {
                'mz': token_ids,
                'intensity': intensities.tolist(),
                'molecule_id': molecule_id
            }
            filter_stats['kept'] += 1
        
        # ===== 新增：打印统计信息 =====
        print(f"\n预处理统计:")
        print(f"  总光谱: {filter_stats['total']}")
        print(f"  过滤:")
        print(f"    空M/Z: {filter_stats['empty_mz']}")
        print(f"    非Positive: {filter_stats['not_positive']}")
        print(f"    无meta: {filter_stats['no_meta']}")
        print(f"    无/异常precursor: {filter_stats['no_precursor']}")
        print(f"    precursor>1000: {filter_stats['precursor_gt_1000']}")
        print(f"    所有peaks>1000: {filter_stats['no_peaks_after_filter']}")
        if min_peaks > 0:
            print(f"    peaks<{min_peaks}: {filter_stats['too_few_peaks']}")
        print(f"  ✓ 保留: {filter_stats['kept']} ({filter_stats['kept']/filter_stats['total']*100:.2f}%)")
        # ================================
        
        # 5) Return processed data and dictionary
        return ms_data, word2idx


    @staticmethod
    def augment_ms2_data_parallel(ms2_data, args, n_workers=None):
        """多进程版本的augment_ms2_data"""
        from multiprocessing import Pool, cpu_count
        
        if n_workers is None:
            n_workers = min(cpu_count() - 1, 8)
        
        augment_noise = getattr(args, 'augment_noise', False)
        augment_multiplier = getattr(args, 'augment_multiplier', 1)
        
        if not augment_noise or augment_multiplier <= 1:
            print("ℹ️  未启用数据增强")
            return ms2_data
        
        print(f"\n🚀 多进程数据增强 (workers={n_workers})...")
        
        # 准备参数
        items = list(ms2_data.items())
        chunk_size = max(1, len(items) // (n_workers * 4))
        
        # 提取参数
        filter_threshold = getattr(args, 'filter_threshold', None)
        noise_ratio = getattr(args, 'noise_ratio', 0.5)
        noise_intensity_range = getattr(args, 'noise_intensity_range', (0.001, 0.05))
        
        # 分批并打包参数
        batches = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
        batch_data = [(batch, filter_threshold, noise_ratio, noise_intensity_range, augment_multiplier) 
                      for batch in batches]
        
        with Pool(n_workers) as pool:
            results = pool.map(_augment_worker, batch_data)
        
        # 合并结果
        augmented_data = {}
        for r in results:
            augmented_data.update(r)
        
        print(f"  ✓ 完成: {len(augmented_data)} 光谱")
        return augmented_data
    

    @staticmethod
    def preprocess_ms2_data_positive_only_parallel(ms2_data, meta_data, maxlen=100, min_peaks=0, n_workers=None,
                                                    precursor_mode='normalize_add', precursor_value=2.0):
        """
        多进程版本的preprocess
        
        Args:
            precursor_mode: 
                - 'scale_fixed': 缩放fragments到precursor_value（如20000），precursor固定为2
                - 'normalize_add': 归一化fragments到1，precursor用precursor_value（如2.0），再整体归一化
                - 'original': 原始MSBERT方式
            precursor_value: 
                - mode='scale_fixed'时: fragments缩放的目标值（默认20000）
                - mode='normalize_add'时: precursor的强度值（默认2.0）
        """
        from multiprocessing import Pool, cpu_count
        import numpy as np
        import pandas as pd
        
        if n_workers is None:
            n_workers = min(cpu_count() - 1, 8)
        
        print(f"\n🚀 多进程预处理 (workers={n_workers}, mode={precursor_mode}, value={precursor_value})...")
        
        # 构建word2idx
        word_list = list(np.round(np.linspace(0, 1000, 100*1000, endpoint=False), 2))
        word_list = ["%.2f" % i for i in word_list]
        word2idx = {'[PAD]': 0, '[MASK]': 1}
        for i, w in enumerate(word_list):
            word2idx[w] = i + 2
        
        # 预处理 meta_data
        meta_data_processed = meta_data.copy()
        if "Polarity" in meta_data_processed.columns:
            meta_data_processed["Polarity"] = meta_data_processed["Polarity"].astype(str).str.lower().str.strip()
        
        # 计算最大碎片数
        max_frag = max(0, min(100, maxlen - 1))
        
        # 准备数据
        items = list(ms2_data.items())
        chunk_size = max(1, len(items) // (n_workers * 4))
        
        # 分批并打包参数
        batches = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
        batch_data = [(batch, word2idx, meta_data_processed, maxlen, max_frag, min_peaks, 
                    precursor_mode, precursor_value)
                    for batch in batches]
        
        with Pool(n_workers) as pool:
            results = pool.map(_preprocess_worker, batch_data)
        
        # 合并
        ms_data = {}
        total_kept = 0
        total_filtered = 0
        for r, stats in results:
            ms_data.update(r)
            total_kept += stats['kept']
            total_filtered += stats['filtered']
        
        print(f"  ✓ 完成: {total_kept}/{len(items)} 光谱 (过滤: {total_filtered})")
        return ms_data, word2idx

    @staticmethod
    def preprocess_ms2_data(ms2_data, meta_data, maxlen=100):
        """
        Preprocess ms2_data for model input.

        Parameters:
        - ms2_data: dict, {ms2_id: {'mz': list, 'intensity': list, 'molecule_id': str}}
        - meta_data: pd.DataFrame, must contain precursor information (column: 'precursor_mass')
        - maxlen: int, maximum sequence length

        Returns:
        - ms_data: dict, same structure as ms2_data but with processed 'mz' and 'intensity' sequences
        - word2idx: dict, maps string-formatted m/z values to token indices
        """
        # 1) Create word list: ["0.00", "0.01", ..., "999.99"]
        word_list = list(np.round(np.linspace(0, 1000, 100*1000, endpoint=False), 2))
        word_list = ["%.2f" % i for i in word_list]

        # 2) Build word2idx dictionary with special tokens
        word2idx = {'[PAD]': 0, '[MASK]': 1}
        for i, w in enumerate(word_list):
            word2idx[w] = i + 2  # Start from 2 to avoid collision with special tokens

        # 3) Initialize output dictionary
        ms_data = {}

        # 4) Iterate through each ms2_id
        for ms2_id, info in ms2_data.items():
            if not info['mz']:
                continue
            peaks = info['mz']
            intensities = info['intensity']
            molecule_id = info.get('molecule_id', None)

            # 4.1 Find precursor mass from meta_data
            if 'HMDB.ID' in meta_data.columns:
                row = meta_data.loc[meta_data['HMDB.ID'] == molecule_id]
            else:
                row = meta_data.loc[meta_data.index == molecule_id]
            if row.empty:
                continue
            precursor_val = float(row['precursor_mass'].values[0])
            if pd.isna(precursor_val):
                continue
            if precursor_val > 1000:
                continue
            precursor_str = "%.2f" % precursor_val

            # 4.2 Convert m/z values to string and map to indices
            peaks_str = []
            for mz_val in peaks:
                if mz_val <= 1000:
                    peaks_str.append("%.2f" % mz_val)
                else:
                    continue
            token_ids = [word2idx[precursor_str]] + [word2idx[p] for p in peaks_str]

            # 4.3 Normalize intensity and prepend a fixed value (2)
            intensities = np.hstack((2, intensities))
            max_intensity = np.max(intensities)
            if max_intensity != 0:
                intensities = intensities / max_intensity

            # 4.4 Pad or truncate to maxlen
            n_pad = maxlen - len(token_ids)
            if n_pad < 0:
                token_ids = token_ids[:maxlen]
                intensities = intensities[:maxlen]
                n_pad = 0
            token_ids += [word2idx['[PAD]']] * n_pad
            if len(intensities) < maxlen:
                intensities = np.hstack([intensities, np.zeros(maxlen - len(intensities))])
            else:
                intensities = intensities[:maxlen]

            # 4.5 Save processed result
            ms_data[ms2_id] = {
                'mz': token_ids,
                'intensity': intensities.tolist(),
                'molecule_id': molecule_id
            }

        # 5) Return processed data and dictionary
        return ms_data, word2idx

    @staticmethod
    def fill_precursor_data(meta_data, ms_data):
        """
        Fill in missing precursor ion mass in mass spectrometry metadata.

        Parameters:
        meta_data (DataFrame): DataFrame containing metadata for mass spectrometry samples.
        ms_data (dict): Dictionary containing peak data, with keys as spectrum IDs and values as dicts with 'mz' and 'intensity'.

        Returns:
        DataFrame: Updated meta_data with filled precursor ion mass.
        """
        # Copy the DataFrame to avoid modifying the original
        meta_data = meta_data.copy()

        # Find column name containing 'precursor'
        precursor_cols = [col for col in meta_data.columns if 'precursor' in col.lower()]
        if not precursor_cols:
            raise ValueError("No column containing 'precursor' found in meta_data")
        precursor_col = precursor_cols[0]
        print(f"Using '{precursor_col}' as the precursor mass column")

        init_nan = meta_data[precursor_col].isna().sum()

        # Find column named 'mz'
        mz_cols = [col for col in meta_data.columns if col.lower() == 'mz']
        has_mz_column = len(mz_cols) > 0
        mz_col = mz_cols[0] if has_mz_column else None

        # Proton mass (H+) is approximately 1.007276 Da
        proton_mass = 1.007276

        # Tolerance threshold for isotopic effect (Da)
        isotope_threshold = 2.0

        # Dictionary of adduct ion modes with their corresponding mass calculations
        adduct_modes = {
            'positive': {
                '[M+H]+': lambda m: m + proton_mass,
                '[M+H-H2O]+': lambda m: m + proton_mass - 18.010565,
                '[M+Na]+': lambda m: m + 22.989218,
                '[M+K]+': lambda m: m + 39.098301,
                '[M+NH4]+': lambda m: m + 18.033823,
                '[2M+H]+': lambda m: 2*m + proton_mass,
                '[2M+Na]+': lambda m: 2*m + 22.989218,
                '[2M+K]+': lambda m: 2*m + 39.098301,
                '[2M+NH4]+': lambda m: 2*m + 18.033823,
                '[2M+H-H2O]+': lambda m: 2*m + proton_mass - 18.010565
            },
            'negative': {
                '[M-H]-': lambda m: m - proton_mass,
                '[M-H2O-H]-': lambda m: m - proton_mass - 18.010565,
                '[M+Cl]-': lambda m: m + 34.969402,
                '[M+HAc-H]-': lambda m: m + 59.013851,
                '[2M-H]-': lambda m: 2*m - proton_mass,
                '[2M+Cl]-': lambda m: 2*m + 34.969402,
                '[2M+HAc-H]-': lambda m: 2*m + 59.013851
            }
        }
        meta_data[precursor_col] = pd.to_numeric(meta_data[precursor_col], errors='coerce')
        # Replace negative precursor values with NaN
        meta_data.loc[meta_data[precursor_col] < 0, precursor_col] = np.nan

        # Fill missing precursor masses
        for idx, row in meta_data.iterrows():
            if pd.isna(row[precursor_col]):
                spectrum_id = idx

                # Determine polarity
                polarity = str(row['Polarity']).lower()
                if 'positive' in polarity:
                    polarity_type = 'positive'
                elif 'negative' in polarity:
                    polarity_type = 'negative'
                else:
                    polarity_type = 'positive'

                # Try to get base mz from meta_data
                base_mz = None
                if has_mz_column and not pd.isna(row[mz_col]):
                    base_mz = row[mz_col]
                else:
                    if row["file_name"] not in ms_data:
                        print(f"Warning: spectrum ID {spectrum_id} not found in ms_data")
                        continue

                    spectrum = ms_data.get(row["file_name"], {})
                    if 'mz' not in spectrum or len(spectrum['mz']) == 0:
                        print(f"Warning: spectrum ID {spectrum_id} has no mz data")
                        continue

                    base_mz = max(spectrum['mz'])

                candidate_precursors = {}
                for mode_name, mode_func in adduct_modes[polarity_type].items():
                    candidate_mass = mode_func(base_mz)
                    candidate_precursors[mode_name] = candidate_mass

                valid_candidates = {}
                spectrum = ms_data.get(row["file_name"], {})
                if 'mz' in spectrum and len(spectrum['mz']) > 0:
                    max_fragment_mz = max(spectrum['mz'])

                    for mode_name, precursor_mass in candidate_precursors.items():
                        if max_fragment_mz <= precursor_mass + isotope_threshold:
                            valid_candidates[mode_name] = precursor_mass

                if valid_candidates:
                    default_mode = '[M+H]+' if polarity_type == 'positive' else '[M-H]-'
                    if default_mode in valid_candidates:
                        selected_mode = default_mode
                    else:
                        selected_mode = list(valid_candidates.keys())[0]
                    precursor_mass = valid_candidates[selected_mode]
                    meta_data.at[idx, precursor_col] = precursor_mass
                else:
                    if 'mz' in spectrum and len(spectrum['mz']) > 0:
                        max_mz = max(spectrum['mz'])
                        if polarity_type == 'positive':
                            adjusted_precursor = max_mz + proton_mass + 1.0
                        else:
                            adjusted_precursor = max_mz - proton_mass + 1.0
                        meta_data.at[idx, precursor_col] = adjusted_precursor
                    else:
                        print(f"Warning: unable to determine precursor mass for spectrum ID {spectrum_id}")

        left_nan = meta_data[precursor_col].isna().sum()
        print(f"Precursor mass missing: initially {init_nan}; filled {init_nan - left_nan}; remaining {left_nan}.")

        return meta_data

    
    @staticmethod
    def preprocess_ms2_data_positive_only(ms2_data, meta_data, maxlen=100):
        """
        Preprocess ms2_data for model input.

        Parameters:
        - ms2_data: dict, {ms2_id: {'mz': list, 'intensity': list, 'molecule_id': str}}
        - meta_data: pd.DataFrame, must contain precursor information (column: 'precursor_mass')
        - maxlen: int, maximum sequence length

        Returns:
        - ms_data: dict, same structure as ms2_data but with processed 'mz' and 'intensity' sequences
        - word2idx: dict, maps string-formatted m/z values to token indices
        """
        import numpy as np
        import pandas as pd

        # 1) Create word list: ["0.00", "0.01", ..., "999.99"]
        word_list = list(np.round(np.linspace(0, 1000, 100 * 1000, endpoint=False), 2))
        word_list = ["%.2f" % i for i in word_list]

        # 2) Build word2idx dictionary with special tokens
        word2idx = {'[PAD]': 0, '[MASK]': 1}
        for i, w in enumerate(word_list):
            word2idx[w] = i + 2  # Start from 2 to avoid collision with special tokens

        # 3) Initialize output dictionary
        ms_data = {}

        # 预计算：正离子模式的判定更稳健（lower+strip）
        if "Polarity" in meta_data.columns:
            meta_data = meta_data.copy()
            meta_data["Polarity"] = meta_data["Polarity"].astype(str).str.lower().str.strip()

        # 允许的最大碎片数（保证前体+碎片 <= maxlen）
        max_frag = max(0, min(100, maxlen - 1))

        # 4) Iterate through each ms2_id
        for ms2_id, info in ms2_data.items():
            # 基础检查
            if not info.get('mz'):
                continue

            peaks = np.asarray(info['mz'], dtype=float)
            intensities = np.asarray(info['intensity'], dtype=float)
            molecule_id = info.get('molecule_id', None)

            # 4.0 文件名对应行用于极性判断
            specific_row = meta_data.loc[meta_data["file_name"] == ms2_id] if "file_name" in meta_data.columns else pd.DataFrame()
            if specific_row.empty:
                # 若找不到，就尽量用 molecule_id 定位一行（不强制）
                if molecule_id is not None:
                    if 'HMDB.ID' in meta_data.columns:
                        specific_row = meta_data.loc[meta_data['HMDB.ID'] == molecule_id]
                    else:
                        specific_row = meta_data.loc[meta_data.index == molecule_id]
            if specific_row.empty:
                continue

            # 只保留正离子
            pol = str(specific_row["Polarity"].values[0]).lower().strip() if "Polarity" in specific_row.columns else ""
            if pol != "positive":
                continue

            # 4.1 Find precursor mass from meta_data
            if 'HMDB.ID' in meta_data.columns and (molecule_id is not None):
                row = meta_data.loc[meta_data['HMDB.ID'] == molecule_id]
            else:
                row = meta_data.loc[meta_data.index == molecule_id]

            if row.empty or ('precursor_mass' not in row.columns):
                continue

            try:
                precursor_val = float(row['precursor_mass'].values[0])
            except Exception:
                continue

            # 前体范围 [10, 1000)；并避免 1000.00 被格式化后越界
            if pd.isna(precursor_val) or (precursor_val < 10.0) or (precursor_val >= 1000.0):
                continue
            precursor_val = min(precursor_val, 999.99)
            precursor_str = "%.2f" % precursor_val

            # 4.2 过滤峰到 [10, 1000)
            if peaks.shape[0] != intensities.shape[0]:
                # 长度不一致直接跳过（也可选择截断到对齐最短）
                n = min(len(peaks), len(intensities))
                peaks = peaks[:n]
                intensities = intensities[:n]

            mask = (peaks >= 10.0) & (peaks < 1000.0) & np.isfinite(peaks) & np.isfinite(intensities)
            peaks = peaks[mask]
            intensities = intensities[mask]

            if peaks.size == 0:
                continue

            # 4.3 按强度选 Top-K 碎片（最多 100，且保证前体+碎片 <= maxlen）
            if peaks.size > max_frag:
                idx = np.argpartition(intensities, -max_frag)[-max_frag:]
                # 选完后按 m/z 升序排序（也可按强度降序，看你需求）
                order = np.argsort(peaks[idx])
                idx = idx[order]
                peaks_sel = peaks[idx]
                intens_sel = intensities[idx]
            else:
                # 直接按 m/z 升序
                order = np.argsort(peaks)
                peaks_sel = peaks[order]
                intens_sel = intensities[order]

            # 4.4 构建 token 序列（前体在最前）
            peaks_str = ["%.2f" % p for p in peaks_sel]
            try:
                token_ids = [word2idx[precursor_str]] + [word2idx[p] for p in peaks_str]
            except KeyError:
                # 理论上不会发生（我们已限制到 [10, 999.99]），但以防万一
                continue

            # 4.5 强度：在最前 prepend 2，并按你原有逻辑整体归一化
            intens_seq = np.hstack((2.0, intens_sel))
            max_intensity = float(np.max(intens_seq)) if intens_seq.size else 1.0
            if max_intensity != 0.0:
                intens_seq = intens_seq / max_intensity

            # 4.6 Pad 或截断到 maxlen（双序列严格对齐）
            if len(token_ids) > maxlen:
                token_ids = token_ids[:maxlen]
                intens_seq = intens_seq[:maxlen]

            n_pad = maxlen - len(token_ids)
            if n_pad > 0:
                token_ids += [word2idx['[PAD]']] * n_pad
                intens_seq = np.hstack([intens_seq, np.zeros(n_pad, dtype=float)])

            # 4.7 Save processed result
            ms_data[ms2_id] = {
                'mz': token_ids,
                'intensity': intens_seq.tolist(),
                'molecule_id': molecule_id
            }

        # 5) Return processed data and dictionary
        return ms_data, word2idx



    @staticmethod
    def load_external_test_dataset(
        external_data_dir,
        biotext_dir,
        paraphrase_dir,
        tokenizer,
        args,
        dataset_configs=None,
        **kwargs
    ):
        """
        加载并处理外部测试数据集（如HILIC和RPLC）
        
        参数:
        external_data_dir (str): 外部数据目录路径
        biotext_dir (str): BioText文本文件目录
        paraphrase_dir (str): Paraphrase文本文件目录
        tokenizer: 文本tokenizer
        args: 包含预处理参数的args对象（precursor_mode, precursor_value, n_workers等）
        dataset_configs (list): 数据集配置列表，每个配置包含name, ms2_file, meta_file
        **kwargs: 传递给MS2BioTextDataset构造函数的其他参数
        
        返回:
        tuple: (external_test_dataset, data_statistics)
            - external_test_dataset: MS2BioTextDataset实例
            - data_statistics: 包含数据统计信息的字典
        """
        import pickle
        import pandas as pd
        import os
        from pathlib import Path
        
        # 默认配置（HILIC和RPLC）
        if dataset_configs is None:
            dataset_configs = [
                {
                    'name': 'HILIC',
                    'ms2_file': os.path.join(external_data_dir, 'hilic_ms_data.pkl'),
                    'meta_file': os.path.join(external_data_dir, 'hilic_meta_data.csv'),
                },
                {
                    'name': 'RPLC', 
                    'ms2_file': os.path.join(external_data_dir, 'rplc_ms_data.pkl'),
                    'meta_file': os.path.join(external_data_dir, 'rplc_meta_data.csv'),
                }
            ]
        
        print("\n" + "="*60)
        print("加载外部测试数据集...")
        print("="*60)
        
        # 1. 加载所有数据集
        all_ms2_data = {}
        all_meta_data = []
        
        for config in dataset_configs:
            print(f"\n📁 加载 {config['name']} 数据集...")
            
            # 加载MS2数据
            with open(config['ms2_file'], 'rb') as f:
                ms2_data = pickle.load(f)
            
            # 加载Meta数据
            meta_data = pd.read_csv(config['meta_file'])
            
            print(f"  ✓ {config['name']}: {len(ms2_data)} 光谱, {len(meta_data)} meta")
            
            # 合并MS2
            all_ms2_data.update(ms2_data)
            all_meta_data.append(meta_data)
        
        # 2. 合并Meta数据（确保列对齐）
        if len(all_meta_data) > 1:
            all_cols = set()
            for df in all_meta_data:
                all_cols.update(df.columns)
            all_cols = sorted(all_cols)
            
            aligned_meta_data = []
            for df in all_meta_data:
                df = df.reindex(columns=all_cols, fill_value=None)
                aligned_meta_data.append(df)
            
            external_meta_data = pd.concat(aligned_meta_data, ignore_index=True)
        else:
            external_meta_data = all_meta_data[0]
        
        external_ms2_data = all_ms2_data
        
        print(f"\n✓ 合并后外部数据集: {len(external_ms2_data)} 光谱, {len(external_meta_data)} meta")
        
        # 3. 设置HMDB.ID为索引
        if 'HMDB.ID' in external_meta_data.columns:
            external_meta_data = external_meta_data.set_index('HMDB.ID')
            print(f"  已设置HMDB.ID为索引")
        
        # 4. 确保MS2数据格式正确（添加molecule_id字段）
        print("\n🔧 修正MS2数据格式...")
        for spectrum_id, spectrum_data in external_ms2_data.items():
            if 'molecule_id' not in spectrum_data:
                molecule_id = spectrum_id.split('_')[0]
                spectrum_data['molecule_id'] = molecule_id
        
        # 5. 获取所有unique的HMDB IDs
        unique_hmdb_ids = set()
        for spec_id in external_ms2_data.keys():
            hmdb_id = spec_id.split('_')[0]
            unique_hmdb_ids.add(hmdb_id)
        
        print(f"  外部数据集包含 {len(unique_hmdb_ids)} 个unique HMDB IDs")
        
        # 6. 加载对应的BioText数据
        print("\n📚 加载BioText数据...")
        external_biotext_data = {}
        missing_biotext = []
        
        biotext_dir = Path(biotext_dir)
        paraphrase_dir = Path(paraphrase_dir) if paraphrase_dir else None
        
        for hmdb_id in unique_hmdb_ids:
            # 处理异常的HMDB ID（如包含{}的）
            if '{}' in hmdb_id:
                missing_biotext.append(hmdb_id)
                continue
                
            biotext_file = biotext_dir / f"{hmdb_id}.txt"
            if biotext_file.exists():
                with open(biotext_file, 'r', encoding='utf-8') as f:
                    original_text = f.read().strip()
                
                # 加载paraphrase（如果有）
                paraphrases = []
                if paraphrase_dir:
                    paraphrase_file = paraphrase_dir / f"{hmdb_id}_paraphrase.txt"
                    if paraphrase_file.exists():
                        with open(paraphrase_file, 'r', encoding='utf-8') as pf:
                            content = pf.read()
                            versions = content.split("=== version")
                            for version in versions[1:]:
                                _, text = version.split("===", 1)
                                text = text.strip()
                                if text:
                                    paraphrases.append(text)
                
                external_biotext_data[hmdb_id] = {
                    'original': original_text,
                    'paraphrases': paraphrases
                }
            else:
                missing_biotext.append(hmdb_id)
        
        print(f"  ✓ 成功加载 {len(external_biotext_data)} 个BioText")
        print(f"  ✗ 缺失BioText: {len(missing_biotext)} 个")
        
        # 7. 处理缺失的biotext（使用drop方法）
        initial_ms2_count = len(external_ms2_data)
        external_ms2_data, _ = MS2BioTextDataset.missing_biotext_handling(
            external_ms2_data, 
            external_biotext_data, 
            method="drop"
        )
        print(f"  删除 {initial_ms2_count - len(external_ms2_data)} 条缺失biotext的光谱")
        
        # 8. 更新meta_data，只保留有MS2数据的条目
        remaining_hmdb_ids = set()
        for spectrum_id in external_ms2_data.keys():
            hmdb_id = spectrum_id.split('_')[0]
            remaining_hmdb_ids.add(hmdb_id)
        
        external_meta_data = external_meta_data[external_meta_data.index.isin(remaining_hmdb_ids)]
        
        # 9. 填充precursor数据
        print("\n⚙️ 填充precursor数据...")
        external_meta_data = MS2BioTextDataset.fill_precursor_data(
            external_meta_data,
            external_ms2_data
        )
        
        # 10. 预处理MS2数据（测试集不做数据增强）
        print("\n🚀 预处理MS2数据（不进行数据增强）...")
        external_processed_ms2, external_word2idx = MS2BioTextDataset.preprocess_ms2_data_positive_only_parallel(
            external_ms2_data,
            external_meta_data,
            n_workers=getattr(args, 'n_workers', 4),
            precursor_mode=getattr(args, 'precursor_mode', 'auto'),
            precursor_value=getattr(args, 'precursor_value', 2.0)
        )
        
        # 11. 统计信息
        data_statistics = {
            'original_ms2_count': initial_ms2_count,
            'processed_ms2_count': len(external_processed_ms2),
            'meta_count': len(external_meta_data),
            'biotext_count': len(external_biotext_data),
            'unique_molecules': len(remaining_hmdb_ids),
            'vocab_size': len(external_word2idx),
            'datasets': [config['name'] for config in dataset_configs]
        }
        
        print("\n" + "="*60)
        print("📊 外部测试数据集最终统计")
        print("="*60)
        print(f"  原始MS2光谱数: {data_statistics['original_ms2_count']}")
        print(f"  处理后MS2光谱数: {data_statistics['processed_ms2_count']}")
        print(f"  Meta记录数: {data_statistics['meta_count']}")
        print(f"  BioText数: {data_statistics['biotext_count']}")
        print(f"  Unique分子数: {data_statistics['unique_molecules']}")
        print(f"  词汇表大小: {data_statistics['vocab_size']}")
        print(f"  数据集来源: {', '.join(data_statistics['datasets'])}")
        
        # 12. 创建Dataset实例
        print("\n🎯 创建外部测试Dataset实例...")
        external_test_dataset = MS2BioTextDataset(
            ms2_data=external_processed_ms2,
            meta_data=external_meta_data,
            biotext_data=external_biotext_data,
            tokenizer=tokenizer,
            use_paraphrase=False,
            **kwargs
        )
        
        print(f"✅ 外部测试Dataset创建成功！大小: {len(external_test_dataset)}")
        
        return external_test_dataset, data_statistics

    
    # @staticmethod
    # def create_train_test_datasets_from_file(
    #     data_dir, 
    #     ms2_data, 
    #     meta_data, 
    #     biotext_data, 
    #     tokenizer, 
    #     test_size=0.2, 
    #     random_state=42, 
    #     use_paraphrase = False,
    #     **kwargs
    # ):
    #     """
    #     Split data into training and test sets based on molecule IDs.
    #     This version checks whether a local file with pre-defined splits exists.
    #     If it exists, it loads the split; otherwise, it creates the split and saves it to file.
    #     (Enhanced to handle empty or corrupted JSON files gracefully)

    #     Parameters:
    #     data_dir (str or Path): Path to the directory containing the split file (molecule_split.json).
    #     ms2_data (dict): Full MS2 data dictionary.
    #     meta_data (pd.DataFrame): Metadata dataframe indexed by molecule ID.
    #     biotext_data (dict): Full BioText data dictionary.
    #     tokenizer: Tokenizer used for initializing the Dataset.
    #     test_size (float): Proportion of test set (used only when creating the split).
    #     random_state (int): Random seed (used only when creating the split).
    #     **kwargs: Other arguments passed to the MS2BioTextDataset constructor.

    #     Returns:
    #     tuple: (train_dataset, test_dataset), two MS2BioTextDataset instances.
    #     """
    #     print("Preparing training and test datasets (with file persistence)...")

    #     # --- 1. Define split file path ---
    #     data_dir = Path(data_dir)
    #     split_file_path = data_dir / 'molecule_split.json'

    #     # --- 2. Load existing split file if valid; otherwise create new split ---
    #     train_mol_ids, test_mol_ids = None, None

    #     if split_file_path.exists() and split_file_path.stat().st_size > 0:
    #         print(f"✅ Found existing split file: {split_file_path}")
    #         print("    Loading molecule IDs from file...")
    #         try:
    #             with open(split_file_path, 'r', encoding='utf-8') as f:
    #                 split_ids = json.load(f)
    #                 train_mol_ids = split_ids['train_ids']
    #                 test_mol_ids = split_ids['test_ids']
    #         except json.JSONDecodeError:
    #             print(f"    ⚠️ Warning: File '{split_file_path}' exists but could not be parsed (may be empty or corrupted). Will recreate.")
    #         except KeyError:
    #             print(f"    ⚠️ Warning: File '{split_file_path}' has invalid format (missing 'train_ids' or 'test_ids'). Will recreate.")

    #     if train_mol_ids is None or test_mol_ids is None:
    #         print(f"⚠️ No valid split file found or could not be loaded. Creating a new split...")
    #         valid_molecule_ids = set(item['molecule_id'] for item in ms2_data.values())
    #         all_molecule_ids = [mol_id for mol_id in meta_data.index.unique() if mol_id in valid_molecule_ids]
    #         print(f"    Found {len(all_molecule_ids)} unique molecules for splitting.")

    #         train_mol_ids, test_mol_ids = train_test_split(
    #             all_molecule_ids,
    #             test_size=test_size,
    #             random_state=random_state
    #         )

    #         print(f"    Saving new split to: {split_file_path}")
    #         split_data_to_save = {'train_ids': train_mol_ids, 'test_ids': test_mol_ids}
    #         split_file_path.parent.mkdir(parents=True, exist_ok=True)
    #         with open(split_file_path, 'w', encoding='utf-8') as f:
    #             json.dump(split_data_to_save, f, indent=4)
    #         print("    Split file saved successfully.")


    #     # --- 3. Filter data sources by ID lists ---
    #     train_mol_ids_set = set(train_mol_ids)
    #     test_mol_ids_set = set(test_mol_ids)

    #     train_meta_data = meta_data[meta_data.index.isin(train_mol_ids_set)]
    #     test_meta_data = meta_data[meta_data.index.isin(test_mol_ids_set)]

    #     train_biotext_data = {mol_id: text for mol_id, text in biotext_data.items() if mol_id in train_mol_ids_set}
    #     test_biotext_data = {mol_id: text for mol_id, text in biotext_data.items() if mol_id in test_mol_ids_set}
    #     train_ms2_data = {ms2_id: info for ms2_id, info in ms2_data.items() if info['molecule_id'] in train_mol_ids_set}
    #     test_ms2_data = {ms2_id: info for ms2_id, info in ms2_data.items() if info['molecule_id'] in test_mol_ids_set}

    #     print(f"Data filtering completed:")

    #     # --- 4. Create Dataset instances ---
    #     print("Creating training Dataset instance...")
    #     train_dataset = MS2BioTextDataset(
    #         ms2_data=train_ms2_data, 
    #         meta_data=train_meta_data, 
    #         biotext_data=train_biotext_data, 
    #         tokenizer=tokenizer, 
    #         use_paraphrase = use_paraphrase,
    #         **kwargs
    #     )

    #     print("Creating test Dataset instance...")
    #     test_dataset = MS2BioTextDataset(
    #         ms2_data=test_ms2_data, 
    #         meta_data=test_meta_data, 
    #         biotext_data=test_biotext_data, 
    #         tokenizer=tokenizer, 
    #         use_paraphrase=False,
    #         **kwargs
    #     )

    #     return train_dataset, test_dataset




    # @staticmethod
    # def create_train_test_datasets_from_file(
    #     data_dir, 
    #     ms2_data, 
    #     meta_data, 
    #     biotext_data, 
    #     tokenizer, 
    #     test_size=0.2,  
    #     use_paraphrase = False,
    #     **kwargs
    # ):
    #     """
    #     Split data into training and test sets based on MS2 spectra within each molecule.
    #     For each molecule with multiple MS2 spectra, one MS2 is reserved for testing,
    #     and the rest are used for training. Molecules with only one MS2 spectrum are
    #     only included in the training set.
        
    #     Parameters:
    #     data_dir (str or Path): Path to the directory containing the split file (ms2_split.json).
    #     ms2_data (dict): Full MS2 data dictionary {ms2_id: {molecule_id: ..., ...}}.
    #     meta_data (pd.DataFrame): Metadata dataframe indexed by molecule ID.
    #     biotext_data (dict): Full BioText data dictionary {molecule_id: text}.
    #     tokenizer: Tokenizer used for initializing the Dataset.
    #     test_size (float): Deprecated in this version (kept for compatibility).
    #     random_state (int): Random seed for reproducible splits.
    #     **kwargs: Other arguments passed to the MS2BioTextDataset constructor.

    #     Returns:
    #     tuple: (train_dataset, test_dataset), two MS2BioTextDataset instances.
    #     """
    #     print("Preparing training and test datasets with per-molecule MS2 splitting...")
        
    #     # --- 1. Define split file path ---
    #     data_dir = Path(data_dir)
    #     split_file_path = data_dir / 'ms2_split.json'  # 改名以区分新的划分方式
        
    #     # --- 2. Load existing split file if valid; otherwise create new split ---
    #     train_ms2_ids, test_ms2_ids = None, None
        
    #     if split_file_path.exists() and split_file_path.stat().st_size > 0:
    #         print(f"✅ Found existing split file: {split_file_path}")
    #         print("    Loading MS2 IDs from file...")
    #         try:
    #             with open(split_file_path, 'r', encoding='utf-8') as f:
    #                 split_ids = json.load(f)
    #                 train_ms2_ids = split_ids['train_ms2_ids']
    #                 test_ms2_ids = split_ids['test_ms2_ids']
    #         except json.JSONDecodeError:
    #             print(f"    ⚠️ Warning: File '{split_file_path}' exists but could not be parsed. Will recreate.")
    #         except KeyError:
    #             print(f"    ⚠️ Warning: File '{split_file_path}' has invalid format. Will recreate.")
        
    #     if train_ms2_ids is None or test_ms2_ids is None:
    #         print(f"⚠️ No valid split file found. Creating a new MS2-level split...")
            
    #         # 按分子ID分组MS2谱图
    #         molecule_to_ms2 = {}
    #         for ms2_id, ms2_info in ms2_data.items():
    #             mol_id = ms2_info['molecule_id']
    #             if mol_id not in molecule_to_ms2:
    #                 molecule_to_ms2[mol_id] = []
    #             molecule_to_ms2[mol_id].append(ms2_id)
            
    #         # 统计信息
    #         single_ms2_molecules = []
    #         multi_ms2_molecules = []
    #         for mol_id, ms2_list in molecule_to_ms2.items():
    #             if len(ms2_list) == 1:
    #                 single_ms2_molecules.append(mol_id)
    #             else:
    #                 multi_ms2_molecules.append(mol_id)
            
    #         print(f"    Found {len(single_ms2_molecules)} molecules with single MS2 spectrum")
    #         print(f"    Found {len(multi_ms2_molecules)} molecules with multiple MS2 spectra")
            
            
    #         train_ms2_ids = []
    #         test_ms2_ids = []
            
    #         # 处理只有一个MS2的分子：全部放入训练集
    #         for mol_id in single_ms2_molecules:
    #             train_ms2_ids.extend(molecule_to_ms2[mol_id])
            
    #         # 处理有多个MS2的分子：随机选择一个作为测试集，其余作为训练集
    #         for mol_id in multi_ms2_molecules:
    #             ms2_list = molecule_to_ms2[mol_id]
    #             # 随机选择一个MS2作为测试集
    #             test_ms2_id = np.random.choice(ms2_list)
    #             test_ms2_ids.append(test_ms2_id)
    #             # 其余的作为训练集
    #             train_ms2_ids.extend([ms2_id for ms2_id in ms2_list if ms2_id != test_ms2_id])
            
    #         print(f"    Split results: {len(train_ms2_ids)} training MS2, {len(test_ms2_ids)} test MS2")
            
    #         # 保存划分结果
    #         print(f"    Saving new split to: {split_file_path}")
    #         split_data_to_save = {
    #             'train_ms2_ids': train_ms2_ids, 
    #             'test_ms2_ids': test_ms2_ids
    #         }
    #         split_file_path.parent.mkdir(parents=True, exist_ok=True)
    #         with open(split_file_path, 'w', encoding='utf-8') as f:
    #             json.dump(split_data_to_save, f, indent=4)
    #         print("    Split file saved successfully.")
        
    #     # --- 3. Filter data sources by MS2 ID lists ---
    #     train_ms2_ids_set = set(train_ms2_ids)
    #     test_ms2_ids_set = set(test_ms2_ids)
        
    #     # 过滤MS2数据
    #     train_ms2_data = {ms2_id: info for ms2_id, info in ms2_data.items() if ms2_id in train_ms2_ids_set}
    #     test_ms2_data = {ms2_id: info for ms2_id, info in ms2_data.items() if ms2_id in test_ms2_ids_set}
        
    #     # 获取涉及的分子ID
    #     train_molecule_ids = set(info['molecule_id'] for info in train_ms2_data.values())
    #     test_molecule_ids = set(info['molecule_id'] for info in test_ms2_data.values())
        
    #     # 过滤元数据和biotext数据
    #     # 注意：训练集和测试集可能包含相同的分子ID（因为同一分子的不同MS2可能分布在训练集和测试集中）
    #     train_meta_data = meta_data[meta_data.index.isin(train_molecule_ids)]
    #     test_meta_data = meta_data[meta_data.index.isin(test_molecule_ids)]
        
    #     train_biotext_data = {mol_id: text for mol_id, text in biotext_data.items() if mol_id in train_molecule_ids}
    #     test_biotext_data = {mol_id: text for mol_id, text in biotext_data.items() if mol_id in test_molecule_ids}
        
    #     print(f"Data filtering completed:")
    #     print(f"    Training: {len(train_ms2_data)} MS2 spectra from {len(train_molecule_ids)} molecules")
    #     print(f"    Test: {len(test_ms2_data)} MS2 spectra from {len(test_molecule_ids)} molecules")
        
    #     # --- 4. Create Dataset instances ---
    #     print("Creating training Dataset instance...")
    #     train_dataset = MS2BioTextDataset(
    #         ms2_data=train_ms2_data, 
    #         meta_data=train_meta_data, 
    #         biotext_data=train_biotext_data, 
    #         tokenizer=tokenizer, 
    #         use_paraphrase=use_paraphrase,
    #         **kwargs
    #     )
        
    #     print("Creating test Dataset instance...")
    #     test_dataset = MS2BioTextDataset(
    #         ms2_data=test_ms2_data, 
    #         meta_data=test_meta_data, 
    #         biotext_data=test_biotext_data, 
    #         tokenizer=tokenizer, 
    #         use_paraphrase=False,
    #         **kwargs
    #     )
        
    #     return train_dataset, test_dataset

    @staticmethod
    def filter_shared_texts(biotext_data, max_sharing_molecules=5):
        """
        删除被过多molecule共享的text
        
        Args:
            biotext_data: {molecule_id: [{'type': ..., 'text': ...}, ...]}
            max_sharing_molecules: text最多可以被多少个molecule共享
        
        Returns:
            filtered_biotext_data: 清洗后的数据
            stats: 统计信息
        """
        from collections import defaultdict
        
        print(f"\n=== Filtering shared texts (max_sharing={max_sharing_molecules}) ===")
        
        # 1. 构建text -> molecules的倒排索引
        text_to_molecules = defaultdict(set)
        
        for mol_id, entry in biotext_data.items():
            texts = []
            if isinstance(entry, list):
                texts = [record['text'] for record in entry]
            elif isinstance(entry, dict):
                texts = [entry.get("original", "")] + entry.get("paraphrases", [])
            elif isinstance(entry, str):
                texts = [entry]
            
            for text in texts:
                if text:  # 避免空字符串
                    text_to_molecules[text].add(mol_id)
        
        # 2. 找出需要删除的高频text
        texts_to_remove = set()
        for text, molecules in text_to_molecules.items():
            if len(molecules) > max_sharing_molecules:
                texts_to_remove.add(text)
        
        print(f"Found {len(texts_to_remove)} texts shared by >{max_sharing_molecules} molecules")
        
        # 3. 从每个molecule的候选text中删除这些高频text
        filtered_biotext_data = {}
        total_removed = 0
        molecules_with_no_text = []
        
        for mol_id, entry in biotext_data.items():
            if isinstance(entry, list):
                # 新格式：列表
                filtered_entry = [record for record in entry 
                                if record['text'] not in texts_to_remove]
                
                if filtered_entry:
                    filtered_biotext_data[mol_id] = filtered_entry
                else:
                    molecules_with_no_text.append(mol_id)
                
                total_removed += len(entry) - len(filtered_entry)
                
            elif isinstance(entry, dict):
                # 旧格式：字典
                original = entry.get("original", "")
                paraphrases = entry.get("paraphrases", [])
                
                filtered_paraphrases = [p for p in paraphrases if p not in texts_to_remove]
                
                # 如果original也被删除了，用第一个paraphrase作为original
                if original in texts_to_remove:
                    if filtered_paraphrases:
                        original = filtered_paraphrases[0]
                        filtered_paraphrases = filtered_paraphrases[1:]
                    else:
                        molecules_with_no_text.append(mol_id)
                        continue
                
                filtered_biotext_data[mol_id] = {
                    'original': original,
                    'paraphrases': filtered_paraphrases
                }
                
                original_count = 1 if entry.get("original", "") not in texts_to_remove else 0
                total_removed += (len(paraphrases) - len(filtered_paraphrases) + 
                                (1 - original_count))
                
            elif isinstance(entry, str):
                # 字符串格式
                if entry not in texts_to_remove:
                    filtered_biotext_data[mol_id] = entry
                else:
                    molecules_with_no_text.append(mol_id)
        
        # 4. 统计信息
        print(f"Statistics:")
        print(f"  Total text entries removed: {total_removed}")
        print(f"  Molecules before filtering: {len(biotext_data)}")
        print(f"  Molecules after filtering: {len(filtered_biotext_data)}")
        print(f"  Molecules with no text left: {len(molecules_with_no_text)}")
        
        if molecules_with_no_text:
            print(f"  Warning: {len(molecules_with_no_text)} molecules lost all texts!")
            print(f"  First 5: {molecules_with_no_text[:5]}")
        
        # 5. 验证过滤效果
        text_to_molecules_after = defaultdict(set)
        for mol_id, entry in filtered_biotext_data.items():
            texts = []
            if isinstance(entry, list):
                texts = [record['text'] for record in entry]
            elif isinstance(entry, dict):
                texts = [entry.get("original", "")] + entry.get("paraphrases", [])
            elif isinstance(entry, str):
                texts = [entry]
            
            for text in texts:
                if text:
                    text_to_molecules_after[text].add(mol_id)
        
        max_sharing_after = max(len(mols) for mols in text_to_molecules_after.values()) if text_to_molecules_after else 0
        print(f"  Max molecules sharing one text after filtering: {max_sharing_after}")
        
        return filtered_biotext_data, {
            'removed_texts': len(texts_to_remove),
            'removed_entries': total_removed,
            'molecules_no_text': len(molecules_with_no_text),
            'max_sharing_after': max_sharing_after
        }



    @staticmethod
    def create_train_test_datasets_from_file(
        data_dir, 
        ms2_data, 
        meta_data, 
        biotext_data, 
        tokenizer, 
        word2idx,      
        args,         
        test_size=0.2,  
        use_paraphrase = False,
        **kwargs
    ):
        """
        Split data into training and test sets.
        If a split file exists, use its test_ms2_ids as the test set,
        and use ALL OTHER MS2 spectra (including any new data) as the training set.
        If no split file exists, create a new split following the original logic.
        
        Parameters:
        data_dir (str or Path): Path to the directory containing the split file (ms2_split.json).
        ms2_data (dict): Full MS2 data dictionary {ms2_id: {molecule_id: ..., ...}}.
        meta_data (pd.DataFrame): Metadata dataframe indexed by molecule ID.
        biotext_data (dict): Full BioText data dictionary {molecule_id: text}.
        tokenizer: Tokenizer used for initializing the Dataset.
        test_size (float): Deprecated in this version (kept for compatibility).
        **kwargs: Other arguments passed to the MS2BioTextDataset constructor.

        Returns:
        tuple: (train_dataset, test_dataset), two MS2BioTextDataset instances.
        """
        print("Preparing training and test datasets with per-molecule MS2 splitting...")
        
        # --- 1. Define split file path ---
        data_dir = Path(data_dir)
        split_file_path = data_dir / 'ms2_split.json'
        
        # --- 2. Load existing split file if valid; otherwise create new split ---
        test_ms2_ids = None
        
        if split_file_path.exists() and split_file_path.stat().st_size > 0:
            print(f"✅ Found existing split file: {split_file_path}")
            print("    Loading test MS2 IDs from file...")
            try:
                with open(split_file_path, 'r', encoding='utf-8') as f:
                    split_ids = json.load(f)
                    test_ms2_ids = split_ids['test_ms2_ids']
                    print(f"    Loaded {len(test_ms2_ids)} test MS2 IDs from existing split.")
            except json.JSONDecodeError:
                print(f"    ⚠️ Warning: File '{split_file_path}' exists but could not be parsed. Will recreate.")
            except KeyError:
                print(f"    ⚠️ Warning: File '{split_file_path}' has invalid format. Will recreate.")
        
        if test_ms2_ids is None:
            print(f"⚠️ No valid split file found. Creating a new MS2-level split...")
            
            # 按分子ID分组MS2谱图
            molecule_to_ms2 = {}
            for ms2_id, ms2_info in ms2_data.items():
                mol_id = ms2_info['molecule_id']
                if mol_id not in molecule_to_ms2:
                    molecule_to_ms2[mol_id] = []
                molecule_to_ms2[mol_id].append(ms2_id)
            
            # 统计信息
            single_ms2_molecules = []
            multi_ms2_molecules = []
            for mol_id, ms2_list in molecule_to_ms2.items():
                if len(ms2_list) == 1:
                    single_ms2_molecules.append(mol_id)
                else:
                    multi_ms2_molecules.append(mol_id)
            
            print(f"    Found {len(single_ms2_molecules)} molecules with single MS2 spectrum")
            print(f"    Found {len(multi_ms2_molecules)} molecules with multiple MS2 spectra")
            
            test_ms2_ids = []
            
            # 处理有多个MS2的分子：随机选择一个作为测试集
            for mol_id in multi_ms2_molecules:
                ms2_list = molecule_to_ms2[mol_id]
                # 随机选择一个MS2作为测试集
                test_ms2_id = np.random.choice(ms2_list)
                test_ms2_ids.append(test_ms2_id)
            
            print(f"    Created new test set with {len(test_ms2_ids)} MS2 spectra")
            
            # 保存划分结果（只保存test_ms2_ids，train会动态计算）
            print(f"    Saving new split to: {split_file_path}")
            split_data_to_save = {
                'test_ms2_ids': test_ms2_ids,
                'note': 'Training set uses all MS2 IDs not in test_ms2_ids'
            }
            split_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(split_file_path, 'w', encoding='utf-8') as f:
                json.dump(split_data_to_save, f, indent=4)
            print("    Split file saved successfully.")
        
        # --- 3. Create train set: ALL MS2 IDs except those in test set ---
        test_ms2_ids_set = set(test_ms2_ids)
        all_ms2_ids = set(ms2_data.keys())
        train_ms2_ids_set = all_ms2_ids - test_ms2_ids_set
        
        print(f"\n📊 Dataset Statistics:")
        print(f"    Total MS2 spectra: {len(all_ms2_ids)}")
        print(f"    Test MS2 spectra: {len(test_ms2_ids_set)}")
        print(f"    Training MS2 spectra: {len(train_ms2_ids_set)}")
        
        # 过滤MS2数据
        train_ms2_data = {ms2_id: info for ms2_id, info in ms2_data.items() if ms2_id in train_ms2_ids_set}
        test_ms2_data = {ms2_id: info for ms2_id, info in ms2_data.items() if ms2_id in test_ms2_ids_set}
        
        # 获取涉及的分子ID
        train_molecule_ids = set(info['molecule_id'] for info in train_ms2_data.values())
        test_molecule_ids = set(info['molecule_id'] for info in test_ms2_data.values())
        
        # 过滤元数据和biotext数据
        train_meta_data = meta_data[meta_data.index.isin(train_molecule_ids)]
        test_meta_data = meta_data[meta_data.index.isin(test_molecule_ids)]
        
        train_biotext_data = {mol_id: text for mol_id, text in biotext_data.items() if mol_id in train_molecule_ids}
        test_biotext_data = {mol_id: text for mol_id, text in biotext_data.items() if mol_id in test_molecule_ids}
        
        print(f"\nData filtering completed:")
        print(f"    Training: {len(train_ms2_data)} MS2 spectra from {len(train_molecule_ids)} molecules")
        print(f"    Test: {len(test_ms2_data)} MS2 spectra from {len(test_molecule_ids)} molecules")
        
        # --- 4. Create Dataset instances ---
        print("\nCreating training Dataset instance...")
        train_dataset = MS2BioTextDataset(
            ms2_data=train_ms2_data, 
            meta_data=train_meta_data, 
            biotext_data=train_biotext_data, 
            tokenizer=tokenizer, 
            use_paraphrase=use_paraphrase,
            word2idx=word2idx,     
            args=args,             
            split='train',         
            **kwargs
        )
        
        print("Creating test Dataset instance...")
        test_dataset = MS2BioTextDataset(
            ms2_data=test_ms2_data, 
            meta_data=test_meta_data, 
            biotext_data=test_biotext_data, 
            tokenizer=tokenizer, 
            use_paraphrase=False,
            word2idx=word2idx,  
            args=args,            
            split='test',         
            **kwargs
        )
        
        return train_dataset, test_dataset
    



# 在文件顶部，import 语句之后，类定义之前

def _augment_worker(batch_data):
    """
    全局 worker 函数，用于数据增强
    batch_data: (batch, filter_threshold, noise_ratio, noise_intensity_range, augment_multiplier)
    """
    import numpy as np
    # ⚠️ 关键修改：不要导入 MS2BioTextDataset，直接在这里定义需要的函数
    
    batch, filter_threshold, noise_ratio, noise_intensity_range, augment_multiplier = batch_data
    
    # 将 filter_low_intensity_peaks 和 add_noise_peaks 的逻辑直接复制到这里
    def filter_low_intensity_peaks(peaks, intensities, threshold):
        """过滤低强度峰"""
        if not peaks or not intensities:
            return peaks, intensities
        
        max_intensity = max(intensities)
        if max_intensity == 0:
            return peaks, intensities
        
        filtered_peaks = []
        filtered_intensities = []
        for mz, intensity in zip(peaks, intensities):
            if intensity / max_intensity >= threshold:
                filtered_peaks.append(mz)
                filtered_intensities.append(intensity)
        
        return filtered_peaks, filtered_intensities
    
    def add_noise_peaks(peaks, intensities, noise_ratio, noise_intensity_range):
        """添加噪声峰"""
        import random
        
        if not peaks:
            return peaks, intensities
        
        n_noise = int(len(peaks) * noise_ratio)
        if n_noise == 0:
            return peaks, intensities
        
        # 获取m/z范围
        min_mz = min(peaks)
        max_mz = max(peaks)
        max_intensity = max(intensities) if intensities else 1.0
        
        # 生成噪声峰
        for _ in range(n_noise):
            # 随机m/z（避免与现有峰重复）
            noise_mz = random.uniform(min_mz, max_mz)
            # 随机低强度
            noise_intensity = random.uniform(
                noise_intensity_range[0] * max_intensity,
                noise_intensity_range[1] * max_intensity
            )
            
            peaks.append(noise_mz)
            intensities.append(noise_intensity)
        
        # 按m/z排序
        sorted_pairs = sorted(zip(peaks, intensities), key=lambda x: x[0])
        peaks = [p for p, _ in sorted_pairs]
        intensities = [i for _, i in sorted_pairs]
        
        return peaks, intensities
    
    # 处理逻辑
    result = {}
    for ms2_id, info in batch:
        molecule_id = info.get('molecule_id')
        peaks_original = info['mz'] if isinstance(info['mz'], list) else list(info['mz'])
        intensities_original = info['intensity'] if isinstance(info['intensity'], list) else list(info['intensity'])
        
        # 过滤
        if filter_threshold:
            peaks_original, intensities_original = filter_low_intensity_peaks(
                peaks_original, intensities_original, filter_threshold
            )
        
        # 原始版本
        result[ms2_id] = {
            'mz': peaks_original,
            'intensity': intensities_original,
            'molecule_id': molecule_id
        }
        
        # 增强版本
        for aug_idx in range(1, augment_multiplier):
            peaks_aug, intensities_aug = add_noise_peaks(
                peaks_original.copy(), intensities_original.copy(),
                noise_ratio,
                noise_intensity_range
            )
            result[f"{ms2_id}_aug{aug_idx}"] = {
                'mz': peaks_aug,
                'intensity': intensities_aug,
                'molecule_id': molecule_id
            }
    return result

def _preprocess_worker(batch_data):
    """
    全局 worker 函数，用于多进程处理
    batch_data: (batch, word2idx, meta_data_processed, maxlen, max_frag, min_peaks, precursor_mode, precursor_value)
    """
    import numpy as np
    import pandas as pd
    
    batch, word2idx, meta_data_processed, maxlen, max_frag, min_peaks, precursor_mode, precursor_value = batch_data
    
    result = {}
    stats = {'kept': 0, 'filtered': 0}
    
    for ms2_id, info in batch:
        # 基础检查
        if not info.get('mz'):
            stats['filtered'] += 1
            continue
        
        # 转为 numpy 数组
        peaks = np.asarray(info['mz'], dtype=float)
        intensities = np.asarray(info['intensity'], dtype=float)
        molecule_id = info.get('molecule_id', None)
        
        # 长度对齐检查
        if peaks.shape[0] != intensities.shape[0]:
            n = min(len(peaks), len(intensities))
            peaks = peaks[:n]
            intensities = intensities[:n]
        
        # 文件名对应行用于极性判断
        specific_row = meta_data_processed.loc[meta_data_processed["file_name"] == ms2_id] if "file_name" in meta_data_processed.columns else pd.DataFrame()
        if specific_row.empty:
            if molecule_id is not None:
                if 'HMDB.ID' in meta_data_processed.columns:
                    specific_row = meta_data_processed.loc[meta_data_processed['HMDB.ID'] == molecule_id]
                else:
                    specific_row = meta_data_processed.loc[meta_data_processed.index == molecule_id]
        
        if specific_row.empty:
            stats['filtered'] += 1
            continue
        
        # 只保留正离子
        pol = str(specific_row["Polarity"].values[0]).lower().strip() if "Polarity" in specific_row.columns else ""
        if pol != "positive":
            stats['filtered'] += 1
            continue
        
        # 获取precursor
        if 'HMDB.ID' in meta_data_processed.columns and (molecule_id is not None):
            row = meta_data_processed.loc[meta_data_processed['HMDB.ID'] == molecule_id]
        else:
            row = meta_data_processed.loc[meta_data_processed.index == molecule_id]
        
        if row.empty or ('precursor_mass' not in row.columns):
            stats['filtered'] += 1
            continue
        
        try:
            precursor_val = float(row['precursor_mass'].values[0])
        except Exception:
            stats['filtered'] += 1
            continue
        
        # 前体范围 [10, 1000)
        if pd.isna(precursor_val) or (precursor_val < 10.0) or (precursor_val >= 1000.0):
            stats['filtered'] += 1
            continue
        
        precursor_val = min(precursor_val, 999.99)
        precursor_str = "%.2f" % precursor_val
        
        # 过滤峰到 [10, 1000)
        mask = (peaks >= 10.0) & (peaks < 1000.0) & np.isfinite(peaks) & np.isfinite(intensities)
        peaks = peaks[mask]
        intensities = intensities[mask]
        
        if peaks.size == 0:
            stats['filtered'] += 1
            continue
        
        # 按强度选 Top-K 碎片
        if peaks.size > max_frag:
            idx = np.argpartition(intensities, -max_frag)[-max_frag:]
            order = np.argsort(peaks[idx])
            idx = idx[order]
            peaks_sel = peaks[idx]
            intens_sel = intensities[idx]
        else:
            order = np.argsort(peaks)
            peaks_sel = peaks[order]
            intens_sel = intensities[order]
        
        # 检查 min_peaks
        if peaks_sel.size < min_peaks:
            stats['filtered'] += 1
            continue
        
        # 构建 token 序列
        peaks_str = ["%.2f" % p for p in peaks_sel]
        try:
            token_ids = [word2idx[precursor_str]] + [word2idx[p] for p in peaks_str]
        except KeyError:
            stats['filtered'] += 1
            continue
        
        # ⭐ 根据 precursor_mode 选择处理方式
        if precursor_mode == 'scale_fixed':
            # 方案一：缩放fragments到固定值precursor_value（如20000），然后precursor添加2
            if np.max(intens_sel) > 0:
                intens_sel = intens_sel / np.max(intens_sel) * precursor_value
            intens_seq = np.hstack((2.0, intens_sel))
            # 整体归一化
            max_intensity = float(np.max(intens_seq))
            if max_intensity > 0:
                intens_seq = intens_seq / max_intensity
                
        elif precursor_mode == 'normalize_add':
            # 方案二：归一化fragments到1，添加precursor_value，再整体归一化
            if np.max(intens_sel) > 0:
                intens_sel = intens_sel / np.max(intens_sel)
            intens_seq = np.hstack((precursor_value, intens_sel))
            # 整体归一化
            max_intensity = float(np.max(intens_seq))
            if max_intensity > 0:
                intens_seq = intens_seq / max_intensity
        
        else:
            # 默认：原始MSBERT方式
            intens_seq = np.hstack((2.0, intens_sel))
            max_intensity = float(np.max(intens_seq))
            if max_intensity > 0:
                intens_seq = intens_seq / max_intensity
        
        # Pad 或截断到 maxlen
        if len(token_ids) > maxlen:
            token_ids = token_ids[:maxlen]
            intens_seq = intens_seq[:maxlen]
        
        n_pad = maxlen - len(token_ids)
        if n_pad > 0:
            token_ids += [word2idx['[PAD]']] * n_pad
            intens_seq = np.hstack([intens_seq, np.zeros(n_pad, dtype=float)])
        
        result[ms2_id] = {
            'mz': token_ids,
            'intensity': intens_seq.tolist(),
            'molecule_id': molecule_id
        }
        stats['kept'] += 1
    
    return result, stats




