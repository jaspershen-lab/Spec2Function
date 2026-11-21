"""
MS2 数据增强和预处理模块

功能：
1. Noise过滤 - 去除低强度peaks
2. Noise增强 - 添加随机noise peaks
3. 强度扰动 - 对intensity添加随机噪声
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union


class MS2DataAugmentation:
    """MS2光谱数据增强类"""
    
    def __init__(
        self,
        # Noise过滤参数
        filter_threshold: float = 0.0,  # 0表示不过滤
        
        # Noise增强参数
        noise_augmentation: bool = False,
        noise_ratio_range: Tuple[float, float] = (0.3, 0.8),  # 添加noise的比例范围
        noise_intensity_range: Tuple[float, float] = (0.001, 0.05),  # noise强度范围(相对值)
        
        # 强度扰动参数
        intensity_perturbation: bool = False,
        perturbation_std: float = 0.1,  # 扰动标准差
        
        # 增强概率
        augmentation_prob: float = 0.5,  # 应用增强的概率
        
        # 其他参数
        seed: Optional[int] = None,
    ):
        """
        初始化数据增强器
        
        Args:
            filter_threshold: 过滤阈值，相对强度<该值的peaks会被移除
            noise_augmentation: 是否启用noise增强
            noise_ratio_range: 添加noise peaks的数量范围（相对原peaks数）
            noise_intensity_range: 生成noise的强度范围（相对最大强度）
            intensity_perturbation: 是否启用强度扰动
            perturbation_std: 强度扰动的标准差
            augmentation_prob: 应用增强的概率
            seed: 随机种子
        """
        self.filter_threshold = filter_threshold
        self.noise_augmentation = noise_augmentation
        self.noise_ratio_range = noise_ratio_range
        self.noise_intensity_range = noise_intensity_range
        self.intensity_perturbation = intensity_perturbation
        self.perturbation_std = perturbation_std
        self.augmentation_prob = augmentation_prob
        
        if seed is not None:
            np.random.seed(seed)
    
    def filter_noise_peaks(
        self, 
        mz: Union[np.ndarray, list], 
        intensity: Union[np.ndarray, list],
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        过滤低强度noise peaks
        
        Args:
            mz: m/z值数组
            intensity: 强度值数组
            threshold: 过滤阈值（如果为None则使用初始化时的值）
        
        Returns:
            filtered_mz, filtered_intensity
        """
        if threshold is None:
            threshold = self.filter_threshold
        
        if threshold <= 0:
            # 不过滤
            return np.array(mz), np.array(intensity)
        
        mz = np.array(mz)
        intensity = np.array(intensity)
        
        if len(intensity) == 0:
            return mz, intensity
        
        # 归一化
        max_int = np.max(intensity)
        if max_int == 0:
            return mz, intensity
        
        norm_int = intensity / max_int
        
        # 过滤
        mask = norm_int >= threshold
        
        return mz[mask], intensity[mask]
    
    def add_noise_peaks(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        noise_ratio: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        添加随机noise peaks
        
        Args:
            mz: m/z值数组
            intensity: 强度值数组
            noise_ratio: noise比例（如果为None则从范围随机采样）
        
        Returns:
            augmented_mz, augmented_intensity
        """
        if len(mz) == 0:
            return mz, intensity
        
        # 确定noise数量
        if noise_ratio is None:
            noise_ratio = np.random.uniform(*self.noise_ratio_range)
        
        n_noise = int(len(mz) * noise_ratio)
        if n_noise == 0:
            return mz, intensity
        
        # 生成noise的m/z值（在光谱范围内随机）
        mz_min, mz_max = np.min(mz), np.max(mz)
        noise_mz = np.random.uniform(mz_min, mz_max, n_noise)
        
        # 生成noise的强度值
        max_int = np.max(intensity)
        noise_int = np.random.uniform(
            self.noise_intensity_range[0] * max_int,
            self.noise_intensity_range[1] * max_int,
            n_noise
        )
        
        # 合并原始peaks和noise
        aug_mz = np.concatenate([mz, noise_mz])
        aug_int = np.concatenate([intensity, noise_int])
        
        # 按m/z排序
        sort_idx = np.argsort(aug_mz)
        
        return aug_mz[sort_idx], aug_int[sort_idx]
    
    def perturb_intensity(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        noise_std: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对强度添加随机扰动
        
        Args:
            mz: m/z值数组
            intensity: 强度值数组
            noise_std: 扰动标准差（如果为None则使用初始化值）
        
        Returns:
            mz, perturbed_intensity
        """
        if len(intensity) == 0:
            return mz, intensity
        
        if noise_std is None:
            noise_std = self.perturbation_std
        
        # 生成乘性噪声 (multiplicative noise)
        # 使用log-normal分布更自然
        perturb = np.random.normal(1.0, noise_std, len(intensity))
        perturb = np.clip(perturb, 0.5, 1.5)  # 限制扰动范围
        
        aug_intensity = intensity * perturb
        aug_intensity = np.clip(aug_intensity, 0, None)  # 确保非负
        
        return mz, aug_intensity
    
    def __call__(
        self,
        mz: Union[np.ndarray, list, torch.Tensor],
        intensity: Union[np.ndarray, list, torch.Tensor],
        training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用数据增强pipeline
        
        Args:
            mz: m/z值
            intensity: 强度值
            training: 是否为训练模式（训练时应用增强，评估时只过滤）
        
        Returns:
            augmented_mz, augmented_intensity
        """
        # 转换为numpy数组
        if isinstance(mz, torch.Tensor):
            mz = mz.cpu().numpy()
        if isinstance(intensity, torch.Tensor):
            intensity = intensity.cpu().numpy()
        
        mz = np.array(mz)
        intensity = np.array(intensity)
        
        # 步骤1: 过滤极低强度noise（训练和评估都做）
        if self.filter_threshold > 0:
            mz, intensity = self.filter_noise_peaks(mz, intensity)
        
        # 步骤2: 训练时的数据增强
        if training:
            # 随机决定是否应用增强
            if np.random.rand() < self.augmentation_prob:
                # 添加noise peaks
                if self.noise_augmentation:
                    mz, intensity = self.add_noise_peaks(mz, intensity)
                
                # 强度扰动
                if self.intensity_perturbation:
                    mz, intensity = self.perturb_intensity(mz, intensity)
        
        return mz, intensity
    
    def __repr__(self):
        return (
            f"MS2DataAugmentation(\n"
            f"  filter_threshold={self.filter_threshold},\n"
            f"  noise_augmentation={self.noise_augmentation},\n"
            f"  noise_ratio_range={self.noise_ratio_range},\n"
            f"  intensity_perturbation={self.intensity_perturbation},\n"
            f"  augmentation_prob={self.augmentation_prob}\n"
            f")"
        )


# 便捷函数
def create_augmentation_pipeline(
    mode: str = 'none',
    filter_threshold: float = 0.01,
    noise_level: str = 'medium',
    **kwargs
) -> MS2DataAugmentation:
    """
    创建预定义的增强pipeline
    
    Args:
        mode: 增强模式
            - 'none': 无增强，只过滤
            - 'light': 轻度增强
            - 'medium': 中度增强
            - 'heavy': 重度增强
        filter_threshold: 过滤阈值
        noise_level: noise水平 ('low', 'medium', 'high')
        **kwargs: 其他参数覆盖
    
    Returns:
        MS2DataAugmentation实例
    """
    # 预定义的noise参数
    noise_configs = {
        'low': {
            'noise_ratio_range': (0.2, 0.4),
            'noise_intensity_range': (0.001, 0.03),
        },
        'medium': {
            'noise_ratio_range': (0.3, 0.8),
            'noise_intensity_range': (0.001, 0.05),
        },
        'high': {
            'noise_ratio_range': (0.5, 1.2),
            'noise_intensity_range': (0.002, 0.08),
        }
    }
    
    # 预定义的增强模式
    mode_configs = {
        'none': {
            'noise_augmentation': False,
            'intensity_perturbation': False,
            'augmentation_prob': 0.0,
        },
        'light': {
            'noise_augmentation': True,
            'intensity_perturbation': False,
            'augmentation_prob': 0.3,
        },
        'medium': {
            'noise_augmentation': True,
            'intensity_perturbation': True,
            'perturbation_std': 0.1,
            'augmentation_prob': 0.5,
        },
        'heavy': {
            'noise_augmentation': True,
            'intensity_perturbation': True,
            'perturbation_std': 0.15,
            'augmentation_prob': 0.7,
        }
    }
    
    # 合并配置
    config = {
        'filter_threshold': filter_threshold,
        **mode_configs.get(mode, mode_configs['none']),
        **noise_configs.get(noise_level, noise_configs['medium']),
        **kwargs
    }
    
    return MS2DataAugmentation(**config)


# 测试代码
if __name__ == '__main__':
    # 创建测试数据
    test_mz = np.array([50, 100, 150, 200, 250, 300])
    test_intensity = np.array([1000, 500, 100, 50, 20, 5])
    
    print("原始数据:")
    print(f"  mz: {test_mz}")
    print(f"  intensity: {test_intensity}")
    print(f"  peaks数: {len(test_mz)}")
    
    # 测试过滤
    print("\n=== 测试1: Noise过滤 (threshold=0.05) ===")
    aug1 = MS2DataAugmentation(filter_threshold=0.05)
    filtered_mz, filtered_int = aug1(test_mz, test_intensity, training=False)
    print(f"过滤后peaks数: {len(filtered_mz)}")
    print(f"  mz: {filtered_mz}")
    print(f"  intensity: {filtered_int}")
    
    # 测试增强
    print("\n=== 测试2: Noise增强 ===")
    aug2 = MS2DataAugmentation(
        filter_threshold=0.01,
        noise_augmentation=True,
        noise_ratio_range=(0.5, 0.5),  # 固定50%
        augmentation_prob=1.0
    )
    aug_mz, aug_int = aug2(test_mz, test_intensity, training=True)
    print(f"增强后peaks数: {len(aug_mz)}")
    print(f"  新增peaks数: {len(aug_mz) - len(test_mz)}")
    
    # 测试预定义pipeline
    print("\n=== 测试3: 预定义Pipeline ===")
    for mode in ['none', 'light', 'medium', 'heavy']:
        aug = create_augmentation_pipeline(mode=mode, filter_threshold=0.01)
        result_mz, result_int = aug(test_mz, test_intensity, training=True)
        print(f"{mode:8s}: {len(test_mz)} -> {len(result_mz)} peaks")