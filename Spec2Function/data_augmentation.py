"""
MS2 data augmentation and preprocessing module

Features:
1. Noise filtering - remove low-intensity peaks
2. Noise augmentation - add random noise peaks
3. Intensity perturbation - add random noise to intensities
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union


class MS2DataAugmentation:
    """MS2 spectrum data augmentation class"""

    def __init__(
        self,
        # Noise filtering parameters
        filter_threshold: float = 0.0,  # 0 means no filtering

        # Noise augmentation parameters
        noise_augmentation: bool = False,
        noise_ratio_range: Tuple[float, float] = (0.3, 0.8),  # range of noise peak ratio to add
        noise_intensity_range: Tuple[float, float] = (0.001, 0.05),  # noise intensity range (relative)

        # Intensity perturbation parameters
        intensity_perturbation: bool = False,
        perturbation_std: float = 0.1,  # perturbation standard deviation

        # Augmentation probability
        augmentation_prob: float = 0.5,  # probability of applying augmentation

        # Other parameters
        seed: Optional[int] = None,
    ):
        """
        Initialize the data augmenter.

        Args:
            filter_threshold: filter threshold; peaks with relative intensity < this value are removed
            noise_augmentation: whether to enable noise augmentation
            noise_ratio_range: range of number of noise peaks to add (relative to original peak count)
            noise_intensity_range: intensity range for generated noise (relative to max intensity)
            intensity_perturbation: whether to enable intensity perturbation
            perturbation_std: standard deviation of intensity perturbation
            augmentation_prob: probability of applying augmentation
            seed: random seed
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
        Filter out low-intensity noise peaks.

        Args:
            mz: array of m/z values
            intensity: array of intensity values
            threshold: filter threshold (uses the value set at init if None)

        Returns:
            filtered_mz, filtered_intensity
        """
        if threshold is None:
            threshold = self.filter_threshold

        if threshold <= 0:
            # No filtering
            return np.array(mz), np.array(intensity)

        mz = np.array(mz)
        intensity = np.array(intensity)

        if len(intensity) == 0:
            return mz, intensity

        # Normalize
        max_int = np.max(intensity)
        if max_int == 0:
            return mz, intensity

        norm_int = intensity / max_int

        # Filter
        mask = norm_int >= threshold

        return mz[mask], intensity[mask]

    def add_noise_peaks(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        noise_ratio: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add random noise peaks.

        Args:
            mz: array of m/z values
            intensity: array of intensity values
            noise_ratio: noise ratio (randomly sampled from range if None)

        Returns:
            augmented_mz, augmented_intensity
        """
        if len(mz) == 0:
            return mz, intensity

        # Determine number of noise peaks
        if noise_ratio is None:
            noise_ratio = np.random.uniform(*self.noise_ratio_range)

        n_noise = int(len(mz) * noise_ratio)
        if n_noise == 0:
            return mz, intensity

        # Generate noise m/z values (random within spectrum range)
        mz_min, mz_max = np.min(mz), np.max(mz)
        noise_mz = np.random.uniform(mz_min, mz_max, n_noise)

        # Generate noise intensity values
        max_int = np.max(intensity)
        noise_int = np.random.uniform(
            self.noise_intensity_range[0] * max_int,
            self.noise_intensity_range[1] * max_int,
            n_noise
        )

        # Merge original peaks and noise
        aug_mz = np.concatenate([mz, noise_mz])
        aug_int = np.concatenate([intensity, noise_int])

        # Sort by m/z
        sort_idx = np.argsort(aug_mz)

        return aug_mz[sort_idx], aug_int[sort_idx]

    def perturb_intensity(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        noise_std: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random perturbation to intensities.

        Args:
            mz: array of m/z values
            intensity: array of intensity values
            noise_std: perturbation standard deviation (uses init value if None)

        Returns:
            mz, perturbed_intensity
        """
        if len(intensity) == 0:
            return mz, intensity

        if noise_std is None:
            noise_std = self.perturbation_std

        # Generate multiplicative noise
        # log-normal distribution feels more natural
        perturb = np.random.normal(1.0, noise_std, len(intensity))
        perturb = np.clip(perturb, 0.5, 1.5)  # clip perturbation range

        aug_intensity = intensity * perturb
        aug_intensity = np.clip(aug_intensity, 0, None)  # ensure non-negative

        return mz, aug_intensity

    def __call__(
        self,
        mz: Union[np.ndarray, list, torch.Tensor],
        intensity: Union[np.ndarray, list, torch.Tensor],
        training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the data augmentation pipeline.

        Args:
            mz: m/z values
            intensity: intensity values
            training: whether in training mode (augmentation applied only in training; evaluation only filters)

        Returns:
            augmented_mz, augmented_intensity
        """
        # Convert to numpy arrays
        if isinstance(mz, torch.Tensor):
            mz = mz.cpu().numpy()
        if isinstance(intensity, torch.Tensor):
            intensity = intensity.cpu().numpy()

        mz = np.array(mz)
        intensity = np.array(intensity)

        # Step 1: filter extremely low-intensity noise (applied in both training and evaluation)
        if self.filter_threshold > 0:
            mz, intensity = self.filter_noise_peaks(mz, intensity)

        # Step 2: data augmentation during training
        if training:
            # Randomly decide whether to apply augmentation
            if np.random.rand() < self.augmentation_prob:
                # Add noise peaks
                if self.noise_augmentation:
                    mz, intensity = self.add_noise_peaks(mz, intensity)

                # Intensity perturbation
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


# Convenience functions
def create_augmentation_pipeline(
    mode: str = 'none',
    filter_threshold: float = 0.01,
    noise_level: str = 'medium',
    **kwargs
) -> MS2DataAugmentation:
    """
    Create a predefined augmentation pipeline.

    Args:
        mode: augmentation mode
            - 'none': no augmentation, only filtering
            - 'light': light augmentation
            - 'medium': medium augmentation
            - 'heavy': heavy augmentation
        filter_threshold: filter threshold
        noise_level: noise level ('low', 'medium', 'high')
        **kwargs: other parameter overrides

    Returns:
        MS2DataAugmentation instance
    """
    # Predefined noise parameters
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

    # Predefined augmentation modes
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

    # Merge configs
    config = {
        'filter_threshold': filter_threshold,
        **mode_configs.get(mode, mode_configs['none']),
        **noise_configs.get(noise_level, noise_configs['medium']),
        **kwargs
    }

    return MS2DataAugmentation(**config)


# Test code
if __name__ == '__main__':
    # Create test data
    test_mz = np.array([50, 100, 150, 200, 250, 300])
    test_intensity = np.array([1000, 500, 100, 50, 20, 5])

    print("Original data:")
    print(f"  mz: {test_mz}")
    print(f"  intensity: {test_intensity}")
    print(f"  peak count: {len(test_mz)}")

    # Test filtering
    print("\n=== Test 1: Noise filtering (threshold=0.05) ===")
    aug1 = MS2DataAugmentation(filter_threshold=0.05)
    filtered_mz, filtered_int = aug1(test_mz, test_intensity, training=False)
    print(f"Peak count after filtering: {len(filtered_mz)}")
    print(f"  mz: {filtered_mz}")
    print(f"  intensity: {filtered_int}")

    # Test augmentation
    print("\n=== Test 2: Noise augmentation ===")
    aug2 = MS2DataAugmentation(
        filter_threshold=0.01,
        noise_augmentation=True,
        noise_ratio_range=(0.5, 0.5),  # fixed 50%
        augmentation_prob=1.0
    )
    aug_mz, aug_int = aug2(test_mz, test_intensity, training=True)
    print(f"Peak count after augmentation: {len(aug_mz)}")
    print(f"  added peaks: {len(aug_mz) - len(test_mz)}")

    # Test predefined pipeline
    print("\n=== Test 3: Predefined pipeline ===")
    for mode in ['none', 'light', 'medium', 'heavy']:
        aug = create_augmentation_pipeline(mode=mode, filter_threshold=0.01)
        result_mz, result_int = aug(test_mz, test_intensity, training=True)
        print(f"{mode:8s}: {len(test_mz)} -> {len(result_mz)} peaks")
