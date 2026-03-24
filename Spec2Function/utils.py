# -*- coding: utf-8 -*-
"""
Utility functions for Spec2Function
"""
import numpy as np
import base64
from typing import List, Dict, Tuple


def parse_mgf(file_content: str) -> List[Dict]:
    """
    Parse an MGF file.

    Args:
        file_content: MGF file content as text.
    Returns:
        List of spectra, each containing:
        {
            'title': str,
            'precursor_mz': float,
            'charge': int,
            'mz': List[float],
            'intensity': List[float]
        }
    """
    spectra = []
    current_spectrum = None

    lines = file_content.strip().split('\n')

    for line in lines:
        line = line.strip()

        if line.startswith('BEGIN IONS'):
            current_spectrum = {
                'title': '',
                'precursor_mz': 0.0,
                'charge': 0,
                'mz': [],
                'intensity': []
            }

        elif line.startswith('END IONS'):
            if current_spectrum and len(current_spectrum['mz']) > 0:
                spectra.append(current_spectrum)
            current_spectrum = None

        elif current_spectrum is not None:
            if line.startswith('TITLE='):
                current_spectrum['title'] = line.split('=', 1)[1]

            elif line.startswith('PEPMASS='):
                current_spectrum['precursor_mz'] = float(line.split('=')[1].split()[0])

            elif line.startswith('CHARGE='):
                charge_str = line.split('=')[1].replace('+', '').replace('-', '')
                try:
                    current_spectrum['charge'] = int(charge_str)
                except:
                    current_spectrum['charge'] = 0

            elif line and not line.startswith(('TITLE', 'PEPMASS', 'CHARGE', 'RTINSECONDS', 'SCANS')):
                # Peak line: m/z intensity
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        current_spectrum['mz'].append(mz)
                        current_spectrum['intensity'].append(intensity)
                except:
                    pass

    return spectra


def parse_msp(file_content: str) -> List[Dict]:
    """
    Parse an MSP file.

    Args:
        file_content: MSP file content as text.
    Returns:
        List of spectra (same format as parse_mgf)
    """
    spectra = []
    current_spectrum = None
    num_peaks = 0
    peaks_read = 0

    lines = file_content.strip().split('\n')

    for line in lines:
        line = line.strip()

        if not line:
            if current_spectrum and len(current_spectrum['mz']) > 0:
                spectra.append(current_spectrum)
            current_spectrum = None
            peaks_read = 0
            continue

        if line.startswith('Name:'):
            current_spectrum = {
                'title': line.split(':', 1)[1].strip(),
                'precursor_mz': 0.0,
                'charge': 0,
                'mz': [],
                'intensity': []
            }

        elif current_spectrum is not None:
            if line.startswith('PrecursorMZ:') or line.startswith('PRECURSORMZ:'):
                current_spectrum['precursor_mz'] = float(line.split(':')[1].strip())

            elif line.startswith('Num peaks:') or line.startswith('Num Peaks:'):
                num_peaks = int(line.split(':')[1].strip())

            elif peaks_read < num_peaks:
                # Peak line
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        current_spectrum['mz'].append(mz)
                        current_spectrum['intensity'].append(intensity)
                        peaks_read += 1
                except:
                    pass

    # Handle the last spectrum
    if current_spectrum and len(current_spectrum['mz']) > 0:
        spectra.append(current_spectrum)

    return spectra


def parse_json_spectrum(json_data: Dict) -> Dict:
    peaks = json_data.get('peaks', [])
    precursor_mz = json_data.get('precursor_mz', 0.0)
    return {
        'title': 'Single Spectrum',
        'precursor_mz': precursor_mz,
        'charge': 0,
        'mz': [peak[0] for peak in peaks],
        'intensity': [peak[1] for peak in peaks]
    }


def decode_uploaded_file(contents: str, filename: str) -> Tuple[str, str]:
    """
    Decode a file uploaded via Dash Upload.
    Args:
        contents: Base64-encoded content string.
        filename: Original filename.
    Returns:
        (file_content, file_type)
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        file_content = decoded.decode('utf-8')
    except:
        file_content = decoded.decode('latin-1')

    file_type = filename.split('.')[-1].lower()

    return file_content, file_type


def preprocess_spectrum(mz: List[float], intensity: List[float], 
                        max_peaks: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess an MS2 spectrum.

    Args:
        mz: m/z values
        intensity: Intensity values
        max_peaks: Maximum number of peaks to keep

    Returns:
        (mz_array, intensity_array) - preprocessed and normalized
    """
    mz = np.array(mz, dtype=np.float32)
    intensity = np.array(intensity, dtype=np.float32)

    # Sort by intensity and keep top peaks
    if len(intensity) > max_peaks:
        top_indices = np.argsort(intensity)[-max_peaks:]
        mz = mz[top_indices]
        intensity = intensity[top_indices]

    # Sort by m/z
    sorted_indices = np.argsort(mz)
    mz = mz[sorted_indices]
    intensity = intensity[sorted_indices]

    # Normalize intensity
    if intensity.max() > 0:
        intensity = intensity / intensity.max()

    return mz, intensity


def format_similarity_score(score: float) -> str:
    """Format a similarity score."""
    return f"{score:.3f}"


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate long text."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
