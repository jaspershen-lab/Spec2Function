import os
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pickle

def parse_ms_xml_folder(folder_path):
    """
    解析包含MS-MS数据的XML文件夹
    
    参数:
    folder_path (str): 包含XML文件的文件夹路径
    
    返回:
    tuple: (ms_data, meta_data)
        - ms_data: 字典，键为不带扩展名的文件名，值为包含mz、intensity和molecule_id的字典
        - meta_data: DataFrame，包含每个文件的元数据
    """
    # 初始化数据结构
    ms_data = {}
    meta_data_list = []
    
    # 获取所有XML文件
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
    
    for file_name in xml_files:
        file_path = os.path.join(folder_path, file_name)
        
        try:
            # 解析XML文件
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # 移除文件扩展名
            file_name_without_ext = os.path.splitext(file_name)[0]
            
            # 提取MS-MS峰值数据
            mz_list = []
            intensity_list = []
            molecule_id_list = []
            
            for peak in root.findall('.//ms-ms-peak'):
                mz = peak.find('mass-charge')
                intensity = peak.find('intensity')
                molecule_id = peak.find('molecule-id')
                
                if mz is not None and intensity is not None:
                    mz_list.append(float(mz.text))
                    intensity_list.append(float(intensity.text))
                    
                    # 提取molecule_id，如果为nil则为None
                    if molecule_id is not None and 'nil' not in molecule_id.attrib:
                        molecule_id_list.append(molecule_id.text)
                    else:
                        molecule_id_list.append(None)
            
            # 获取database-id
            database_id_elem = root.find('database-id')
            database_id = database_id_elem.text if database_id_elem is not None and database_id_elem.text else np.nan
            
            # 获取ionization-mode (Polarity)
            polarity_elem = root.find('ionization-mode')
            polarity = polarity_elem.text if polarity_elem is not None and polarity_elem.text else np.nan
            
            # 获取precursor_mass (adduct-mass)
            adduct_mass_elem = root.find('adduct-mass')
            precursor_mass = adduct_mass_elem.text if adduct_mass_elem is not None and adduct_mass_elem.text else np.nan
            
            # 获取splash-key
            splash_id_elem = root.find('splash-key')
            splash_id = splash_id_elem.text if splash_id_elem is not None and splash_id_elem.text else np.nan
            
            # 存储MS数据 - 使用不带扩展名的文件名
            ms_data[file_name_without_ext] = {
                'mz': mz_list,
                'intensity': intensity_list,
                'molecule_id': database_id  # 使用database-id作为molecule_id
            }
            
            # 存储元数据 - 使用不带扩展名的文件名
            meta_data_list.append({
                'file_name': file_name_without_ext,
                'HMDB.ID': database_id,
                'Polarity': polarity,
                'precursor_mass': precursor_mass,
                'splash_id': splash_id
            })
            
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
    
    # 创建元数据DataFrame
    meta_data = pd.DataFrame(meta_data_list)
    
    return ms_data, meta_data

def save_ms_data(ms_data, output_file):
    """
    保存MS数据到pickle文件
    
    参数:
    ms_data (dict): MS数据字典
    output_file (str): 输出文件路径
    """
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(ms_data, f)
    print(f"MS数据已保存到 {output_file}")

def save_meta_data(meta_data, output_file):
    """
    保存元数据到CSV文件
    
    参数:
    meta_data (DataFrame): 元数据DataFrame
    output_file (str): 输出文件路径
    """
    meta_data.to_csv(output_file, index=False)
    print(f"元数据已保存到 {output_file}")

def main():
    # 示例用法
    folder_path = "/Users/cgxjdzz/Desktop/NTU phd/ms2_database_feifan/HMDB raw/hmdb_experimental_msms_spectra"  # 替换为实际XML文件夹路径
    output_dir = "/Users/cgxjdzz/Desktop/NTU phd/ms2_database_feifan/MS2BioText"  # 替换为实际输出目录路径
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析XML文件
    ms_data, meta_data = parse_ms_xml_folder(folder_path)
    
    # 打印结果示例
    print("MS数据样例:")
    for file_name, data in list(ms_data.items())[:1]:  # 只打印第一个文件的数据
        print(f"文件: {file_name}")
        print(f"质荷比数量: {len(data['mz'])}")
        print(f"前5个质荷比: {data['mz'][:5]}")
        print(f"前5个强度值: {data['intensity'][:5]}")
        print(f"molecule_id: {data['molecule_id']}")
        print()
    
    print("元数据:")
    print(meta_data.head())
    
    # 保存数据
    ms_data_file = os.path.join(output_dir, "new_ms_data.pkl")
    meta_data_file = os.path.join(output_dir, "new_meta_data.csv")
    
    save_ms_data(ms_data, ms_data_file)
    save_meta_data(meta_data, meta_data_file)

if __name__ == "__main__":
    main()