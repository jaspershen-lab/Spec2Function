import torch
from torch.utils.data import Dataset
import pandas as pd
import re
from typing import Dict, Optional, Callable, List, Union, Any, Tuple
from abc import ABC, abstractmethod


class BioTextProcessor(ABC):
    """
    生物文本处理器的抽象基类
    定义了处理生物文本数据的统一接口
    """
    
    def __init__(self, fields_to_keep: Union[List[str], str] = "all"):
        """
        初始化处理器
        
        参数:
            fields_to_keep: 需要保留的数据字段，可以是"all"或字段名列表
        """
        self.fields_to_keep = fields_to_keep
    
    @abstractmethod
    def process(self, biotext_data: Dict, meta_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        处理生物文本数据的抽象方法，子类必须实现
        
        参数:
            biotext_data: 原始的生物文本数据 {molecule_id: text_str}
            meta_data: 元数据DataFrame
            
        返回:
            处理后的数据 {molecule_id: processed_str}
        """
        pass
    
    def __call__(self, biotext_data: Dict, meta_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        使处理器对象可以像函数一样被调用
        
        参数:
            biotext_data: 原始的生物文本数据
            meta_data: 元数据DataFrame
            
        返回:
            处理后的数据
        """
        return self.process(biotext_data, meta_data)

class HMDBProcessor(BioTextProcessor):
    """HMDB数据集的处理器实现，支持字典和字符串输出格式"""
    
    def __init__(self, fields_to_keep: Union[List[str], str] = "all", 
                return_type: str = "dict",
                max_synonyms: int = 5,
                delimiter: str = "; "):
        """
        初始化HMDB处理器
        
        参数:
            fields_to_keep: 需要保留的数据字段，可以是"all"或字段名列表
            return_type: 返回类型 - "dict"返回字典结构，"str"返回字符串结构
            max_synonyms: 最多保留的同义词数量
            delimiter: 用于替换原始数据中"{}"分隔符的新分隔符
        """
        super().__init__(fields_to_keep)
        # 定义所有可能的字段
        self.all_fields = [
            "molecular_function", 
            "enzymes_proteins_pathways", 
            "toxicity_or_benefit", 
            "disease_association", 
            "distribution", 
            "smiles_synonyms", 
            "kingdom"
        ]
        
        # 决定要保留哪些字段
        if self.fields_to_keep == "all":
            self.fields_to_keep = self.all_fields
            
        # 设置返回类型
        if return_type not in ["dict", "str"]:
            raise ValueError("return_type必须是'dict'或'str'")
        self.return_type = return_type
        
        # 设置最大同义词数量
        self.max_synonyms = max_synonyms
        
        # 设置分隔符
        self.delimiter = delimiter
        
        # 定义缺失数据的默认句子模板
        self.default_sentences = {
            "molecular_function": "No known molecular function or biological role has been reported.",
            "enzymes_proteins_pathways": "No specific enzymes, proteins or pathways associated with this compound have been documented.",
            "toxicity_or_benefit": "No information regarding toxicity or health benefits is available for this compound.",
            "disease_association": "No disease associations have been reported for this compound.",
            "distribution": "The distribution of this compound in biological systems is not well characterized.",
            "smiles_synonyms": "No synonyms or SMILES structure are available for this compound.",
            "kingdom": "The taxonomic classification of this compound is not available."
        }
    
    def process(self, biotext_data: Dict, meta_data) -> Dict:
        """
        处理HMDB数据集的生物文本数据
        
        参数:
            biotext_data: 原始的生物文本数据 {molecule_id: text_str}
            meta_data: 元数据DataFrame，包含synonyms和kingdom等信息 
            
        返回:
            处理后的数据，根据return_type设置返回:
            - dict模式: {molecule_id: processed_dict}
            - str模式: {molecule_id: processed_str}
        """
        processed_data = {}
        
        # 处理每个分子ID
        for molecule_id, text_data in biotext_data.items():
            # 初始化该分子的处理结果
            processed_dict = {field: None for field in self.all_fields}
            
            # 从文本中提取数据
            if isinstance(text_data, str) and text_data.strip():
                # 解析分子功能
                if "molecular_function" in self.fields_to_keep:
                    molecular_function_match = re.search(
                        r'=== molecular_function ===\s*{.*?"biological_function_sentence":\s*"(.*?)"\s*}', 
                        text_data, re.DOTALL
                    )
                    if molecular_function_match and molecular_function_match.group(1).strip():
                        processed_dict["molecular_function"] = molecular_function_match.group(1).strip()
                
                # 解析酶蛋白和路径
                if "enzymes_proteins_pathways" in self.fields_to_keep:
                    enzymes_match = re.search(
                        r'=== enzymes_proteins_pathways ===\s*{.*?"enzymes_proteins_pathways_sentence":\s*"(.*?)"\s*}', 
                        text_data, re.DOTALL
                    )
                    if enzymes_match and enzymes_match.group(1).strip():
                        processed_dict["enzymes_proteins_pathways"] = enzymes_match.group(1).strip()
                
                # 解析毒性和益处
                if "toxicity_or_benefit" in self.fields_to_keep:
                    toxicity_match = re.search(
                        r'=== toxicity_or_benefit ===\s*{.*?"toxicity_or_benefit_sentence":\s*"(.*?)"\s*}', 
                        text_data, re.DOTALL
                    )
                    if toxicity_match and toxicity_match.group(1).strip():
                        processed_dict["toxicity_or_benefit"] = toxicity_match.group(1).strip()
                
                # 解析疾病关联
                if "disease_association" in self.fields_to_keep:
                    disease_match = re.search(
                        r'=== disease_association ===\s*{.*?"disease_association_sentence":\s*"(.*?)"\s*}', 
                        text_data, re.DOTALL
                    )
                    if disease_match and disease_match.group(1).strip():
                        processed_dict["disease_association"] = disease_match.group(1).strip()
            
            # 从元数据中获取额外信息
            if meta_data is not None:
                # 检查molecule_id是否在元数据中
                meta_row = None
                if 'HMDB.ID' in meta_data.columns and molecule_id in meta_data['HMDB.ID'].values:
                    meta_row = meta_data[meta_data['HMDB.ID'] == molecule_id].iloc[0]
                elif molecule_id in meta_data.index:
                    meta_row = meta_data.loc[molecule_id]
                
                if meta_row is not None:
                    # 处理同义词和SMILES
                    if "smiles_synonyms" in self.fields_to_keep:
                        synonyms_dict = {
                            'smiles': None,
                            'common_names': []
                        }
                        
                        # 添加SMILES结构
                        if 'SMILES.ID' in meta_row and not pd.isna(meta_row['SMILES.ID']):
                            smiles = meta_row['SMILES.ID']
                            if isinstance(smiles, str) and smiles.strip():
                                synonyms_dict['smiles'] = smiles
                        
                        # 添加同义词
                        if 'Synonyms' in meta_row and not pd.isna(meta_row['Synonyms']):
                            synonyms = meta_row['Synonyms']
                            if isinstance(synonyms, str):
                                # 替换"{}"为delimiter
                                synonyms_processed = synonyms.replace('{}', self.delimiter)
                                # 拆分为列表
                                synonyms_list = synonyms_processed.split(';')
                                # 过滤掉空字符串
                                synonyms_list = [s.strip() for s in synonyms_list if s.strip()]
                                # 限制数量
                                synonyms_dict['common_names'] = synonyms_list[:self.max_synonyms]
                            elif isinstance(synonyms, list):
                                # 替换"{}"为delimiter
                                synonyms_list = [s.replace('{}', self.delimiter) for s in synonyms if s.strip()]
                                # 限制数量
                                synonyms_dict['common_names'] = synonyms_list[:self.max_synonyms]
                        
                        # 如果有任何有效数据，保存到处理结果
                        if synonyms_dict['smiles'] is not None or synonyms_dict['common_names']:
                            processed_dict["smiles_synonyms"] = synonyms_dict
                    
                    # 处理kingdom分类信息
                    if "kingdom" in self.fields_to_keep:
                        kingdom_info = {}
                        for category in ['Kingdom', 'Super_class', 'Class', 'Sub_class']:
                            if category in meta_row and not pd.isna(meta_row[category]):
                                kingdom_info[category.lower()] = meta_row[category]
                        
                        if kingdom_info:
                            processed_dict["kingdom"] = kingdom_info
                    
                    # 处理分布信息
                    if "distribution" in self.fields_to_keep:
                        distribution_info = {}
                        for location_type in ['Biospecimen_locations', 'Cellular_locations', 'Tissue_locations']:
                            if location_type in meta_row and not pd.isna(meta_row[location_type]):
                                key = location_type.replace('_locations', '')
                                locations = meta_row[location_type]
                                if isinstance(locations, str):
                                    locations_list = locations.split(';')
                                    # 替换分隔符
                                    locations_list = [loc.replace('{}', self.delimiter) for loc in locations_list]
                                    distribution_info[key.lower()] = locations_list
                                elif isinstance(locations, list):
                                    # 替换分隔符
                                    locations_list = [loc.replace('{}', self.delimiter) for loc in locations]
                                    distribution_info[key.lower()] = locations_list
                        
                        if distribution_info:
                            processed_dict["distribution"] = distribution_info
            
            # 转换为句子形式
            processed_dict = self._convert_to_sentences(processed_dict, molecule_id, meta_data)
            
            # 根据返回类型进行处理
            if self.return_type == "str":
                # 将字典转换为字符串
                processed_str = self._dict_to_text(processed_dict)
                processed_data[molecule_id] = processed_str
            else:
                # 保存处理结果，只保留需要的字段
                processed_data[molecule_id] = {k: v for k, v in processed_dict.items() if k in self.fields_to_keep}
        
        return processed_data
    
    def _convert_to_sentences(self, data_dict: Dict, molecule_id: str, meta_data) -> Dict:
        """
        将字典格式的数据转换为句子格式
        
        参数:
            data_dict: 包含提取数据的字典
            molecule_id: 分子ID，用于生成更具体的描述
            
        返回:
            包含句子形式数据的字典
        """
        result_dict = {}
        
        for field in self.all_fields:
            if field not in self.fields_to_keep:
                continue
                
            value = data_dict.get(field)
            
            # 如果值为None，使用默认句子
            if value is None:
                result_dict[field] = self.default_sentences[field]
                continue
                
            # 根据不同字段类型进行处理
            if field == "smiles_synonyms":
                if isinstance(value, dict):
                    smiles = value.get('smiles')
                    common_names = value.get('common_names', [])
                    
                    # 构建句子
                    sentence_parts = []
                    
                    # 添加SMILES信息
                    if smiles:
                        sentence_parts.append(f"SMILES structure: {smiles}")
                    
                    # 添加同义词信息
                    if common_names:
                        if len(common_names) == 1:
                            sentence_parts.append(f"Also known as: {common_names[0]}")
                        else:
                            names_str = self.delimiter.join(common_names)
                            sentence_parts.append(f"Common names include: {names_str}")
                            
                            # 如果同义词数量超过显示限制，添加提示
                            total_count = len(common_names)
                            if 'Synonyms' in meta_data.columns and molecule_id in meta_data['HMDB.ID'].values:
                                meta_row = meta_data[meta_data['HMDB.ID'] == molecule_id].iloc[0]
                                if 'Synonyms' in meta_row and not pd.isna(meta_row['Synonyms']):
                                    synonyms = meta_row['Synonyms']
                                    if isinstance(synonyms, str):
                                        total_count = len(synonyms.split(';'))
                                    elif isinstance(synonyms, list):
                                        total_count = len(synonyms)
                            
                            if total_count > len(common_names):
                                sentence_parts.append(f"({total_count - len(common_names)} additional synonyms not shown)")
                    
                    if sentence_parts:
                        result_dict[field] = f"This compound has the following identifiers: {'. '.join(sentence_parts)}."
                    else:
                        result_dict[field] = self.default_sentences[field]
                else:
                    result_dict[field] = self.default_sentences[field]
                    
            elif field == "kingdom":
                if isinstance(value, dict) and value:
                    parts = []
                    if "kingdom" in value:
                        parts.append(f"Kingdom: {value['kingdom']}")
                    if "super_class" in value:
                        parts.append(f"Super class: {value['super_class']}")
                    if "class" in value:
                        parts.append(f"Class: {value['class']}")
                    if "sub_class" in value:
                        parts.append(f"Sub class: {value['sub_class']}")
                    
                    if parts:
                        result_dict[field] = f"The taxonomic classification of this compound is: {'; '.join(parts)}."
                    else:
                        result_dict[field] = self.default_sentences[field]
                else:
                    result_dict[field] = self.default_sentences[field]
                    
            elif field == "distribution":
                if isinstance(value, dict) and value:
                    parts = []
                    if "biospecimen" in value and value["biospecimen"]:
                        parts.append(f"Biospecimen locations: {self.delimiter.join(value['biospecimen'])}")
                    if "cellular" in value and value["cellular"]:
                        parts.append(f"Cellular locations: {self.delimiter.join(value['cellular'])}")
                    if "tissue" in value and value["tissue"]:
                        parts.append(f"Tissue locations: {self.delimiter.join(value['tissue'])}")
                    
                    if parts:
                        result_dict[field] = f"This compound is distributed in the following locations: {'; '.join(parts)}."
                    else:
                        result_dict[field] = self.default_sentences[field]
                else:
                    result_dict[field] = self.default_sentences[field]
            else:
                # 其他字段已经是字符串形式
                result_dict[field] = value
                
        return result_dict
    
    def _dict_to_text(self, data_dict: Dict) -> str:
        """
        将字典转换为文本形式
        
        参数:
            data_dict: 包含句子形式数据的字典
            
        返回:
            合并后的文本
        """
        text_parts = []
        
        for field in self.fields_to_keep:
            if field in data_dict:
                text_parts.append(f"=== {field} ===\n{data_dict[field]}")
        
        return "\n\n".join(text_parts)
    

class KEGGProcessor(BioTextProcessor):
    def process(self, biotext_data: Dict, meta_data: Optional[pd.DataFrame] = None) -> Dict:
        raise NotImplementedError("KEGGProcessor 的 process 方法尚未实现")