import torch
from torch.utils.data import Dataset
import pandas as pd
import re
from typing import Dict, Optional, Callable, List, Union, Any, Tuple
from abc import ABC, abstractmethod


class BioTextProcessor(ABC):
    """
    Abstract base class for biological-text processors.
    Defines a unified interface for processing biological text data.
    """

    def __init__(self, fields_to_keep: Union[List[str], str] = "all"):
        """
        Initialize the processor.

        Args:
            fields_to_keep: data fields to keep; either "all" or a list of field names
        """
        self.fields_to_keep = fields_to_keep

    @abstractmethod
    def process(self, biotext_data: Dict, meta_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Abstract method to process biological text data; must be implemented by subclasses.

        Args:
            biotext_data: raw biological text data {molecule_id: text_str}
            meta_data: metadata DataFrame

        Returns:
            processed data {molecule_id: processed_str}
        """
        pass

    def __call__(self, biotext_data: Dict, meta_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Make the processor callable like a function.

        Args:
            biotext_data: raw biological text data
            meta_data: metadata DataFrame

        Returns:
            processed data
        """
        return self.process(biotext_data, meta_data)

class HMDBProcessor(BioTextProcessor):
    """HMDB dataset processor implementation; supports dict and string output formats."""

    def __init__(self, fields_to_keep: Union[List[str], str] = "all",
                return_type: str = "dict",
                max_synonyms: int = 5,
                delimiter: str = "; "):
        """
        Initialize the HMDB processor.

        Args:
            fields_to_keep: data fields to keep; either "all" or a list of field names
            return_type: return type - "dict" returns a dict; "str" returns a string
            max_synonyms: maximum number of synonyms to keep
            delimiter: delimiter used to replace the "{}" separator in the raw data
        """
        super().__init__(fields_to_keep)
        # All possible fields
        self.all_fields = [
            "molecular_function",
            "enzymes_proteins_pathways",
            "toxicity_or_benefit",
            "disease_association",
            "distribution",
            "smiles_synonyms",
            "kingdom"
        ]

        # Decide which fields to keep
        if self.fields_to_keep == "all":
            self.fields_to_keep = self.all_fields

        # Set return type
        if return_type not in ["dict", "str"]:
            raise ValueError("return_type must be 'dict' or 'str'")
        self.return_type = return_type

        # Max number of synonyms
        self.max_synonyms = max_synonyms

        # Delimiter
        self.delimiter = delimiter

        # Default sentence templates for missing data
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
        Process biological text data from the HMDB dataset.

        Args:
            biotext_data: raw biological text data {molecule_id: text_str}
            meta_data: metadata DataFrame containing fields like synonyms and kingdom

        Returns:
            Processed data; format depends on return_type:
            - dict mode: {molecule_id: processed_dict}
            - str mode: {molecule_id: processed_str}
        """
        processed_data = {}

        # Process each molecule ID
        for molecule_id, text_data in biotext_data.items():
            # Initialize processing result for this molecule
            processed_dict = {field: None for field in self.all_fields}

            # Extract data from text
            if isinstance(text_data, str) and text_data.strip():
                # Parse molecular function
                if "molecular_function" in self.fields_to_keep:
                    molecular_function_match = re.search(
                        r'=== molecular_function ===\s*{.*?"biological_function_sentence":\s*"(.*?)"\s*}',
                        text_data, re.DOTALL
                    )
                    if molecular_function_match and molecular_function_match.group(1).strip():
                        processed_dict["molecular_function"] = molecular_function_match.group(1).strip()

                # Parse enzymes/proteins/pathways
                if "enzymes_proteins_pathways" in self.fields_to_keep:
                    enzymes_match = re.search(
                        r'=== enzymes_proteins_pathways ===\s*{.*?"enzymes_proteins_pathways_sentence":\s*"(.*?)"\s*}',
                        text_data, re.DOTALL
                    )
                    if enzymes_match and enzymes_match.group(1).strip():
                        processed_dict["enzymes_proteins_pathways"] = enzymes_match.group(1).strip()

                # Parse toxicity and benefit
                if "toxicity_or_benefit" in self.fields_to_keep:
                    toxicity_match = re.search(
                        r'=== toxicity_or_benefit ===\s*{.*?"toxicity_or_benefit_sentence":\s*"(.*?)"\s*}',
                        text_data, re.DOTALL
                    )
                    if toxicity_match and toxicity_match.group(1).strip():
                        processed_dict["toxicity_or_benefit"] = toxicity_match.group(1).strip()

                # Parse disease association
                if "disease_association" in self.fields_to_keep:
                    disease_match = re.search(
                        r'=== disease_association ===\s*{.*?"disease_association_sentence":\s*"(.*?)"\s*}',
                        text_data, re.DOTALL
                    )
                    if disease_match and disease_match.group(1).strip():
                        processed_dict["disease_association"] = disease_match.group(1).strip()

            # Get extra information from metadata
            if meta_data is not None:
                # Check whether molecule_id is in the metadata
                meta_row = None
                if 'HMDB.ID' in meta_data.columns and molecule_id in meta_data['HMDB.ID'].values:
                    meta_row = meta_data[meta_data['HMDB.ID'] == molecule_id].iloc[0]
                elif molecule_id in meta_data.index:
                    meta_row = meta_data.loc[molecule_id]

                if meta_row is not None:
                    # Process synonyms and SMILES
                    if "smiles_synonyms" in self.fields_to_keep:
                        synonyms_dict = {
                            'smiles': None,
                            'common_names': []
                        }

                        # Add SMILES structure
                        if 'SMILES.ID' in meta_row and not pd.isna(meta_row['SMILES.ID']):
                            smiles = meta_row['SMILES.ID']
                            if isinstance(smiles, str) and smiles.strip():
                                synonyms_dict['smiles'] = smiles

                        # Add synonyms
                        if 'Synonyms' in meta_row and not pd.isna(meta_row['Synonyms']):
                            synonyms = meta_row['Synonyms']
                            if isinstance(synonyms, str):
                                # Replace "{}" with delimiter
                                synonyms_processed = synonyms.replace('{}', self.delimiter)
                                # Split into list
                                synonyms_list = synonyms_processed.split(';')
                                # Filter out empty strings
                                synonyms_list = [s.strip() for s in synonyms_list if s.strip()]
                                # Cap the number
                                synonyms_dict['common_names'] = synonyms_list[:self.max_synonyms]
                            elif isinstance(synonyms, list):
                                # Replace "{}" with delimiter
                                synonyms_list = [s.replace('{}', self.delimiter) for s in synonyms if s.strip()]
                                # Cap the number
                                synonyms_dict['common_names'] = synonyms_list[:self.max_synonyms]

                        # Save processing result if there is any valid data
                        if synonyms_dict['smiles'] is not None or synonyms_dict['common_names']:
                            processed_dict["smiles_synonyms"] = synonyms_dict

                    # Process kingdom/taxonomy info
                    if "kingdom" in self.fields_to_keep:
                        kingdom_info = {}
                        for category in ['Kingdom', 'Super_class', 'Class', 'Sub_class']:
                            if category in meta_row and not pd.isna(meta_row[category]):
                                kingdom_info[category.lower()] = meta_row[category]

                        if kingdom_info:
                            processed_dict["kingdom"] = kingdom_info

                    # Process distribution info
                    if "distribution" in self.fields_to_keep:
                        distribution_info = {}
                        for location_type in ['Biospecimen_locations', 'Cellular_locations', 'Tissue_locations']:
                            if location_type in meta_row and not pd.isna(meta_row[location_type]):
                                key = location_type.replace('_locations', '')
                                locations = meta_row[location_type]
                                if isinstance(locations, str):
                                    locations_list = locations.split(';')
                                    # Replace delimiter
                                    locations_list = [loc.replace('{}', self.delimiter) for loc in locations_list]
                                    distribution_info[key.lower()] = locations_list
                                elif isinstance(locations, list):
                                    # Replace delimiter
                                    locations_list = [loc.replace('{}', self.delimiter) for loc in locations]
                                    distribution_info[key.lower()] = locations_list

                        if distribution_info:
                            processed_dict["distribution"] = distribution_info

            # Convert to sentence form
            processed_dict = self._convert_to_sentences(processed_dict, molecule_id, meta_data)

            # Handle return type
            if self.return_type == "str":
                # Convert dict to string
                processed_str = self._dict_to_text(processed_dict)
                processed_data[molecule_id] = processed_str
            else:
                # Save the result, keeping only the desired fields
                processed_data[molecule_id] = {k: v for k, v in processed_dict.items() if k in self.fields_to_keep}

        return processed_data
    
    def _convert_to_sentences(self, data_dict: Dict, molecule_id: str, meta_data) -> Dict:
        """
        Convert dict-format data into sentence form.

        Args:
            data_dict: dict containing extracted data
            molecule_id: molecule ID, used to generate more specific descriptions

        Returns:
            dict with sentence-form data
        """
        result_dict = {}

        for field in self.all_fields:
            if field not in self.fields_to_keep:
                continue

            value = data_dict.get(field)

            # If the value is None, use the default sentence
            if value is None:
                result_dict[field] = self.default_sentences[field]
                continue

            # Handle each field type
            if field == "smiles_synonyms":
                if isinstance(value, dict):
                    smiles = value.get('smiles')
                    common_names = value.get('common_names', [])

                    # Build sentence
                    sentence_parts = []

                    # Add SMILES info
                    if smiles:
                        sentence_parts.append(f"SMILES structure: {smiles}")

                    # Add synonyms info
                    if common_names:
                        if len(common_names) == 1:
                            sentence_parts.append(f"Also known as: {common_names[0]}")
                        else:
                            names_str = self.delimiter.join(common_names)
                            sentence_parts.append(f"Common names include: {names_str}")

                            # If synonyms exceed display limit, add a hint
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
                # Other fields are already in string form
                result_dict[field] = value

        return result_dict

    def _dict_to_text(self, data_dict: Dict) -> str:
        """
        Convert a dict into text form.

        Args:
            data_dict: dict containing sentence-form data

        Returns:
            merged text
        """
        text_parts = []
        
        for field in self.fields_to_keep:
            if field in data_dict:
                text_parts.append(f"=== {field} ===\n{data_dict[field]}")
        
        return "\n\n".join(text_parts)
    

class KEGGProcessor(BioTextProcessor):
    def process(self, biotext_data: Dict, meta_data: Optional[pd.DataFrame] = None) -> Dict:
        raise NotImplementedError("KEGGProcessor.process is not implemented yet")