# -*- coding: utf-8 -*-
"""
GPT-based functional inference for Spec2Function (optimized).
Two-tier GPT architecture:
  1. Cluster-level: generate functional theme names and detailed reports
  2. Global-level: synthesize a narrative from cluster reports

Optimization notes:
- Keep the focus on biological process themes, not structure identification.
- Cluster names should be biological processes (e.g., "Phospholipid Membrane Remodeling").
- Avoid using metabolite names (e.g., "LPC 16:0").
- LogFC is used only for feature filtering, NOT for GPT generation (metabolomics doesn't emphasize up/down regulation like transcriptomics).
"""
from typing import List, Dict, Optional

from .llm_client import LLMClient

class GPTInference:
    """GPT functional inference with a two-tier architecture."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_tokens: int = 2000,
        temperature: float = 0.4,
        provider: str = "openai",
        base_url: Optional[str] = None
    ):
        """
        Args:
            api_key: OpenAI API key
            model: GPT model name
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
        """
        self.client = LLMClient(provider=provider, api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        print(f"GPT initialized: provider={provider}, model={model}, max_tokens={max_tokens}, temp={temperature}")
    
    # =========================================================================
    #  TIER 1: Cluster-Level Analysis
    # =========================================================================
    
    def generate_cluster_functional_name(self,
                                    cluster_stats: Dict,
                                    top_metabolites: List[tuple],
                                    retrieved_texts: List[Dict],
                                    background_info: Optional[str] = None,
                                    debug_output_path: Optional[str] = None,
                                    return_prompt: bool = False) -> str:
        """
        Generate a functional theme name for a cluster (KEGG-style).
        """
            
        # ========== 从retrieved_texts中提取生物学相关信息 ==========
        disease_mentions = []
        pathway_mentions = []
        function_mentions = []
        biospecimen_mentions = []
        
        for hit in retrieved_texts[:8]:
            for text in hit.get('texts', [])[:3]:
                text_lower = text.lower()
                # 提取疾病关联
                if 'associated with' in text_lower or 'disease' in text_lower:
                    disease_mentions.append(text)
                # 提取pathway/process关键词
                if any(kw in text_lower for kw in ['pathway', 'metabolism', 'biosynthesis', 'degradation', 'cycle', 'signaling']):
                    pathway_mentions.append(text)
                # 提取功能描述
                if any(kw in text_lower for kw in ['function', 'role', 'involved in', 'participates']):
                    function_mentions.append(text)
                # 提取生物样本信息
                if any(kw in text_lower for kw in ['blood', 'urine', 'liver', 'brain', 'tissue', 'biospecimen']):
                    biospecimen_mentions.append(text)
        
        # ========== 构建prompt ==========
        prompt = f"""You are a metabolic pathway expert. Your task is to name a metabolite cluster using KEGG-style pathway nomenclature.

**Experimental Context:**
{background_info if background_info else 'Comparing metabolic profiles between Case and Control groups.'}

**Cluster Composition:**
- {cluster_stats['known_count']} annotated + {cluster_stats['unknown_count']} unannotated MS2 features
"""
        
        # ========== 高权重：已知化合物 ==========
        if top_metabolites and len(top_metabolites) > 0:
            known_names = [t[0] for t in top_metabolites[:5]]
            prompt += f"""
**KNOWN METABOLITES (HIGH CONFIDENCE - PRIORITIZE THESE):**
{', '.join(known_names)}
↑ These are structurally identified compounds. Base your pathway name primarily on these.
"""
        
        # ========== 低权重：检索到的功能相似代谢物 ==========
        prompt += """
**INFERRED EVIDENCE (LOWER CONFIDENCE - USE AS SUPPORTING INFO ONLY):**
"""
        
        # 添加提取的生物学信息
        if disease_mentions:
            prompt += f"\n[Disease associations]: {' | '.join(disease_mentions[:3])[:300]}"
        if pathway_mentions:
            prompt += f"\n[Pathway/Process hints]: {' | '.join(pathway_mentions[:3])[:300]}"
        if function_mentions:
            prompt += f"\n[Functional roles]: {' | '.join(function_mentions[:3])[:300]}"
        
        # 添加原始retrieved texts作为补充
        if retrieved_texts:
            all_texts = []
            for hit in retrieved_texts[:6]:
                texts = hit.get('texts', [])[:2]
                all_texts.extend(texts)
            if all_texts:
                prompt += f"\n[Additional context]: {' | '.join(all_texts[:6])[:400]}"
        prompt += """

    **NAMING RULES (CRITICAL):**

    1. Use KEGG-style biological process names, NOT chemical class names
    2. Focus on: metabolism, biosynthesis, degradation, signaling, transport
    3. Be SPECIFIC to the pathway branch, not generic categories

    **KEGG-STYLE EXAMPLES (GOOD):**
    ✓ "Tryptophan metabolism"
    ✓ "Primary bile acid biosynthesis"  
    ✓ "Glycerophospholipid metabolism"
    ✓ "Steroid hormone biosynthesis"
    ✓ "Arachidonic acid metabolism"
    ✓ "Sphingolipid signaling pathway"
    ✓ "Lysine degradation"
    ✓ "Purine metabolism"
    ✓ "Fatty acid elongation"
    ✓ "Glycolysis / Gluconeogenesis"

    **BAD EXAMPLES (AVOID):**
    ✗ "Mitochondrial Energy Metabolism" ← Too vague, sounds like a textbook chapter
    ✗ "Phospholipid Membrane Dynamics" ← Too generic, not a real pathway
    ✗ "Amino Acid Catabolism & Nitrogen Handling" ← Too broad, pick ONE specific pathway
    ✗ "Oxidative Stress Response" ← Too generic, specify the actual metabolic process
    ✗ "Long-chain fatty acids" ← This is a chemical class, not a process
    ✗ "LPC species" ← Chemical nomenclature, not biological

    **DECISION GUIDE:**
    - If you see steroids/cholesterol → "Steroid hormone biosynthesis" or "Bile acid biosynthesis"
    - If you see phospholipids (PC, PE, LPC) → "Glycerophospholipid metabolism" or "Phospholipase signaling"
    - If you see amino acids → Name the SPECIFIC amino acid pathway (e.g., "Tryptophan metabolism")
    - If you see sugars/glycolytic intermediates → "Glycolysis" or "Pentose phosphate pathway"
    - If you see sphingolipids → "Sphingolipid metabolism" or "Ceramide signaling"

    **ORGANISM CONSTRAINT (CRITICAL - HUMAN STUDY):**
    - This is a HUMAN metabolomics study
    - ONLY use pathways from KEGG's human metabolism categories
    - FORBIDDEN pathways (will be rejected):
    ✗ Carbazole, Furan, Penicillin, Streptomycin (industrial/microbial)
    ✗ Mugineic acid, Terpenoid backbone, Flavonoid (plant-specific)
    ✗ Any pathway not found in human KEGG
    - When uncertain, default to: Lipid/Amino acid/Carbohydrate metabolism
    
    **OUTPUT:** Return ONLY the pathway name (2-6 words). No quotes, no explanation.
    """

        # Save prompt for debugging if requested
        if debug_output_path:
            try:
                with open(debug_output_path, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write(f"CLUSTER {cluster_stats['id']} - THEME NAME PROMPT\n")
                    f.write("="*80 + "\n\n")
                    f.write(prompt)
                print(f"      Prompt saved to: {debug_output_path}")
            except Exception as e:
                print(f"      Failed to save prompt: {e}")
        
        try:
            theme_name = self.client.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a KEGG pathway database curator. Name metabolite clusters using standard pathway nomenclature."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=30,  # 减少token数，强制简洁
                temperature=0.2  # 降低temperature，更确定性
            )
            theme_name = theme_name.strip('"\'').strip()
            if return_prompt:
                return theme_name, prompt
            return theme_name
            
        except Exception as e:
            print(f" GPT theme name generation failed: {e}")
            fallback = f"Metabolic Cluster {cluster_stats['id']}"
            if return_prompt:
                return fallback, prompt
            return fallback
    
    def generate_cluster_report(self,
                                cluster_stats: Dict,
                                functional_name: str,
                                top_metabolites: List[tuple],
                                retrieved_texts: List[Dict],
                                example_logfc: Optional[List[float]] = None,
                                background_info: Optional[str] = None) -> str:
            """
            Generate a detailed cluster report.
            """
            prompt = f"""You are a systems biology expert writing a PROCESS-FOCUSED metabolomics module description for: **{functional_name}**

**Hidden Study Context (INTERNAL ONLY - DO NOT EXPOSE IN OUTPUT):**
{background_info if background_info else 'General metabolomics context for internal reasoning only.'}

**Module Composition (for internal reasoning only):**
- {cluster_stats['known_count']} known + {cluster_stats['unknown_count']} unknown features

**KNOWN STRUCTURAL ANCHORS (FOR REASONING ONLY, DO NOT QUOTE NAMES IN OUTPUT):**
"""

            if top_metabolites and len(top_metabolites) > 0:
                for t in top_metabolites[:5]:
                    name = t[0]
                    score = t[2] if len(t) > 2 else None
                    if score is not None:
                        prompt += f"  - {name} | MS2 Score: {score:.3f}\n"
                    else:
                        prompt += f"  - {name} | MS2 Score: N/A\n"

            prompt += """
**INFERRED FUNCTIONAL EVIDENCE (SUPPORTING CONTEXT):**
"""
            if retrieved_texts:
                for i, hit in enumerate(retrieved_texts[:6], 1):
                    texts = hit.get('texts', [])[:2]
                    combined = " ".join(texts)[:300]
                    prompt += f"  {i}. {combined}\n"
            else:
                prompt += "  Limited functional information available\n"

            prompt += f"""
**TASK**: Write one process-focused paragraph (150-250 words).

**HARD CONSTRAINTS (MUST FOLLOW):**
1. Describe ONLY biological functions/processes/pathways.
2. DO NOT mention any specific metabolite names, chemical formulas, or compound abbreviations.
3. DO NOT mention cluster IDs, numbers, or phrases like "Cluster X".
4. DO NOT mention whether structurally identified metabolites are present/absent.
5. Avoid contrastive wording such as "Although there are no structurally identified metabolites...".
6. DO NOT start with templates like "In the context of ..." or similar framing openers.
7. DO NOT explicitly mention experimental background, study design, sample groups, case/control, treatment/control, species labels, disease names, or aging labels.
8. Use background only as silent guidance for emphasis; keep wording context-agnostic to readers.

**STYLE TARGET:**
- Use pathway/process wording (e.g., energy metabolism, lipid remodeling, inflammatory signaling, redox balance).
- Keep the paragraph biologically interpretable and mechanism-oriented.
- Write direct scientific prose with no visible study-context narration.
- Treat unknowns generically as functionally related features when needed.

**OUTPUT FORMAT**: Return only the paragraph (no headers, no bullet points).
"""

            try:
                return self.client.chat_completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a metabolomics expert writing process-only module descriptions. Never output specific metabolite names, cluster IDs, statements about structural-identification presence/absence, or explicit study background labels (case/control, disease names, aging labels, species labels)."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=self.temperature
                )
            except Exception as e:
                print(f" GPT cluster report generation failed: {e}")
                return f"Unable to generate detailed report for {functional_name}."

    def generate_pubmed_query(self,
                             functional_name: str,
                             top_metabolites: List[tuple],
                             background_info: Optional[str] = None) -> str:
        """
        Generate a PubMed search query using LLM.
        
        Args:
            functional_name: Functional theme name
            top_metabolites: [(name, count), ...]
            background_info: Experimental background
        
        Returns:
            PubMed query string, e.g., "liver cirrhosis fatty acid oxidation"
        """
        prompt = f"""You are a biomedical literature expert. Generate a concise PubMed search query.

**Experimental Context:**
{background_info if background_info else 'General metabolomics study'}

**Cluster Information:**
- Functional Theme: {functional_name}
- Representative Metabolites: {', '.join([t[0] for t in top_metabolites[:3]]) if top_metabolites else 'Unknown'}

**TASK**: Generate a PubMed search query (3-6 keywords) that:
1. Extracts the KEY DISEASE/CONDITION from background (e.g., "liver cirrhosis", "diabetes", "cancer")
2. Combines it with the functional process (e.g., "fatty acid oxidation", "glycolysis")
3. Optionally adds a metabolite class if relevant

**GOOD EXAMPLES**:
OK "liver cirrhosis fatty acid oxidation"
OK "hepatocellular carcinoma lipid metabolism"
OK "diabetes mellitus amino acid catabolism"
OK "heart failure mitochondrial dysfunction"

**BAD EXAMPLES**:
NO "C18:1 carnitine" (too specific metabolite name)
NO "phospholipid membrane remodeling" (missing disease context)
NO "comparing liver vs healthy" (too vague)

**OUTPUT**: Return ONLY the search query (3-6 keywords), NO quotes, NO explanation.
"""
        
        try:
            query = self.client.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in biomedical literature search."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=30,
                temperature=0.3
            ).strip('"\'')
            return query
        except Exception as e:
            print(f" PubMed query generation failed: {e}")
            return functional_name
    
    def generate_global_story(self,
                            cluster_reports: List[Dict],
                            background_info: Optional[str] = None) -> str:
        """
        Synthesize a global biological narrative from cluster reports.
        
        Args:
            cluster_reports: [{'id': int, 'functional_name': str, 'report': str, 
                              'direction': str, 'avg_logfc': float, 'papers': [...]}, ...]
            background_info: Experimental background
        
        Returns:
            Global narrative text
        """
        prompt = "You are a senior systems biologist writing the DISCUSSION section of a metabolomics paper.\n\n"
        
        if background_info:
            prompt += f"**Experimental Context:**\n{background_info}\n\n"
        else:
            prompt += "**Experimental Context:** Case vs. Control metabolomics comparison using MS2 functional clustering.\n\n"
        
        prompt += f"**Identified Functional Modules ({len(cluster_reports)} clusters):**\n\n"
        
        for cr in cluster_reports:
            # Note: Direction/LogFC icons and trends are intentionally excluded
            # as metabolomics doesn't emphasize up/down regulation like transcriptomics
            prompt += f"### {cr['functional_name']} (Cluster {cr['id']})\n"
            prompt += f"- Composition: {cr.get('known_count', 0)} known + {cr.get('unknown_count', 0)} unknown metabolites\n"
            prompt += f"- Analysis:\n{cr['report']}\n"
            
            if cr.get('papers'):
                prompt += f"- Supporting Literature ({len(cr['papers'])} papers):\n"
                for p in cr['papers'][:2]:
                    prompt += f"   {p['title']} (PMID: {p['pmid']})\n"
            prompt += "\n"
        
        prompt += """
**TASK**: Write a cohesive SYSTEMS-LEVEL narrative (400-600 words) that:

1. **Cross-Module Integration**: 
   - Don't just summarize each cluster separately
   - Identify metabolic cross-talk between functional modules
   - Look for compensatory mechanisms or pathway feedback loops

2. **Physiological Context**:
   - What is the OVERALL metabolic phenotype?
   - Tissue-specific effects? Organ cross-talk?
   - How do these functional changes explain the disease/condition?

3. **Embrace the Unknown**:
   - Acknowledge that many clustered metabolites are structurally uncharacterized
   - Emphasize that functional similarity (via MS2) reveals their likely biological roles
   - Suggest that unknowns may represent novel pathway intermediates or biomarkers

4. **Biological Insight**:
   - Conclude with a hypothesis about the underlying mechanism
   - What does this metabolic rewiring tell us about the condition?

**STYLE REQUIREMENTS**:
- Professional, narrative prose (like a journal Discussion section)
- Use process/pathway terminology (avoid listing individual metabolites)
- No bullet points; write in cohesive paragraphs
- Be evidence-based but intellectually bold

**OUTPUT**: Return the narrative as continuous text (no section headers).
"""
        
        try:
            return self.client.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert systems biologist writing integrative metabolomics discussions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        except Exception as e:
            print(f" GPT global story generation failed: {e}")
            return f"Error generating global narrative: {str(e)}"
    
    # =========================================================================
    #  LEGACY METHODS (Mode 1 Compatibility)
    # =========================================================================
    
    def single_annotation(self, 
                          retrieved_fragments: List[Dict],
                          papers: Optional[List[Dict]] = None,
                          user_focus: Optional[str] = None) -> str:
        """
        Generate functional annotation for a single spectrum.
        Used in Mode 1.
        """
        prompt = self._build_single_prompt(retrieved_fragments, papers, user_focus)
        
        try:
            annotation = self.client.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in metabolomics and biochemistry. Your task is to provide comprehensive functional annotations for metabolites based on MS2 spectral similarity matches and supporting literature from PubMed."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return annotation
        
        except Exception as e:
            return f"Error generating annotation: {str(e)}"
    
    def batch_annotation_summary(self,
                                individual_results: List[Dict],
                                user_focus: Optional[str] = None) -> str:
        """
        Generate batch summary for multiple spectra.
        """
        prompt = self._build_batch_prompt(individual_results, user_focus)
        
        try:
            summary = self.client.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in metabolomics and systems biology. Your task is to provide pathway enrichment analysis and functional summaries for sets of metabolites."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return summary
        
        except Exception as e:
            return f"Error generating batch summary: {str(e)}"

    # =========================================================================
    #  Internal Helper Methods (Prompt Builders)
    # =========================================================================

    def _build_single_prompt(self, retrieved_fragments, papers, user_focus):
        """Mode 1 prompt construction."""
        fragments = sorted(retrieved_fragments, key=lambda x: x['similarity'], reverse=True)
        
        prompt = """You are an expert in metabolomics and mass spectrometry data interpretation. 
Your task is to infer the biological function of an unknown metabolic feature based on its MS2 spectral similarity to known metabolites.

### INPUT DATA EXPLANATION:
1. **Retrieved Metabolites**: A list of known metabolites whose MS2 spectra match the unknown feature.
- **Similarity Score**: Ranges from 0 to 1. A score > 0.85 indicates a likely structural match (high reliability). A score between 0.6-0.85 indicates structural similarity (e.g., same compound class or substructure).
- **Weighting Strategy**: You must prioritize information from metabolites with higher similarity scores. 
2. **Supporting Literature**: Abstracts from PubMed that may provide functional context for the retrieved metabolites.

"""
        prompt += "### RETRIEVED METABOLITES (Evidence):\n"
        for i, frag in enumerate(fragments[:10], 1):
            sim = frag['similarity']
            if sim >= 0.85:
                tier, text_len = 'High', 500
            elif sim >= 0.65:
                tier, text_len = 'Medium', 300
            else:
                tier, text_len = 'Low', 200
            desc = frag.get('text', '')[:text_len] + "..." if len(frag.get('text', '')) > text_len else frag.get('text', '')

            prompt += f"Hit {i} | Name: {frag.get('molecule_name', 'Unknown')} | HMDB: {frag['accession']}\n"
            prompt += f"   Similarity: {sim:.4f} (Confidence: {tier})\n"
            prompt += f"   Function Description: {desc}\n\n"

        if papers and len(papers) > 0:
            prompt += "### SUPPORTING LITERATURE (Context):\n"
            for i, paper in enumerate(papers, 1):
                prompt += f"Paper {i} | Title: {paper['title']}\n"
                prompt += f"   Summary: {paper.get('abstract', '')[:400]}...\n\n"

        if user_focus:
            prompt += f"### USER FOCUS:\nPlease specifically address: {user_focus}\n\n"

        prompt += """### INSTRUCTIONS FOR FUNCTIONAL INFERENCE:
**Step 1: Analyze Consensus & Divergence**
- Identify common functional themes shared across the top retrieved metabolites.
**Step 2: Synthesize Annotation**
- Combine the structural evidence with literature context. 
- **Do not hallucinate.**

### OUTPUT FORMAT (Strict Markdown):
**1. Primary Biological Classification**
- **Class/Superclass**: [e.g., Glycerophospholipids]
- **Main Functional Role**: [One sentence summary]

**2. Detailed Functional Annotation**
- **Biological Processes**: [Describe processes]
- **Pathway Involvement**: [List KEGG/HMDB pathways]
- **Disease Relevance**: [If evidence supports]

**3. Inference Confidence & Evidence**
- **Confidence Level**: [High / Medium / Low]
- **Reasoning**: [Explain why]
"""
        return prompt

    def _build_batch_prompt(self, individual_results, user_focus):
        """Mode 1 batch prompt construction."""
        prompt = f"Analyze the following set of {len(individual_results)} MS2 spectra with their functionally similar metabolites.\n\n"
        prompt += "=" * 70 + "\nINDIVIDUAL SPECTRUM RESULTS:\n" + "=" * 70 + "\n\n"
        
        for result in individual_results[:20]:
            prompt += f"Spectrum {result['spectrum_id'] + 1}:\n"
            prompt += f"  Top Metabolites: {', '.join(result['top_metabolites'][:3])}\n"
            prompt += f"  Functions: {', '.join(result['top_functions'][:2])}\n\n"
        
        if len(individual_results) > 20:
            prompt += f"... and {len(individual_results) - 20} more spectra.\n\n"
        
        prompt += "=" * 70 + "\n"
        if user_focus:
            prompt += f"\nUser-Specified Focus:\n{user_focus}\n\nPlease emphasize aspects related to the user's focus.\n\n"
        
        prompt += """
Please provide a comprehensive batch analysis that includes:
1. **Global Functional Enrichment**
2. **Pathway Analysis**
3. **Biological Interpretation**
4. **Confidence Assessment**
Format as structured text. Be comprehensive but concise.
"""
        return prompt

    # =========================================================================
    #  Post-processing: Disambiguate Duplicate Theme Names
    # =========================================================================
    
    def disambiguate_duplicate_themes(self, 
                                    cluster_reports: List[Dict],
                                    cluster_retrieved_texts_map: Dict[int, List[Dict]] = None) -> List[Dict]:
        """
        Post-process cluster reports to disambiguate duplicate theme names.
        
        Args:
            cluster_reports: List of cluster report dicts
            cluster_retrieved_texts_map: Optional dict mapping cluster_id to retrieved_texts
        
        Returns:
            Updated cluster_reports with disambiguated functional_name
        """
        from collections import Counter
        
        name_counts = Counter(cr['functional_name'] for cr in cluster_reports)
        duplicates = {name for name, count in name_counts.items() if count > 1}
        
        if not duplicates:
            print("   No duplicate theme names found, skipping disambiguation.")
            return cluster_reports
        
        print(f"   Found {len(duplicates)} duplicate theme names: {duplicates}")
        
        for dup_name in duplicates:
            dup_clusters = [cr for cr in cluster_reports if cr['functional_name'] == dup_name]
            
            print(f"\n   Disambiguating '{dup_name}' ({len(dup_clusters)} clusters)...")
            
            cluster_metabolites = []
            cluster_texts = []
            
            for cr in dup_clusters:
                mets = cr.get('top_metabolites', [])
                cluster_metabolites.append({
                    'cluster_id': cr['id'],
                    'metabolites': mets[:5] if mets else ['Unknown']
                })
                # 获取该cluster的retrieved_texts
                if cluster_retrieved_texts_map and cr['id'] in cluster_retrieved_texts_map:
                    cluster_texts.append(cluster_retrieved_texts_map[cr['id']])
                else:
                    cluster_texts.append(None)
            
            try:
                suffixes = self._generate_disambiguation_suffixes(
                    dup_name, 
                    cluster_metabolites,
                    cluster_texts if any(cluster_texts) else None
                )
                
                for cr, suffix in zip(dup_clusters, suffixes):
                    if suffix and suffix.lower() not in ['unknown', 'variant']:
                        cr['functional_name'] = f"{dup_name} ({suffix})"
                        print(f"     Cluster {cr['id']}: -> {cr['functional_name']}")
            except Exception as e:
                print(f"     Failed to disambiguate '{dup_name}': {e}")
                for i, cr in enumerate(dup_clusters):
                    cr['functional_name'] = f"{dup_name} (branch {i+1})"
        
        return cluster_reports
    
    def _generate_disambiguation_suffixes(self, 
                                        theme_name: str, 
                                        cluster_metabolites: List[Dict],
                                        cluster_retrieved_texts: List[Dict] = None) -> List[str]:
        """
        Use GPT to generate short distinguishing suffixes for duplicate themes.
        
        Args:
            theme_name: The duplicate theme name
            cluster_metabolites: List of {'cluster_id': int, 'metabolites': [str]}
            cluster_retrieved_texts: Optional, list of retrieved_texts for each cluster
        
        Returns:
            List of suffix strings, one per cluster
        """
        
        prompt = f"""You are a metabolic pathway expert. Multiple clusters share the theme: "{theme_name}"

    Your task: Generate a SHORT distinguishing suffix (1-3 words) for each cluster to specify the PATHWAY BRANCH or BIOLOGICAL CONTEXT.

    **Clusters to distinguish:**
    """
        
        for i, cm in enumerate(cluster_metabolites):
            mets_str = ', '.join(cm['metabolites'][:5]) if cm['metabolites'] else 'Unknown'
            prompt += f"\nCluster {cm['cluster_id']}: {mets_str}"
            
            # 如果有retrieved_texts，提取关键生物学信息
            if cluster_retrieved_texts and i < len(cluster_retrieved_texts):
                texts = cluster_retrieved_texts[i]
                if texts:
                    # 提取疾病、组织、pathway关键词
                    bio_hints = []
                    for hit in texts[:3]:
                        for t in hit.get('texts', [])[:2]:
                            t_lower = t.lower()
                            if 'associated with' in t_lower:
                                # 提取疾病名
                                bio_hints.append(t.split('associated with')[-1][:50].strip())
                            if any(kw in t_lower for kw in ['blood', 'liver', 'brain', 'urine', 'feces']):
                                bio_hints.append(t[:50])
                    if bio_hints:
                        prompt += f"\n   [Bio-context]: {' | '.join(bio_hints[:2])}"
        
        prompt += """

    **SUFFIX RULES:**
    1. Focus on PATHWAY BRANCH or BIOLOGICAL PROCESS, not chemical structure
    2. Use terms that would appear in KEGG or biological databases
    3. Avoid chemical nomenclature (no "C16:0", "Long-chain", etc.)

    **GOOD SUFFIX EXAMPLES:**
    ✓ "bile acid branch" - specifies pathway branch
    ✓ "hepatic" - specifies tissue context
    ✓ "inflammatory" - specifies biological context  
    ✓ "biosynthesis" vs "degradation" - specifies direction
    ✓ "mitochondrial" vs "peroxisomal" - specifies compartment
    ✓ "omega-oxidation" - specifies specific sub-pathway

    **BAD SUFFIX EXAMPLES:**
    ✗ "Module 1" - no information
    ✗ "Long-chain species" - chemical, not biological
    ✗ "PC/LPC" - chemical abbreviations
    ✗ "Unknown Metabolites" - useless
    ✗ "Detergent-like" - not a biological term

    **OUTPUT FORMAT:**
    Cluster [ID]: [suffix]
    Cluster [ID]: [suffix]

    Return ONLY cluster-suffix pairs. No explanation.
    """
        
        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a KEGG pathway curator. Generate biological pathway branch names."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.2
            )
            
            # Parse response
            suffixes = []
            lines = response.strip().split('\n')
            
            suffix_map = {}
            for line in lines:
                line = line.strip()
                if ':' in line and 'Cluster' in line:
                    parts = line.split(':', 1)
                    try:
                        cluster_id = int(''.join(filter(str.isdigit, parts[0])))
                        suffix = parts[1].strip().strip('"\'')
                        # 清理suffix，确保简洁
                        suffix = suffix.split('.')[0].strip()  # 移除句号后的内容
                        if len(suffix) > 25:  # 太长则截断
                            suffix = suffix[:25].rsplit(' ', 1)[0]
                        suffix_map[cluster_id] = suffix
                    except:
                        continue
            
            for cm in cluster_metabolites:
                cid = cm['cluster_id']
                suffixes.append(suffix_map.get(cid, f"variant {cid}"))
            
            return suffixes
            
        except Exception as e:
            print(f"     GPT disambiguation failed: {e}")
            return [f"variant {cm['cluster_id']}" for cm in cluster_metabolites]