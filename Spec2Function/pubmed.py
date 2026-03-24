# -*- coding: utf-8 -*-
"""
PubMed literature search for Spec2Function
"""
from Bio import Entrez
from typing import List, Dict
import time


class PubMedSearcher:
    """PubMed literature search helper."""

    def __init__(self, email: str = "your_email@example.com"):
        """
        Args:
            email: Your email address (required by PubMed API).
        """
        Entrez.email = email

    def search_by_metabolites(self, metabolite_names: List[str], 
                            max_results: int = 5) -> List[Dict]:
        """
        Search by multiple metabolite names (combined query).
        Use OR to join keywords rather than full phrases.
        """
        # Clean metabolite names
        clean_names = [self._clean_metabolite_name(name) for name in metabolite_names[:3]]

        # Extract keywords by splitting on whitespace
        keywords = []
        for name in clean_names:
            words = name.split()
            keywords.extend(words)

        # Deduplicate and limit count (avoid overly long queries)
        keywords = list(dict.fromkeys(keywords))[:6]

        if not keywords:
            print("Warning: No valid keywords extracted")
            return []

        # Build query with OR
        query = ' OR '.join([f'"{kw}"[Title/Abstract]' for kw in keywords])

        print(f"  Keywords: {keywords}")
        print(f"PubMed query: {query}")

        try:
            handle = Entrez.esearch(
                db="pubmed", 
                term=query, 
                retmax=max_results, 
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()

            id_list = record.get("IdList", [])
            print(f"Found {len(id_list)} papers")

            if not id_list:
                return []

            # Fetch details
            time.sleep(0.5)
            handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            papers = []
            for i, article in enumerate(records.get('PubmedArticle', [])):
                try:
                    medline = article['MedlineCitation']
                    article_data = medline['Article']

                    title = article_data.get('ArticleTitle', 'No title')

                    pub_date = article_data['Journal']['JournalIssue']['PubDate']
                    year = pub_date.get('Year', pub_date.get('MedlineDate', 'Unknown'))
                    try:
                        year = int(str(year)[:4])
                    except:
                        year = 2023

                    authors = []
                    if 'AuthorList' in article_data:
                        for author in article_data['AuthorList'][:3]:
                            if 'LastName' in author:
                                authors.append(author['LastName'])
                    authors_str = ', '.join(authors)
                    if len(article_data.get('AuthorList', [])) > 3:
                        authors_str += ' et al.'

                    journal = article_data['Journal'].get('Title', 'Unknown journal')

                    abstract = ''
                    if 'Abstract' in article_data:
                        abstract_texts = article_data['Abstract'].get('AbstractText', [])
                        if abstract_texts:
                            if isinstance(abstract_texts, list):
                                abstract = ' '.join(str(text) for text in abstract_texts)
                            else:
                                abstract = str(abstract_texts)

                    pmid = str(medline['PMID'])
                    relevance = 90 - i * 3

                    papers.append({
                        'pmid': pmid,
                        'title': title,
                        'year': year,
                        'authors': authors_str,
                        'journal': journal,
                        'abstract': abstract,
                        'relevance': relevance
                    })

                except Exception as e:
                    print(f"Warning: Error parsing article: {e}")
                    continue

            return papers

        except Exception as e:
            print(f"PubMed search error: {e}")
            return []

    def _clean_metabolite_name(self, name: str) -> str:
        """Clean metabolite names and remove noisy tokens."""
        import re

        # Remove stereochemical markers like (R)-, (S)-, (E)-, (Z)-, (+)-, (-)-
        name = re.sub(r'\([RSZE+\-]\)-', '', name)

        # Remove numeric prefixes like 1,1,2-
        name = re.sub(r'^\d+,[\d,]+-', '', name)

        # Collapse extra whitespace
        name = ' '.join(name.split())

        return name.strip()

    def search_by_metabolite(self, metabolite_name: str, 
                            max_results: int = 5) -> List[Dict]:
        """Search PubMed by a single metabolite name."""

        # 1. Clean the metabolite name
        clean_name = self._clean_metabolite_name(metabolite_name)

        # 2. Try multiple query strategies
        queries = [
            f'"{clean_name}"[Title/Abstract]',  # exact match
            f'{clean_name}[Title/Abstract]',     # loose match
            f'{clean_name}',                      # broadest
        ]

        for i, query in enumerate(queries):
            print(f"Try #{i+1}: {query}")

            try:
                handle = Entrez.esearch(
                    db="pubmed", 
                    term=query, 
                    retmax=max_results, 
                    sort="relevance"
                )
                record = Entrez.read(handle)
                handle.close()

                id_list = record.get("IdList", [])
                print(f"  Found {len(id_list)} results")

                if id_list:  # Return on first hit
                    return self._fetch_paper_details(id_list)

            except Exception as e:
                print(f"  Query failed: {e}")
                continue

        print(f"Warning: No papers found for: {metabolite_name}")
        return []

    def _fetch_paper_details(self, id_list: List[str]) -> List[Dict]:
        """Fetch detailed article info for a list of PubMed IDs."""
        time.sleep(0.3)

        try:
            handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
            records = Entrez.read(handle)
            handle.close()
        except Exception as e:
            print(f"Fetch error: {e}")
            return []

        papers = []
        for i, article in enumerate(records.get('PubmedArticle', [])):
            try:
                medline = article['MedlineCitation']
                article_data = medline['Article']

                title = article_data.get('ArticleTitle', 'No title')

                # Year
                pub_date = article_data['Journal']['JournalIssue']['PubDate']
                year = pub_date.get('Year', pub_date.get('MedlineDate', 'Unknown'))
                try:
                    year = int(str(year)[:4])
                except:
                    year = 2023

                # Authors
                authors = []
                if 'AuthorList' in article_data:
                    for author in article_data['AuthorList'][:3]:
                        if 'LastName' in author:
                            authors.append(author['LastName'])
                authors_str = ', '.join(authors)
                if len(article_data.get('AuthorList', [])) > 3:
                    authors_str += ' et al.'

                # Journal
                journal = article_data['Journal'].get('Title', 'Unknown')

                # Abstract
                abstract = ''
                if 'Abstract' in article_data:
                    abstract_texts = article_data['Abstract'].get('AbstractText', [])
                    if abstract_texts:
                        if isinstance(abstract_texts, list):
                            abstract = ' '.join(str(t) for t in abstract_texts)
                        else:
                            abstract = str(abstract_texts)

                pmid = str(medline['PMID'])
                relevance = 95 - i * 5

                papers.append({
                    'pmid': pmid,
                    'title': title,
                    'year': year,
                    'authors': authors_str,
                    'journal': journal,
                    'abstract': abstract,
                    'relevance': relevance
                })

            except Exception as e:
                print(f"Warning: Parse error: {e}")
                continue

        return papers
