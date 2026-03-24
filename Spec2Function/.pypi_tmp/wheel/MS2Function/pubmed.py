from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Dict, List, Optional


class PubMedSearcher:
    def __init__(self, email: Optional[str] = None, tool: str = "MS2BioText") -> None:
        self.email = email
        self.tool = tool

    def _get_json(self, url: str, params: Dict[str, str], *, timeout: int = 30) -> Dict:
        q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        req = urllib.request.Request(f"{url}?{q}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def search(self, term: str, max_results: int = 3) -> List[Dict]:
        try:
            base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
            esearch = self._get_json(
                f"{base}/esearch.fcgi",
                {
                    "db": "pubmed",
                    "retmode": "json",
                    "retmax": str(max_results),
                    "term": term,
                    "tool": self.tool,
                    "email": self.email,
                    "sort": "relevance",
                },
            )
            id_list = esearch.get("esearchresult", {}).get("idlist", []) or []
            if not id_list:
                return []

            ids = ",".join(id_list)
            esummary = self._get_json(
                f"{base}/esummary.fcgi",
                {
                    "db": "pubmed",
                    "retmode": "json",
                    "id": ids,
                    "tool": self.tool,
                    "email": self.email,
                },
            )
            result = esummary.get("result", {})
            out: List[Dict] = []
            for pmid in id_list:
                item = result.get(pmid)
                if not isinstance(item, dict):
                    continue
                out.append(
                    {
                        "pmid": pmid,
                        "title": item.get("title"),
                        "journal": item.get("fulljournalname") or item.get("source"),
                        "pubdate": item.get("pubdate"),
                        "authors": [a.get("name") for a in (item.get("authors") or []) if a.get("name")],
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    }
                )
            return out
        except Exception:
            return []

    def search_by_metabolites(self, metabolites: List[str], max_results: int = 3) -> List[Dict]:
        terms = [m for m in (metabolites or []) if m and str(m).strip()]
        if not terms:
            return []
        if len(terms) == 1:
            query = terms[0]
        else:
            query = "(" + " OR ".join([f"\"{t}\"" for t in terms[:5]]) + ")"
        return self.search(query, max_results=max_results)

