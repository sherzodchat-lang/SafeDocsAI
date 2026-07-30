from typing import Any

from app.domain_profiles.base import DomainProfile
from app.modules.rag.text_utils import boost_article_chunks, detect_article_reference


class LegalDomainProfile(DomainProfile):
    def __init__(self) -> None:
        super().__init__(name="legal", assistant_name="SafeDocsAI Legal")

    def rerank_results(
        self, query_text: str, results: dict[str, Any]
    ) -> dict[str, Any]:
        article_ref = detect_article_reference(query_text)
        if not article_ref:
            return results
        return boost_article_chunks(results, article_ref)

    def answer_rules(self, language: str) -> str:
        return (
            "Answer using only the provided legal or regulatory context. "
            "Prefer concise, grounded explanations and do not invent missing provisions."
        )
