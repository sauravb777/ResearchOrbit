from __future__ import annotations
from dataclasses import dataclass
from ..llm import LLMClient
from ..schemas import FollowupQA, PaperAnalysis, ResearchResult, Source
from .qa import answer_followup
from .targeting import answer_about_uploaded_paper, find_named_source
@dataclass
class Grounder:
    llm: LLMClient
    @staticmethod
    def _chat_context(result: ResearchResult) -> tuple[list[Source], list]:
        """Return the case corpus plus verification evidence with canonical source ids."""
        sources = list(result.sources)
        dedupe_to_id = {s.dedupe_key(): s.id for s in sources}
        source_by_id = {s.id: s for s in sources}
        evidence = list(result.evidence)
        seen_evidence = {e.id for e in evidence}
        for verification in result.verifications:
            local_id_map: dict[str, str] = {}
            for source in verification.new_sources:
                canonical_id = dedupe_to_id.get(source.dedupe_key())
                if canonical_id is None:
                    sources.append(source)
                    dedupe_to_id[source.dedupe_key()] = source.id
                    source_by_id[source.id] = source
                    canonical_id = source.id
                local_id_map[source.id] = canonical_id
            for item in verification.new_evidence:
                canonical_id = local_id_map.get(item.source_id)
                if canonical_id is None and item.source_id in source_by_id:
                    canonical_id = dedupe_to_id.get(source_by_id[item.source_id].dedupe_key(), item.source_id)
                canonical_id = canonical_id or item.source_id
                if item.id in seen_evidence:
                    continue
                if canonical_id != item.source_id:
                    item = item.model_copy(update={"source_id": canonical_id})
                evidence.append(item)
                seen_evidence.add(item.id)
        return sources, evidence
    def answer(self, result: ResearchResult, question: str) -> FollowupQA:
        """Answer only from the current case's report corpus plus verified chat evidence."""
        sources, evidence = self._chat_context(result)
        return answer_followup(
            question,
            evidence,
            sources,
            llm=self.llm,
            analyses=result.paper_analyses,
            claims=result.claims,
            report_markdown=result.report_markdown,
            prior_followups=result.followups,
            prior_verifications=result.verifications,
        )
    def answer_about_source(
        self,
        result: ResearchResult,
        question: str,
        source: Source,
        cached_analysis: PaperAnalysis | None = None,
    ) -> tuple[FollowupQA, PaperAnalysis | None]:
        sources, evidence = self._chat_context(result)
        named = find_named_source(question, sources)
        return answer_about_uploaded_paper(
            question,
            source,
            named,
            sources,
            evidence,
            llm=self.llm,
            existing_analysis=cached_analysis,
            report_type=result.report_type,
        )
