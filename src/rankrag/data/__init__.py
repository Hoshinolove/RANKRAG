from rankrag.data.base import DatasetAdapter
from rankrag.data.hotpotqa import HotpotQAAdapter
from rankrag.data.scidocs import SCIDOCSAdapter
from rankrag.data.candidate_corpus import CandidateCorpus, CandidateRecord
from rankrag.data.factory import create_candidate_corpus, create_dataset_adapter

__all__ = [
    "CandidateCorpus",
    "CandidateRecord",
    "DatasetAdapter",
    "HotpotQAAdapter",
    "SCIDOCSAdapter",
    "create_candidate_corpus",
    "create_dataset_adapter",
]
