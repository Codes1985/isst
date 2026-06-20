"""
Sequence Processor — FASTA parsing, validation, segment ID for influenza genomes.
"""

import re
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import SEGMENTS, SUBTYPES, SEGMENT_LENGTH_RANGES

logger = logging.getLogger(__name__)

VALID_BASES = set("ACGTUacgtuNnRYSWKMBDHV")
AMBIGUOUS_BASES = set("NnRYSWKMBDHV")
COMPLEMENT = str.maketrans("ACGT", "TGCA")

SEGMENT_PATTERNS = {
    "PB2": [r"\bPB2\b", r"\bsegment\s*1\b"], "PB1": [r"\bPB1\b", r"\bsegment\s*2\b"],
    "PA":  [r"\bPA\b", r"\bsegment\s*3\b"],   "HA":  [r"\bHA\b", r"\bsegment\s*4\b", r"\bhemagglutinin\b"],
    "NP":  [r"\bNP\b", r"\bsegment\s*5\b"],   "NA":  [r"\bNA\b", r"\bsegment\s*6\b", r"\bneuraminidase\b"],
    "M":   [r"\bM\b", r"\bMP\b", r"\bsegment\s*7\b", r"\bmatrix\b"],
    "NS":  [r"\bNS\b", r"\bsegment\s*8\b", r"\bnon-?structural\b"],
}
SEGMENT_NUMBER_MAP = {"1":"PB2","2":"PB1","3":"PA","4":"HA","5":"NP","6":"NA","7":"M","8":"NS"}


@dataclass
class SegmentRecord:
    segment_name: str
    sequence: str
    length: int
    is_valid: bool = True
    validation_notes: List[str] = field(default_factory=list)


@dataclass
class SequenceRecord:
    sequence_id: str
    subtype: str
    collection_date: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    segments: Dict[str, SegmentRecord] = field(default_factory=dict)

    @property
    def segments_found(self) -> int:
        return len(self.segments)

    @property
    def is_complete(self) -> bool:
        return self.segments_found == len(SEGMENTS)

    @property
    def completeness(self) -> float:
        return self.segments_found / len(SEGMENTS)

    @property
    def valid_segments(self) -> Dict[str, SegmentRecord]:
        return {k: v for k, v in self.segments.items() if v.is_valid}


def clean_sequence(seq: str) -> str:
    return re.sub(r"\s+", "", seq).upper().replace("U", "T")


def validate_sequence(sequence: str, segment_name: str) -> Tuple[bool, List[str]]:
    notes = []
    is_valid = True
    if not sequence:
        return False, ["Empty sequence"]
    invalid_chars = set(sequence) - VALID_BASES
    if invalid_chars:
        notes.append(f"Invalid characters: {invalid_chars}")
        is_valid = False
    length = len(sequence)
    if segment_name in SEGMENT_LENGTH_RANGES:
        min_len, max_len = SEGMENT_LENGTH_RANGES[segment_name]
        if length < min_len * 0.5:
            notes.append(f"Very short ({length} bp, expected {min_len}-{max_len})")
        elif length < min_len:
            notes.append(f"Below range ({length} bp, expected {min_len}-{max_len})")
        elif length > max_len * 1.1:
            notes.append(f"Above range ({length} bp, expected {min_len}-{max_len})")
    ambig_count = sum(1 for b in sequence if b in AMBIGUOUS_BASES)
    ambig_pct = ambig_count / length if length > 0 else 0
    if ambig_pct > 0.05:
        notes.append(f"High ambiguity: {ambig_pct:.1%}")
    if ambig_pct > 0.20:
        notes.append("Excessive ambiguity")
        is_valid = False
    return is_valid, notes


def identify_segment(header: str) -> Optional[str]:
    for segment, patterns in SEGMENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, header, re.IGNORECASE):
                return segment
    num_match = re.search(r"segment[_\s]*(\d)", header, re.IGNORECASE)
    if num_match and num_match.group(1) in SEGMENT_NUMBER_MAP:
        return SEGMENT_NUMBER_MAP[num_match.group(1)]
    return None


def identify_subtype(header: str) -> Optional[str]:
    h = header.upper()
    if "H1N1PDM09" in h or "H1N1PDM" in h:
        return "H1N1pdm09"
    if "H3N2" in h:
        return "H3N2"
    if "H1N1" in h:
        return "H1N1pdm09"
    return None


def extract_collection_date(header: str) -> Optional[str]:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", header)
    if match:
        return match.group(1)
    match = re.search(r"(\d{4}/\d{2}/\d{2})", header)
    if match:
        return match.group(1).replace("/", "-")
    return None


def parse_fasta(filepath: str) -> List[Tuple[str, str]]:
    records = []
    current_header = None
    current_seq_parts = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    records.append((current_header, "".join(current_seq_parts)))
                current_header = line[1:].strip()
                current_seq_parts = []
            else:
                current_seq_parts.append(line)
    if current_header is not None:
        records.append((current_header, "".join(current_seq_parts)))
    return records


def parse_fasta_string(fasta_text: str) -> List[Tuple[str, str]]:
    records = []
    current_header = None
    current_seq_parts = []
    for line in fasta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                records.append((current_header, "".join(current_seq_parts)))
            current_header = line[1:].strip()
            current_seq_parts = []
        else:
            current_seq_parts.append(line)
    if current_header is not None:
        records.append((current_header, "".join(current_seq_parts)))
    return records


class SequenceProcessor:
    def __init__(self, default_subtype: Optional[str] = None):
        self.default_subtype = default_subtype

    def process_file(self, filepath: str) -> List[SequenceRecord]:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"FASTA file not found: {filepath}")
        raw_records = parse_fasta(filepath)
        logger.info(f"Parsed {len(raw_records)} sequences from {path.name}")
        return self._process_records(raw_records)

    def process_string(self, fasta_text: str) -> List[SequenceRecord]:
        return self._process_records(parse_fasta_string(fasta_text))

    def _process_records(self, raw_records: List[Tuple[str, str]]) -> List[SequenceRecord]:
        isolates: Dict[str, SequenceRecord] = {}

        for header, raw_seq in raw_records:
            sequence = clean_sequence(raw_seq)
            segment = identify_segment(header)
            if segment is None:
                logger.warning(f"Could not identify segment: {header}")
                continue

            is_valid, notes = validate_sequence(sequence, segment)
            isolate_id = self._extract_isolate_id(header)

            if isolate_id not in isolates:
                subtype = identify_subtype(header) or self.default_subtype or "H3N2"
                if subtype not in SUBTYPES:
                    logger.warning(
                        f"Unrecognised subtype {subtype!r} for {isolate_id!r}; "
                        f"known subtypes: {SUBTYPES}. Proceeding anyway."
                    )
                isolates[isolate_id] = SequenceRecord(
                    sequence_id=isolate_id, subtype=subtype,
                    collection_date=extract_collection_date(header),
                    metadata={"source_headers": []},
                )
            record = isolates[isolate_id]
            record.metadata["source_headers"].append(header)

            if segment in record.segments:
                logger.warning(f"Duplicate segment {segment} for {isolate_id}, keeping first")
                continue

            record.segments[segment] = SegmentRecord(
                segment_name=segment, sequence=sequence,
                length=len(sequence), is_valid=is_valid, validation_notes=notes,
            )

        results = list(isolates.values())
        complete = sum(1 for r in results if r.is_complete)
        logger.info(f"Processed {len(results)} isolates: {complete} complete, {len(results)-complete} partial")
        return results

    def _extract_isolate_id(self, header: str) -> str:
        cleaned = header
        for seg in SEGMENTS:
            cleaned = re.sub(rf"\|{seg}\b", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(rf"_{seg}\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\|segment[_\s]*\d", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"_segment[_\s]*\d", "", cleaned, flags=re.IGNORECASE)
        isolate_id = cleaned.split("|")[0].strip()
        if not isolate_id:
            isolate_id = f"unknown_{hashlib.md5(header.encode()).hexdigest()[:8]}"
        return isolate_id

    def get_processing_summary(self, records: List[SequenceRecord]) -> Dict:
        summary = {
            "total_isolates": len(records),
            "complete_genomes": sum(1 for r in records if r.is_complete),
            "partial_genomes": sum(1 for r in records if not r.is_complete),
            "by_subtype": {},
            "segment_coverage": {seg: 0 for seg in SEGMENTS},
        }
        for record in records:
            summary["by_subtype"][record.subtype] = summary["by_subtype"].get(record.subtype, 0) + 1
            for seg_name in record.segments:
                summary["segment_coverage"][seg_name] += 1
        return summary
