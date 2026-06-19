"""Knowledge graph models for ENTSO-E regulation documents."""

from enum import Enum

from pydantic import BaseModel, Field


class NodeLabel(str, Enum):
    """Node types in the regulation knowledge graph."""

    DOCUMENT = "Document"
    SECTION = "Section"
    CONCEPT = "Concept"
    BIDDING_ZONE = "BiddingZone"
    NETWORK_CODE = "NetworkCode"
    BALANCING_PRODUCT = "BalancingProduct"
    TSO = "TSO"


class RegulationTriple(BaseModel):
    """A subject-predicate-object relation extracted from regulation text."""

    subject: str
    predicate: str
    object: str
    subject_label: NodeLabel = NodeLabel.CONCEPT
    object_label: NodeLabel = NodeLabel.CONCEPT
    source_doc: str = ""
    page: int = 0


class ChunkExtraction(BaseModel):
    """Structured triples extracted from a single document chunk."""

    triples: list[RegulationTriple] = Field(default_factory=list)


class GraphNodeView(BaseModel):
    """Serializable view of a graph node for search results."""

    node_id: str
    label: str
    name: str
    source_doc: str = ""
    page: int = 0
