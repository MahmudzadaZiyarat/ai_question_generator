#src/advanced_chunking.py
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ChunkType(Enum):
    """Types of content chunks"""
    THEOREM = "theorem"
    DEFINITION = "definition"
    EXAMPLE = "example"
    PROOF = "proof"
    FORMULA = "formula"
    PROBLEM = "problem"
    SOLUTION = "solution"
    SECTION = "section"
    PARAGRAPH = "paragraph"

@dataclass
class AdvancedChunk:
    """Enhanced chunk with mathematical content awareness"""
    content: str
    chunk_type: ChunkType
    metadata: Dict[str, Any]
    mathematical_entities: List[str]
    formulas: List[str]
    concepts: List[str]
    difficulty_indicators: List[str]
    chunk_index: int
    importance_score: float = 0.0

class MathematicalEntityExtractor:
    """Extracts mathematical entities and formulas from text"""
    
    def __init__(self):
        # Mathematical patterns
        self.formula_patterns = [
            r'\$\$.*?\$\$',  # Display math (LaTeX)
            r'\$.*?\$',      # Inline math (LaTeX)
            r'[a-zA-Z]\s*=\s*[^,\.\s]+',  # Simple equations
            r'\b\d+\s*[+\-*/=]\s*\d+',    # Basic arithmetic
            r'∀|∃|∈|∉|⊂|⊆|∪|∩|∅|ℝ|ℕ|ℤ|ℚ|ℂ',  # Math symbols
        ]
        
        # Mathematical keywords and concepts
        self.math_keywords = {
            'algebra': ['variable', 'equation', 'polynomial', 'linear', 'quadratic', 'matrix'],
            'calculus': ['derivative', 'integral', 'limit', 'continuity', 'differential'],
            'geometry': ['angle', 'triangle', 'circle', 'polygon', 'area', 'volume'],
            'analysis': ['convergence', 'sequence', 'series', 'topology', 'metric', 'norm'],
            'vector': ['vector', 'space', 'basis', 'dimension', 'span', 'linear']
        }
        
        # Structure markers
        self.structure_markers = {
            ChunkType.THEOREM: [r'theorem\s*\d*\.?', r'proposition\s*\d*\.?', r'lemma\s*\d*\.?'],
            ChunkType.DEFINITION: [r'definition\s*\d*\.?', r'def\s*\d*\.?', r'define'],
            ChunkType.EXAMPLE: [r'example\s*\d*\.?', r'ex\s*\d*\.?', r'for\s+example'],
            ChunkType.PROOF: [r'proof\.?', r'solution\.?', r'show\s+that'],
            ChunkType.PROBLEM: [r'problem\s*\d*\.?', r'exercise\s*\d*\.?', r'question\s*\d*\.?'],
        }
        
        # Difficulty indicators
        self.difficulty_indicators = {
            'beginner': ['basic', 'simple', 'elementary', 'introduction'],
            'intermediate': ['advanced', 'complex', 'intermediate', 'requires'],
            'advanced': ['theorem', 'proof', 'rigorous', 'sophisticated']
        }
    
    def extract_formulas(self, text: str) -> List[str]:
        """Extract mathematical formulas from text"""
        formulas = []
        for pattern in self.formula_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            formulas.extend(matches)
        return list(set(formulas))  # Remove duplicates
    
    def extract_concepts(self, text: str) -> Tuple[List[str], str]:
        """Extract mathematical concepts and determine domain"""
        concepts = []
        domain_scores = {domain: 0 for domain in self.math_keywords}
        
        text_lower = text.lower()
        
        for domain, keywords in self.math_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    concepts.append(keyword)
                    domain_scores[domain] += 1
        
        # Determine primary domain
        primary_domain = max(domain_scores, key=domain_scores.get) if any(domain_scores.values()) else 'general'
        
        return list(set(concepts)), primary_domain
    
    def detect_chunk_type(self, text: str) -> ChunkType:
        """Detect the type of mathematical content"""
        text_lower = text.lower()
        
        for chunk_type, patterns in self.structure_markers.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return chunk_type
        
        # Default classification based on content
        if any(word in text_lower for word in ['prove', 'proof', 'show that']):
            return ChunkType.PROOF
        elif '=' in text and any(char in text for char in '+-*/'):
            return ChunkType.FORMULA
        elif any(word in text_lower for word in ['example', 'instance', 'consider']):
            return ChunkType.EXAMPLE
        else:
            return ChunkType.PARAGRAPH
    
    def assess_difficulty(self, text: str) -> List[str]:
        """Assess difficulty indicators in text"""
        text_lower = text.lower()
        indicators = []
        
        for level, keywords in self.difficulty_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                indicators.append(level)
        
        return indicators if indicators else ['intermediate']

class AdvancedTextChunker:
    """Advanced chunking strategy optimized for mathematical content"""
    
    def __init__(self, 
                 target_chunk_size: int = 512,
                 max_chunk_size: int = 1024,
                 min_chunk_size: int = 100):
        
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.entity_extractor = MathematicalEntityExtractor()
    
    def simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting for mathematical content"""
        # Split by common sentence endings, but preserve mathematical structure
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            # If adding this sentence would exceed target size
            if current_size + sentence_size > self.target_chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.strip()) >= self.min_chunk_size:
                    chunks.append(chunk_text.strip())
                
                # Start new chunk
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
            
            # Force split if we exceed max size
            if current_size > self.max_chunk_size:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.strip()) >= self.min_chunk_size:
                    chunks.append(chunk_text.strip())
                current_chunk = []
                current_size = 0
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(chunk_text.strip())
        
        return chunks
    
    def create_advanced_chunks(self, text: str, metadata: Dict[str, Any] = None) -> List[AdvancedChunk]:
        """Create advanced chunks with mathematical content analysis"""
        if metadata is None:
            metadata = {}
        
        # First level: sentence-based chunking
        text_chunks = self.simple_sentence_split(text)
        
        advanced_chunks = []
        
        for i, chunk_text in enumerate(text_chunks):
            # Extract mathematical entities
            formulas = self.entity_extractor.extract_formulas(chunk_text)
            concepts, domain = self.entity_extractor.extract_concepts(chunk_text)
            chunk_type = self.entity_extractor.detect_chunk_type(chunk_text)
            difficulty = self.entity_extractor.assess_difficulty(chunk_text)
            
            # Calculate importance score
            importance_score = self._calculate_importance_score(
                chunk_text, formulas, concepts, chunk_type
            )
            
            # Create enhanced metadata
            enhanced_metadata = {
                **metadata,
                'domain': domain,
                'word_count': len(chunk_text.split()),
                'char_count': len(chunk_text),
                'formula_count': len(formulas),
                'concept_count': len(concepts)
            }
            
            advanced_chunk = AdvancedChunk(
                content=chunk_text.strip(),
                chunk_type=chunk_type,
                metadata=enhanced_metadata,
                mathematical_entities=concepts,
                formulas=formulas,
                concepts=concepts,
                difficulty_indicators=difficulty,
                chunk_index=i,
                importance_score=importance_score
            )
            
            advanced_chunks.append(advanced_chunk)
        
        return advanced_chunks
    
    def _calculate_importance_score(self, text: str, formulas: List[str], 
                                  concepts: List[str], chunk_type: ChunkType) -> float:
        """Calculate importance score for chunk prioritization"""
        score = 0.0
        
        # Base score by chunk type
        type_scores = {
            ChunkType.THEOREM: 1.0,
            ChunkType.DEFINITION: 0.9,
            ChunkType.PROOF: 0.8,
            ChunkType.FORMULA: 0.7,
            ChunkType.EXAMPLE: 0.6,
            ChunkType.PROBLEM: 0.7,
            ChunkType.SOLUTION: 0.6,
            ChunkType.SECTION: 0.4,
            ChunkType.PARAGRAPH: 0.3
        }
        score += type_scores.get(chunk_type, 0.3)
        
        # Formula bonus
        score += min(len(formulas) * 0.1, 0.3)
        
        # Concept bonus
        score += min(len(concepts) * 0.05, 0.2)
        
        # Keyword importance
        important_keywords = ['theorem', 'proof', 'definition', 'important', 'key']
        text_lower = text.lower()
        for keyword in important_keywords:
            if keyword in text_lower:
                score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0