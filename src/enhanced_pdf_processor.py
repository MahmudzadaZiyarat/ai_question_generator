# src/enhanced_pdf_processor.py
import fitz  # PyMuPDF
import io
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MathPDFChunk:
    """Enhanced chunk specifically for mathematical PDF content"""
    content: str
    page_number: int
    chunk_index: int
    bbox: Optional[List[float]]  # Bounding box coordinates
    images: List[Dict[str, Any]]  # Embedded images/diagrams
    formulas: List[str]
    tables: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    confidence_score: float = 1.0

class EnhancedMathPDFProcessor:
    """Advanced PDF processor optimized for mathematical content - FIXED VERSION"""
    
    def __init__(self, preserve_layout: bool = True, extract_images: bool = True):
        self.preserve_layout = preserve_layout
        self.extract_images = extract_images
        
        # Mathematical content patterns
        self.math_patterns = {
            'equations': [
                r'\$\$[^$]+\$\$',  # LaTeX display math
                r'\$[^$]+\$',      # LaTeX inline math
                r'[a-zA-Z]\s*=\s*[^,\.\n]+',  # Basic equations
                r'\b\d+\s*[+\-×÷*/=]\s*\d+',  # Arithmetic
                r'[∀∃∈∉⊂⊆∪∩∅ℝℕℤℚℂ∑∏∫∂∇Δ∞±≤≥≠≈√∛∜]',  # Math symbols
                r'[αβγδεζηθικλμνξοπρστυφχψω]',  # Greek letters
            ],
            'formulas': [
                r'[a-zA-Z]\([^)]+\)\s*=',  # Function definitions
                r'[a-zA-Z]_\{[^}]+\}',     # Subscripts
                r'[a-zA-Z]\^\{[^}]+\}',    # Superscripts
                r'\b(?:sin|cos|tan|log|ln|exp|sqrt|sum|int)\b',  # Math functions
            ],
            'theorems': [
                r'(?i)\b(theorem|lemma|proposition|corollary|definition|proof)\b\s*\d*\.?',
            ],
            'references': [
                r'(?i)\b(equation|formula|theorem|lemma|figure|table)\s*\(?(\d+\.?\d*)\)?',
            ]
        }
    
    def extract_text_with_layout(self, pdf_path: str) -> List[MathPDFChunk]:
        """Extract text while preserving mathematical layout - FIXED VERSION"""
        chunks = []
        pdf_document = None
        
        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                try:
                    page = pdf_document[page_num]
                    
                    # Extract text with layout preservation
                    page_chunks = self._process_page_safe(page, page_num)
                    chunks.extend(page_chunks)
                    
                except Exception as e:
                    logger.warning(f"Error processing page {page_num}: {e}. Trying fallback method.")
                    # Try simple text extraction as fallback
                    try:
                        simple_text = page.get_text()
                        if simple_text and simple_text.strip():
                            fallback_chunk = self._create_fallback_chunk(simple_text, page_num)
                            chunks.append(fallback_chunk)
                    except Exception as e2:
                        logger.error(f"Fallback extraction also failed for page {page_num}: {e2}")
            
            logger.info(f"Extracted {len(chunks)} chunks from {len(pdf_document)} pages")
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
        finally:
            if pdf_document:
                try:
                    pdf_document.close()
                except:
                    pass  # Ignore close errors
            
        return chunks
    
    def _process_page_safe(self, page: fitz.Page, page_num: int) -> List[MathPDFChunk]:
        """Process a single page with safe error handling"""
        chunks = []
        
        try:
            # Method 1: Text blocks (preserves layout)
            if self.preserve_layout:
                layout_chunks = self._extract_text_blocks_safe(page, page_num)
                chunks.extend(layout_chunks)
            
            # Method 2: Raw text extraction (fallback)
            if not chunks:  # Only if layout extraction failed
                raw_chunks = self._extract_raw_text_safe(page, page_num)
                chunks.extend(raw_chunks)
            
            # Method 3: OCR for images/diagrams if needed
            if self.extract_images:
                try:
                    image_chunks = self._extract_image_content_safe(page, page_num)
                    chunks.extend(image_chunks)
                except Exception as e:
                    logger.warning(f"Image extraction failed for page {page_num}: {e}")
        
        except Exception as e:
            logger.error(f"Error in safe page processing for page {page_num}: {e}")
            # Ultimate fallback
            try:
                simple_text = page.get_text()
                if simple_text and simple_text.strip():
                    fallback_chunk = self._create_fallback_chunk(simple_text, page_num)
                    chunks.append(fallback_chunk)
            except:
                pass
        
        return chunks
    
    def _extract_text_blocks_safe(self, page: fitz.Page, page_num: int) -> List[MathPDFChunk]:
        """Extract text blocks with safe error handling - FIXED"""
        chunks = []
        
        try:
            # Get text blocks with position information
            blocks = page.get_text("dict")
            
            if not isinstance(blocks, dict) or "blocks" not in blocks:
                logger.warning(f"Invalid blocks structure on page {page_num}")
                return []
            
            for block_idx, block in enumerate(blocks.get("blocks", [])):
                try:
                    if not isinstance(block, dict) or "lines" not in block:
                        continue  # Skip non-text blocks
                    
                    block_text = ""
                    block_bbox = block.get("bbox", [0, 0, 0, 0])
                    
                    # Process lines within block
                    for line in block.get("lines", []):
                        if not isinstance(line, dict):
                            continue
                        
                        line_text = ""
                        for span in line.get("spans", []):
                            if not isinstance(span, dict):
                                continue
                            
                            # FIXED: Safe text extraction
                            text = span.get("text", "")
                            if isinstance(text, str) and text.strip():
                                # Preserve formatting information
                                font = span.get("font", "")
                                size = span.get("size", 0)
                                flags = span.get("flags", 0)
                                
                                # Detect mathematical content by font or formatting
                                if self._is_mathematical_text(text, font, flags):
                                    text = f"[MATH]{text}[/MATH]"
                                
                                line_text += text + " "
                        
                        if line_text.strip():
                            block_text += line_text.strip() + "\n"
                    
                    if block_text.strip():
                        # Analyze mathematical content
                        formulas = self._extract_formulas(block_text)
                        tables = self._detect_tables(block_text)
                        
                        chunk = MathPDFChunk(
                            content=block_text.strip(),
                            page_number=page_num + 1,
                            chunk_index=block_idx,
                            bbox=block_bbox,
                            images=[],
                            formulas=formulas,
                            tables=tables,
                            metadata={
                                "extraction_method": "text_blocks_safe",
                                "block_type": "text",
                                "has_math": len(formulas) > 0,
                                "char_count": len(block_text),
                                "line_count": len(block_text.split('\n'))
                            },
                            confidence_score=0.9
                        )
                        chunks.append(chunk)
                
                except Exception as e:
                    logger.warning(f"Error processing block {block_idx} on page {page_num}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error extracting text blocks from page {page_num}: {e}")
        
        return chunks
    
    def _extract_raw_text_safe(self, page: fitz.Page, page_num: int) -> List[MathPDFChunk]:
        """Fallback raw text extraction - FIXED"""
        chunks = []
        
        try:
            # Get raw text - SAFE VERSION
            raw_text = page.get_text()
            
            if raw_text and isinstance(raw_text, str) and raw_text.strip():
                # Split into meaningful chunks
                text_chunks = self._intelligent_text_split(raw_text)
                
                for i, chunk_text in enumerate(text_chunks):
                    if not isinstance(chunk_text, str) or len(chunk_text.strip()) < 50:
                        continue  # Skip invalid or very short chunks
                    
                    formulas = self._extract_formulas(chunk_text)
                    
                    chunk = MathPDFChunk(
                        content=chunk_text.strip(),
                        page_number=page_num + 1,
                        chunk_index=i,
                        bbox=None,
                        images=[],
                        formulas=formulas,
                        tables=[],
                        metadata={
                            "extraction_method": "raw_text_safe",
                            "block_type": "text",
                            "has_math": len(formulas) > 0,
                            "char_count": len(chunk_text),
                        },
                        confidence_score=0.7
                    )
                    chunks.append(chunk)
        
        except Exception as e:
            logger.error(f"Error extracting raw text from page {page_num}: {e}")
        
        return chunks
    
    def _extract_image_content_safe(self, page: fitz.Page, page_num: int) -> List[MathPDFChunk]:
        """Extract and process images/diagrams - SAFE VERSION"""
        chunks = []
        
        if not self.extract_images:
            return chunks
        
        try:
            # Get images on the page
            image_list = page.get_images()
            
            for img_idx, img in enumerate(image_list):
                try:
                    # FIXED: Safe image extraction
                    if not isinstance(img, (list, tuple)) or len(img) < 1:
                        continue
                    
                    # Extract image safely
                    xref = img[0]
                    try:
                        pix = fitz.Pixmap(page.parent, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            
                            # Basic image analysis
                            image_info = {
                                "image_index": img_idx,
                                "format": "png",
                                "size": len(img_data),
                                "dimensions": (pix.width, pix.height)
                            }
                            
                            # Create chunk for image
                            chunk = MathPDFChunk(
                                content=f"[IMAGE: Mathematical diagram or figure on page {page_num + 1}]",
                                page_number=page_num + 1,
                                chunk_index=img_idx,
                                bbox=None,
                                images=[image_info],
                                formulas=[],
                                tables=[],
                                metadata={
                                    "extraction_method": "image_safe",
                                    "block_type": "image",
                                    "has_math": True,  # Assume images contain mathematical content
                                    "image_data": img_data
                                },
                                confidence_score=0.6
                            )
                            chunks.append(chunk)
                        
                        pix = None
                    
                    except Exception as e:
                        logger.warning(f"Error processing image {img_idx} on page {page_num}: {e}")
                        continue
                
                except Exception as e:
                    logger.warning(f"Error with image {img_idx} on page {page_num}: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"Error extracting images from page {page_num}: {e}")
        
        return chunks
    
    def _create_fallback_chunk(self, text: str, page_num: int) -> MathPDFChunk:
        """Create a fallback chunk when all else fails"""
        formulas = self._extract_formulas(text)
        
        return MathPDFChunk(
            content=text.strip(),
            page_number=page_num + 1,
            chunk_index=0,
            bbox=None,
            images=[],
            formulas=formulas,
            tables=[],
            metadata={
                "extraction_method": "fallback",
                "block_type": "text",
                "has_math": len(formulas) > 0,
                "char_count": len(text),
            },
            confidence_score=0.5
        )
    
    def _is_mathematical_text(self, text: str, font: str, flags: int) -> bool:
        """Detect if text is mathematical based on font and content"""
        if not isinstance(text, str) or not text.strip():
            return False
        
        # Check font names for math fonts
        math_fonts = ['symbol', 'math', 'equation', 'times', 'cambria']
        font_lower = str(font).lower()
        
        if any(math_font in font_lower for math_font in math_fonts):
            return True
        
        # Check for mathematical symbols
        try:
            for pattern_list in self.math_patterns.values():
                for pattern in pattern_list:
                    if re.search(pattern, text):
                        return True
        except Exception:
            pass
        
        # Check formatting flags (italic often used for variables)
        try:
            if flags & 2**1:  # Italic flag
                if re.match(r'^[a-zA-Z]$', text.strip()):  # Single letter variables
                    return True
        except Exception:
            pass
        
        return False
    
    def _extract_formulas(self, text: str) -> List[str]:
        """Extract mathematical formulas from text - SAFE VERSION"""
        formulas = []
        
        if not isinstance(text, str):
            return formulas
        
        try:
            for pattern_list in self.math_patterns.values():
                for pattern in pattern_list:
                    try:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        formulas.extend(matches)
                    except Exception:
                        continue
            
            # Clean and deduplicate
            formulas = [f.strip() for f in formulas if isinstance(f, str) and f.strip()]
            return list(set(formulas))
        
        except Exception as e:
            logger.warning(f"Error extracting formulas: {e}")
            return []
    
    def _detect_tables(self, text: str) -> List[Dict[str, Any]]:
        """Detect and extract table structures - SAFE VERSION"""
        tables = []
        
        if not isinstance(text, str):
            return tables
        
        try:
            # Simple table detection
            lines = text.split('\n')
            table_lines = []
            
            for line in lines:
                if not isinstance(line, str):
                    continue
                    
                # Look for lines with multiple numbers/values separated by spaces or tabs
                if re.search(r'\s+\d+.*\s+\d+', line) or '\t' in line:
                    table_lines.append(line.strip())
                elif table_lines:
                    # End of table
                    if len(table_lines) >= 2:  # At least 2 rows
                        tables.append({
                            "content": '\n'.join(table_lines),
                            "rows": len(table_lines),
                            "type": "detected_table"
                        })
                    table_lines = []
            
            # Check for remaining table at end
            if len(table_lines) >= 2:
                tables.append({
                    "content": '\n'.join(table_lines),
                    "rows": len(table_lines),
                    "type": "detected_table"
                })
        
        except Exception as e:
            logger.warning(f"Error detecting tables: {e}")
        
        return tables
    
    def _intelligent_text_split(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text intelligently preserving mathematical context - SAFE VERSION"""
        chunks = []
        
        if not isinstance(text, str):
            return chunks
        
        try:
            # Split by double newlines first (paragraphs)
            paragraphs = text.split('\n\n')
            
            current_chunk = ""
            
            for paragraph in paragraphs:
                if not isinstance(paragraph, str):
                    continue
                    
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Check if adding this paragraph would exceed size limit
                if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
            
            # Add remaining chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        except Exception as e:
            logger.warning(f"Error in intelligent text split: {e}")
            # Fallback to simple split
            try:
                if isinstance(text, str) and text.strip():
                    chunks = [text.strip()]
            except:
                pass
        
        return chunks
    
    def process_pdf_comprehensive(self, pdf_path: str) -> Dict[str, Any]:
        """Comprehensive PDF processing with quality assessment - FIXED VERSION"""
        try:
            logger.info(f"Starting comprehensive processing of {pdf_path}")
            
            # Extract chunks
            chunks = self.extract_text_with_layout(pdf_path)
            
            if not chunks:
                logger.warning("No chunks extracted from PDF")
                return {
                    "chunks": [],
                    "raw_chunks": [],
                    "stats": {},
                    "success": False,
                    "error": "No content extracted"
                }
            
            # Quality assessment
            total_chars = sum(len(chunk.content) for chunk in chunks if hasattr(chunk, 'content'))
            math_chunks = [c for c in chunks if hasattr(c, 'metadata') and c.metadata.get('has_math', False)]
            formula_count = sum(len(c.formulas) for c in chunks if hasattr(c, 'formulas'))
            
            # Combine chunks for embedding if needed
            combined_chunks = self._combine_related_chunks(chunks)
            
            processing_stats = {
                "total_chunks": len(chunks),
                "combined_chunks": len(combined_chunks),
                "math_chunks": len(math_chunks),
                "total_characters": total_chars,
                "formula_count": formula_count,
                "avg_chunk_size": total_chars / len(chunks) if chunks else 0,
                "quality_score": self._calculate_quality_score(chunks)
            }
            
            logger.info(f"Processing complete: {processing_stats}")
            
            return {
                "chunks": combined_chunks,
                "raw_chunks": chunks,
                "stats": processing_stats,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive processing: {e}")
            return {
                "chunks": [],
                "raw_chunks": [],
                "stats": {},
                "success": False,
                "error": str(e)
            }
    
    def _combine_related_chunks(self, chunks: List[MathPDFChunk]) -> List[MathPDFChunk]:
        """Combine related chunks to optimize for embedding - SAFE VERSION"""
        if not chunks:
            return []
        
        try:
            combined = []
            current_group = []
            current_size = 0
            max_size = 800  # Optimal for embeddings
            
            for chunk in chunks:
                if not hasattr(chunk, 'content'):
                    continue
                    
                chunk_size = len(chunk.content)
                
                # Check if we should start a new group
                if (current_size + chunk_size > max_size and current_group) or \
                   (current_group and hasattr(chunk, 'page_number') and hasattr(current_group[-1], 'page_number') and 
                    chunk.page_number != current_group[-1].page_number):
                    
                    # Finalize current group
                    combined_chunk = self._merge_chunks(current_group)
                    if combined_chunk:
                        combined.append(combined_chunk)
                    
                    # Start new group
                    current_group = [chunk]
                    current_size = chunk_size
                else:
                    current_group.append(chunk)
                    current_size += chunk_size
            
            # Add remaining group
            if current_group:
                combined_chunk = self._merge_chunks(current_group)
                if combined_chunk:
                    combined.append(combined_chunk)
            
            return combined
        
        except Exception as e:
            logger.error(f"Error combining chunks: {e}")
            return chunks  # Return original chunks if combination fails
    
    def _merge_chunks(self, chunks: List[MathPDFChunk]) -> Optional[MathPDFChunk]:
        """Merge multiple chunks into one optimized chunk - SAFE VERSION"""
        if not chunks:
            return None
        
        if len(chunks) == 1:
            return chunks[0]
        
        try:
            # Combine content
            combined_content = "\n\n".join(chunk.content for chunk in chunks if hasattr(chunk, 'content'))
            
            # Combine formulas
            all_formulas = []
            for chunk in chunks:
                if hasattr(chunk, 'formulas') and chunk.formulas:
                    all_formulas.extend(chunk.formulas)
            unique_formulas = list(set(all_formulas))
            
            # Combine metadata
            combined_metadata = {
                "extraction_method": "combined_safe",
                "source_chunks": len(chunks),
                "has_math": any(hasattr(chunk, 'metadata') and chunk.metadata.get('has_math', False) for chunk in chunks),
                "char_count": len(combined_content),
                "formula_count": len(unique_formulas),
                "page_range": f"{chunks[0].page_number}-{chunks[-1].page_number}" if hasattr(chunks[0], 'page_number') and hasattr(chunks[-1], 'page_number') else "unknown"
            }
            
            # Calculate average confidence
            confidences = [chunk.confidence_score for chunk in chunks if hasattr(chunk, 'confidence_score')]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            return MathPDFChunk(
                content=combined_content,
                page_number=chunks[0].page_number if hasattr(chunks[0], 'page_number') else 1,
                chunk_index=chunks[0].chunk_index if hasattr(chunks[0], 'chunk_index') else 0,
                bbox=None,
                images=[],
                formulas=unique_formulas,
                tables=[],
                metadata=combined_metadata,
                confidence_score=avg_confidence
            )
        
        except Exception as e:
            logger.error(f"Error merging chunks: {e}")
            return chunks[0] if chunks else None  # Return first chunk as fallback
    
    def _calculate_quality_score(self, chunks: List[MathPDFChunk]) -> float:
        """Calculate quality score for extraction - SAFE VERSION"""
        if not chunks:
            return 0.0
        
        try:
            score = 0.0
            
            # Check content extraction quality
            total_chars = sum(len(chunk.content) for chunk in chunks if hasattr(chunk, 'content'))
            if total_chars > 1000:  # Reasonable amount of content
                score += 0.3
            
            # Check mathematical content detection
            math_chunks = [c for c in chunks if hasattr(c, 'metadata') and c.metadata.get('has_math', False)]
            if math_chunks:
                score += 0.3
            
            # Check formula extraction
            total_formulas = sum(len(c.formulas) for c in chunks if hasattr(c, 'formulas'))
            if total_formulas > 0:
                score += 0.2
            
            # Check confidence scores
            confidences = [c.confidence_score for c in chunks if hasattr(c, 'confidence_score')]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                score += avg_confidence * 0.2
            
            return min(score, 1.0)
        
        except Exception as e:
            logger.warning(f"Error calculating quality score: {e}")
            return 0.5  # Default score