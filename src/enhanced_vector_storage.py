# src/enhanced_vector_storage.py
import os
import asyncio
import asyncpg
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
from dotenv import load_dotenv
from mistralai import Mistral
from enhanced_pdf_processor import EnhancedMathPDFProcessor, MathPDFChunk

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMaterialProcessor:
    """Enhanced material processor with better PDF handling"""
    
    def __init__(self, database_url: str, mistral_api_key: str):
        self.database_url = database_url
        self.embedder = MistralEmbedder(mistral_api_key)
        self.pdf_processor = EnhancedMathPDFProcessor(
            preserve_layout=True, 
            extract_images=True
        )
        self.pool = None
    
    async def initialize(self):
        """Initialize database connections"""
        self.pool = await asyncpg.create_pool(self.database_url)
        await self._ensure_enhanced_tables()
        logger.info("Enhanced material processor initialized")
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connections closed")
    
    async def _ensure_enhanced_tables(self):
        """Create enhanced tables for better chunk storage"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                -- Enhanced chunks table with mathematical content support
                CREATE TABLE IF NOT EXISTS enhanced_chunks (
                    id SERIAL PRIMARY KEY,
                    material_id INTEGER REFERENCES course_materials(id) ON DELETE CASCADE,
                    embedding_id INTEGER REFERENCES embeddings(id) ON DELETE CASCADE,
                    
                    -- Content information
                    content TEXT NOT NULL,
                    content_preview TEXT,
                    
                    -- Position and structure
                    page_number INTEGER,
                    chunk_index INTEGER,
                    bbox FLOAT[],
                    
                    -- Mathematical content
                    formulas TEXT[],
                    math_symbols TEXT[],
                    theorems TEXT[],
                    definitions TEXT[],
                    
                    -- Content classification
                    chunk_type TEXT,
                    difficulty_level TEXT,
                    importance_score FLOAT DEFAULT 0.0,
                    confidence_score FLOAT DEFAULT 1.0,
                    
                    -- Images and tables
                    has_images BOOLEAN DEFAULT FALSE,
                    has_tables BOOLEAN DEFAULT FALSE,
                    image_count INTEGER DEFAULT 0,
                    
                    -- Quality metrics
                    char_count INTEGER,
                    formula_count INTEGER,
                    extraction_method TEXT,
                    
                    -- Metadata
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                
                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_material_id ON enhanced_chunks(material_id);
                CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_embedding_id ON enhanced_chunks(embedding_id);
                CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_page ON enhanced_chunks(page_number);
                CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_type ON enhanced_chunks(chunk_type);
                CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_importance ON enhanced_chunks(importance_score DESC);
                CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_formulas ON enhanced_chunks USING GIN(formulas);
                CREATE INDEX IF NOT EXISTS idx_enhanced_chunks_math_symbols ON enhanced_chunks USING GIN(math_symbols);
                
                -- Table for storing formula references and relationships
                CREATE TABLE IF NOT EXISTS formula_references (
                    id SERIAL PRIMARY KEY,
                    chunk_id INTEGER REFERENCES enhanced_chunks(id) ON DELETE CASCADE,
                    formula_text TEXT NOT NULL,
                    formula_type TEXT, -- equation, inequality, function, etc.
                    variables TEXT[],
                    constants TEXT[],
                    operations TEXT[],
                    complexity_score FLOAT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_formula_refs_chunk_id ON formula_references(chunk_id);
                CREATE INDEX IF NOT EXISTS idx_formula_refs_type ON formula_references(formula_type);
                CREATE INDEX IF NOT EXISTS idx_formula_refs_variables ON formula_references USING GIN(variables);
            """)
    
    async def process_pdf_with_enhanced_extraction(self, file_path: str, course_id: int, 
                                                 topic_id: int, uploaded_by: int) -> bool:
        """Process PDF with enhanced mathematical content extraction"""
        try:
            file_path_obj = Path(file_path)
            file_name = file_path_obj.name
            file_type = file_path_obj.suffix.lower()
            
            logger.info(f"Starting enhanced processing of {file_name}")
            
            # Step 1: Enhanced PDF processing
            processing_result = self.pdf_processor.process_pdf_comprehensive(file_path)
            
            if not processing_result["success"]:
                logger.error(f"PDF processing failed: {processing_result.get('error', 'Unknown error')}")
                return False
            
            chunks = processing_result["chunks"]
            stats = processing_result["stats"]
            
            logger.info(f"Extracted {len(chunks)} optimized chunks with {stats['formula_count']} formulas")
            
            # Step 2: Store material metadata
            material_id = await self._store_material_metadata(
                course_id, topic_id, file_name, file_path, file_type, uploaded_by, stats
            )
            
            # Step 3: Process and store chunks with embeddings
            successful_chunks = 0
            for chunk in chunks:
                try:
                    success = await self._process_and_store_chunk(material_id, chunk)
                    if success:
                        successful_chunks += 1
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk.chunk_index}: {e}")
            
            # Step 4: Update material with processing results
            await self._update_material_processing_status(material_id, successful_chunks, len(chunks))
            
            logger.info(f"Successfully processed {successful_chunks}/{len(chunks)} chunks for {file_name}")
            return successful_chunks > 0
            
        except Exception as e:
            logger.error(f"Error in enhanced PDF processing: {e}")
            return False
    
    async def _store_material_metadata(self, course_id: int, topic_id: int, file_name: str, 
                                     file_path: str, file_type: str, uploaded_by: int, 
                                     stats: Dict[str, Any]) -> int:
        """Store material with enhanced metadata"""
        async with self.pool.acquire() as conn:
            material_id = await conn.fetchval("""
                INSERT INTO course_materials 
                (course_id, topic_id, material_name, file_path, file_type, uploaded_by)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """, course_id, topic_id, file_name, file_path, file_type, uploaded_by)
            
            # Store processing statistics
            await conn.execute("""
                INSERT INTO material_processing_stats 
                (material_id, total_chunks, math_chunks, formula_count, quality_score, processing_stats)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (material_id) DO UPDATE SET
                    total_chunks = EXCLUDED.total_chunks,
                    math_chunks = EXCLUDED.math_chunks,
                    formula_count = EXCLUDED.formula_count,
                    quality_score = EXCLUDED.quality_score,
                    processing_stats = EXCLUDED.processing_stats,
                    updated_at = NOW()
            """, material_id, stats.get('total_chunks', 0), stats.get('math_chunks', 0),
                stats.get('formula_count', 0), stats.get('quality_score', 0.0), stats)
            
            return material_id
    
    async def _ensure_stats_table(self):
        """Ensure processing stats table exists"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS material_processing_stats (
                    id SERIAL PRIMARY KEY,
                    material_id INTEGER REFERENCES course_materials(id) ON DELETE CASCADE UNIQUE,
                    total_chunks INTEGER DEFAULT 0,
                    math_chunks INTEGER DEFAULT 0,
                    formula_count INTEGER DEFAULT 0,
                    quality_score FLOAT DEFAULT 0.0,
                    processing_stats JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_stats_material_id ON material_processing_stats(material_id);
            """)
    
    async def _process_and_store_chunk(self, material_id: int, chunk: MathPDFChunk) -> bool:
        """Process and store individual chunk with embedding"""
        try:
            # Generate embedding
            embedding = await self.embedder.embed_text(chunk.content)
            if not embedding:
                logger.warning(f"Failed to generate embedding for chunk {chunk.chunk_index}")
                return False
            
            async with self.pool.acquire() as conn:
                # Store embedding
                embedding_id = await conn.fetchval("""
                    INSERT INTO embeddings (material_id, embedding)
                    VALUES ($1, $2)
                    RETURNING id
                """, material_id, embedding)
                
                # Extract mathematical entities
                math_entities = self._analyze_mathematical_content(chunk)
                
                # Store enhanced chunk
                chunk_id = await conn.fetchval("""
                    INSERT INTO enhanced_chunks (
                        material_id, embedding_id, content, content_preview,
                        page_number, chunk_index, bbox, formulas, math_symbols,
                        chunk_type, importance_score, confidence_score,
                        has_images, has_tables, image_count, char_count,
                        formula_count, extraction_method, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                    RETURNING id
                """, 
                    material_id, embedding_id, chunk.content, chunk.content[:500],
                    chunk.page_number, chunk.chunk_index, chunk.bbox, chunk.formulas,
                    math_entities['symbols'], chunk.chunk_type.value if hasattr(chunk, 'chunk_type') else 'paragraph',
                    chunk.importance_score, chunk.confidence_score,
                    len(chunk.images) > 0, len(chunk.tables) > 0, len(chunk.images),
                    len(chunk.content), len(chunk.formulas),
                    chunk.metadata.get('extraction_method', 'unknown'), chunk.metadata
                )
                
                # Store formula references
                await self._store_formula_references(chunk_id, chunk.formulas)
                
                return True
                
        except Exception as e:
            logger.error(f"Error storing chunk: {e}")
            return False
    
    def _analyze_mathematical_content(self, chunk: MathPDFChunk) -> Dict[str, List[str]]:
        """Analyze mathematical content to extract symbols, operations, etc."""
        content = chunk.content
        
        # Mathematical symbols
        symbols = []
        symbol_patterns = [
            r'[αβγδεζηθικλμνξοπρστυφχψω]',  # Greek letters
            r'[∀∃∈∉⊂⊆∪∩∅ℝℕℤℚℂ∑∏∫∂∇Δ∞±≤≥≠≈√∛∜]',  # Math symbols
            r'[⊕⊗⊙⊥‖⌊⌋⌈⌉]',  # Additional symbols
        ]
        
        for pattern in symbol_patterns:
            matches = re.findall(pattern, content)
            symbols.extend(matches)
        
        # Mathematical operations
        operations = []
        operation_patterns = [
            r'\b(?:sin|cos|tan|log|ln|exp|sqrt|sum|int|lim|max|min)\b',
            r'[+\-*/=<>≤≥]',
            r'\b(?:derivative|integral|limit|summation)\b'
        ]
        
        for pattern in operation_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            operations.extend(matches)
        
        # Variables (single letters, especially italic or in math context)
        variables = re.findall(r'\b[a-zA-Z]\b', content)
        
        return {
            'symbols': list(set(symbols)),
            'operations': list(set(operations)),
            'variables': list(set(variables))
        }
    
    async def _store_formula_references(self, chunk_id: int, formulas: List[str]):
        """Store detailed formula analysis"""
        if not formulas:
            return
        
        async with self.pool.acquire() as conn:
            for formula in formulas:
                # Analyze formula components
                analysis = self._analyze_formula(formula)
                
                await conn.execute("""
                    INSERT INTO formula_references 
                    (chunk_id, formula_text, formula_type, variables, constants, operations, complexity_score)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, chunk_id, formula, analysis['type'], analysis['variables'],
                    analysis['constants'], analysis['operations'], analysis['complexity'])
    
    def _analyze_formula(self, formula: str) -> Dict[str, Any]:
        """Analyze individual formula"""
        # Basic formula analysis
        variables = re.findall(r'\b[a-zA-Z]\b', formula)
        constants = re.findall(r'\b\d+\.?\d*\b', formula)
        operations = re.findall(r'[+\-*/=<>≤≥∑∏∫]', formula)
        
        # Determine formula type
        formula_type = 'unknown'
        if '=' in formula:
            formula_type = 'equation'
        elif any(op in formula for op in ['<', '>', '≤', '≥']):
            formula_type = 'inequality'
        elif any(func in formula.lower() for func in ['sin', 'cos', 'log', 'exp']):
            formula_type = 'function'
        elif '∫' in formula:
            formula_type = 'integral'
        elif '∑' in formula:
            formula_type = 'summation'
        
        # Calculate complexity score
        complexity = len(variables) * 0.3 + len(operations) * 0.5 + len(constants) * 0.2
        
        return {
            'type': formula_type,
            'variables': list(set(variables)),
            'constants': list(set(constants)),
            'operations': list(set(operations)),
            'complexity': min(complexity, 10.0)  # Cap at 10
        }
    
    async def _update_material_processing_status(self, material_id: int, successful_chunks: int, total_chunks: int):
        """Update material processing status"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE course_materials 
                SET embedding_id = (
                    SELECT e.id FROM embeddings e 
                    WHERE e.material_id = $1 
                    LIMIT 1
                )
                WHERE id = $1
            """, material_id)
    
    async def enhanced_similarity_search(self, query: str, course_id: Optional[int] = None,
                                       topic_id: Optional[int] = None, 
                                       formula_search: bool = False,
                                       difficulty_filter: Optional[str] = None,
                                       min_importance: float = 0.0,
                                       limit: int = 10) -> List[Dict[str, Any]]:
        """Enhanced similarity search with mathematical content awareness"""
        try:
            # Generate embedding for query
            query_embedding = await self.embedder.embed_text(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            async with self.pool.acquire() as conn:
                # Build dynamic query based on filters
                where_conditions = ["e.embedding IS NOT NULL"]
                params = [query_embedding]
                param_count = 1
                
                if course_id:
                    param_count += 1
                    where_conditions.append(f"cm.course_id = ${param_count}")
                    params.append(course_id)
                
                if topic_id:
                    param_count += 1
                    where_conditions.append(f"cm.topic_id = ${param_count}")
                    params.append(topic_id)
                
                if min_importance > 0:
                    param_count += 1
                    where_conditions.append(f"ec.importance_score >= ${param_count}")
                    params.append(min_importance)
                
                if difficulty_filter:
                    param_count += 1
                    where_conditions.append(f"ec.difficulty_level = ${param_count}")
                    params.append(difficulty_filter)
                
                # Formula-specific search
                if formula_search:
                    # Look for chunks with mathematical content
                    where_conditions.append("(ec.formula_count > 0 OR array_length(ec.formulas, 1) > 0)")
                
                where_clause = " AND ".join(where_conditions)
                
                query_sql = f"""
                    SELECT 
                        cm.id as material_id,
                        cm.material_name,
                        cm.file_path,
                        cm.file_type,
                        c.course_name,
                        t.topic_name,
                        ec.page_number,
                        ec.chunk_index,
                        ec.content,
                        ec.content_preview,
                        ec.formulas,
                        ec.math_symbols,
                        ec.chunk_type,
                        ec.importance_score,
                        ec.confidence_score,
                        ec.formula_count,
                        ec.char_count,
                        ec.extraction_method,
                        ec.metadata,
                        e.embedding <=> $1 as similarity_distance,
                        
                        -- Add formula relevance scoring
                        CASE 
                            WHEN ec.formula_count > 0 THEN ec.importance_score * 1.2
                            ELSE ec.importance_score
                        END as relevance_score
                        
                    FROM course_materials cm
                    JOIN embeddings e ON cm.id = e.material_id
                    JOIN enhanced_chunks ec ON e.id = ec.embedding_id
                    LEFT JOIN courses c ON cm.course_id = c.id
                    LEFT JOIN topics t ON cm.topic_id = t.id
                    WHERE {where_clause}
                    ORDER BY 
                        relevance_score DESC,
                        e.embedding <=> $1 ASC
                    LIMIT ${param_count + 1}
                """
                
                params.append(limit)
                results = await conn.fetch(query_sql, *params)
                
                # Enrich results with formula details if requested
                enriched_results = []
                for row in results:
                    result_dict = dict(row)
                    
                    if formula_search and result_dict['formula_count'] > 0:
                        # Get detailed formula information
                        formula_details = await conn.fetch("""
                            SELECT formula_text, formula_type, variables, operations, complexity_score
                            FROM formula_references 
                            WHERE chunk_id = $1
                            ORDER BY complexity_score DESC
                        """, result_dict.get('chunk_id'))
                        
                        result_dict['formula_details'] = [dict(f) for f in formula_details]
                    
                    enriched_results.append(result_dict)
                
                return enriched_results
        
        except Exception as e:
            logger.error(f"Error in enhanced similarity search: {e}")
            return []
    
    async def get_material_analytics(self, material_id: Optional[int] = None, 
                                   course_id: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive analytics for materials"""
        try:
            async with self.pool.acquire() as conn:
                where_condition = "1=1"
                params = []
                
                if material_id:
                    where_condition = "cm.id = $1"
                    params = [material_id]
                elif course_id:
                    where_condition = "cm.course_id = $1"
                    params = [course_id]
                
                # Get overall statistics
                stats = await conn.fetchrow(f"""
                    SELECT 
                        COUNT(DISTINCT cm.id) as total_materials,
                        COUNT(ec.id) as total_chunks,
                        SUM(ec.formula_count) as total_formulas,
                        AVG(ec.importance_score) as avg_importance,
                        AVG(ec.confidence_score) as avg_confidence,
                        SUM(ec.char_count) as total_characters,
                        COUNT(CASE WHEN ec.formula_count > 0 THEN 1 END) as math_chunks
                    FROM course_materials cm
                    LEFT JOIN enhanced_chunks ec ON cm.id = ec.material_id
                    WHERE {where_condition}
                """, *params)
                
                # Get chunk type distribution
                chunk_types = await conn.fetch(f"""
                    SELECT 
                        ec.chunk_type,
                        COUNT(*) as count,
                        AVG(ec.importance_score) as avg_importance
                    FROM course_materials cm
                    JOIN enhanced_chunks ec ON cm.id = ec.material_id
                    WHERE {where_condition}
                    GROUP BY ec.chunk_type
                    ORDER BY count DESC
                """, *params)
                
                # Get formula complexity distribution
                formula_stats = await conn.fetch(f"""
                    SELECT 
                        fr.formula_type,
                        COUNT(*) as count,
                        AVG(fr.complexity_score) as avg_complexity
                    FROM course_materials cm
                    JOIN enhanced_chunks ec ON cm.id = ec.material_id
                    JOIN formula_references fr ON ec.id = fr.chunk_id
                    WHERE {where_condition}
                    GROUP BY fr.formula_type
                    ORDER BY count DESC
                """, *params)
                
                return {
                    "overall_stats": dict(stats) if stats else {},
                    "chunk_type_distribution": [dict(row) for row in chunk_types],
                    "formula_statistics": [dict(row) for row in formula_stats],
                    "success": True
                }
        
        except Exception as e:
            logger.error(f"Error getting material analytics: {e}")
            return {"success": False, "error": str(e)}

class MistralEmbedder:
    """Enhanced Mistral embedder with better error handling"""
    
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-embed"
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text with error handling"""
        try:
            # Clean text for better embeddings
            cleaned_text = self._clean_text_for_embedding(text)
            
            response = self.client.embeddings.create(
                model=self.model,
                inputs=[cleaned_text]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def _clean_text_for_embedding(self, text: str) -> str:
        """Clean text while preserving mathematical content"""
        # Remove excessive whitespace but preserve structure
        cleaned = re.sub(r'\n\s*\n', '\n\n', text)
        cleaned = re.sub(r' +', ' ', cleaned)
        
        # Preserve mathematical notation markers
        cleaned = cleaned.strip()
        
        # Ensure reasonable length for embedding
        if len(cleaned) > 8000:  # Mistral's context limit
            cleaned = cleaned[:8000] + "..."
        
        return cleaned