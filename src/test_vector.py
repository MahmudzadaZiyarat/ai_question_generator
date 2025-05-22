#src/test_vector.py
"""
Test script to verify B44.pdf storage in the database
"""
import os
import sys
import asyncio
import asyncpg
from pathlib import Path
from typing import Dict, Any
import logging
from dotenv import load_dotenv

# Add src directory to path
sys.path.append('src')

try:
    from vector_storage import MaterialProcessor
except ImportError:
    print("Error: Could not import vector_storage module. Make sure src/vector_storage.py exists.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFTest:
    """Test class specifically for B44.pdf"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        self.database_url = os.getenv('DATABASE_URL')
        self.mistral_api_key = os.getenv('MISTRAL_API_KEY')
        
        # Your specific PDF file
        self.pdf_path = r"D:\ai_question_generator\input_materials\B44.pdf"
        
        if not self.database_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        if not self.mistral_api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        
        self.processor = MaterialProcessor(self.database_url, self.mistral_api_key)
    
    async def verify_pdf_exists(self):
        """Check if the PDF file exists and is readable"""
        pdf_file = Path(self.pdf_path)
        
        if not pdf_file.exists():
            logger.error(f"PDF file not found: {self.pdf_path}")
            return False
        
        if not pdf_file.is_file():
            logger.error(f"Path is not a file: {self.pdf_path}")
            return False
        
        # Check file size
        file_size = pdf_file.stat().st_size
        logger.info(f"PDF file found: {pdf_file.name}")
        logger.info(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        # Try to read the file to ensure it's accessible
        try:
            with open(self.pdf_path, 'rb') as f:
                # Read first few bytes to verify it's a PDF
                header = f.read(4)
                if header != b'%PDF':
                    logger.warning(f"File might not be a valid PDF (header: {header})")
                else:
                    logger.info("✓ Valid PDF file detected")
        except Exception as e:
            logger.error(f"Error reading PDF file: {e}")
            return False
        
        return True
    
    async def setup_test_data(self):
        """Create test data in the database"""
        try:
            conn = await asyncpg.connect(self.database_url)
            
            # Create test user
            user_id = await conn.fetchval("""
                INSERT INTO users (name, email, password_hash, role)
                VALUES ('Test User', 'test_b44@example.com', 'dummy_hash', 'instructor')
                ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name
                RETURNING id
            """)
            
            # Create test course for B44 material
            course_id = await conn.fetchval("""
                INSERT INTO courses (course_name, course_code, department, created_by)
                VALUES ('Engineering Mathematics', 'ENG_B44', 'Engineering', $1)
                RETURNING id
            """, user_id)
            
            # Create test topic
            topic_id = await conn.fetchval("""
                INSERT INTO topics (course_id, topic_name, description)
                VALUES ($1, 'B44 Course Material', 'Material from B44.pdf document')
                RETURNING id
            """, course_id)
            
            await conn.close()
            
            logger.info(f"Created test data - User ID: {user_id}, Course ID: {course_id}, Topic ID: {topic_id}")
            return user_id, course_id, topic_id
            
        except Exception as e:
            logger.error(f"Error setting up test data: {e}")
            raise
    
    async def test_pdf_extraction(self):
        """Test PDF text extraction before processing"""
        try:
            from vector_storage import DocumentProcessor
            
            processor = DocumentProcessor()
            chunks = processor.extract_text_from_pdf(self.pdf_path)
            
            if chunks:
                logger.info(f"✓ Successfully extracted {len(chunks)} pages from PDF")
                
                # Show preview of extracted content
                for i, chunk in enumerate(chunks[:3]):  # Show first 3 pages
                    preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                    logger.info(f"Page {chunk.page_number}: {preview}")
                
                if len(chunks) > 3:
                    logger.info(f"... and {len(chunks) - 3} more pages")
                
                # Calculate total content size
                total_chars = sum(len(chunk.content) for chunk in chunks)
                logger.info(f"Total extracted text: {total_chars:,} characters")
                
                return True
            else:
                logger.error("✗ No text extracted from PDF")
                return False
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return False
    
    async def test_simple_processing(self, course_id: int, topic_id: int, user_id: int):
        """Test simple PDF processing and storage"""
        try:
            logger.info("=== Testing Simple Processing ===")
            
            await self.processor.initialize()
            
            success = await self.processor.process_and_store_material(
                file_path=self.pdf_path,
                course_id=course_id,
                topic_id=topic_id,
                uploaded_by=user_id
            )
            
            if success:
                logger.info("✓ Simple processing completed successfully")
                return True
            else:
                logger.error("✗ Simple processing failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in simple processing: {e}")
            return False
        finally:
            await self.processor.cleanup()
    
    async def test_advanced_processing(self, course_id: int, topic_id: int, user_id: int):
        """Test advanced PDF processing with chunking"""
        try:
            logger.info("=== Testing Advanced Processing ===")
            
            await self.processor.initialize()
            
            success = await self.processor.process_and_store_material_advanced(
                file_path=self.pdf_path,
                course_id=course_id,
                topic_id=topic_id,
                uploaded_by=user_id
            )
            
            if success:
                logger.info("✓ Advanced processing completed successfully")
                return True
            else:
                logger.error("✗ Advanced processing failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in advanced processing: {e}")
            return False
        finally:
            await self.processor.cleanup()
    
    async def verify_stored_data(self):
        """Verify B44.pdf data was stored correctly"""
        try:
            conn = await asyncpg.connect(self.database_url)
            
            # Check course_materials for B44.pdf
            materials = await conn.fetch("""
                SELECT id, material_name, file_path, file_type, embedding_id, uploaded_at
                FROM course_materials
                WHERE material_name LIKE '%B44%' OR file_path LIKE '%B44%'
                ORDER BY id DESC
            """)
            
            logger.info(f"=== B44.pdf Materials in Database ===")
            if materials:
                for material in materials:
                    logger.info(f"Material ID: {material['id']}")
                    logger.info(f"  Name: {material['material_name']}")
                    logger.info(f"  Path: {material['file_path']}")
                    logger.info(f"  Type: {material['file_type']}")
                    logger.info(f"  Embedding ID: {material['embedding_id']}")
                    logger.info(f"  Uploaded: {material['uploaded_at']}")
                    logger.info("")
            else:
                logger.warning("No B44.pdf materials found in database")
                return False
            
            # Check embeddings for B44 materials
            embeddings = await conn.fetch("""
                SELECT e.id, e.material_id, array_length(e.embedding, 1) as embedding_dimension
                FROM embeddings e
                JOIN course_materials cm ON e.material_id = cm.id
                WHERE cm.material_name LIKE '%B44%' OR cm.file_path LIKE '%B44%'
                ORDER BY e.id DESC
            """)
            
            logger.info(f"=== B44.pdf Embeddings ===")
            if embeddings:
                for embedding in embeddings:
                    logger.info(f"Embedding ID: {embedding['id']}")
                    logger.info(f"  Material ID: {embedding['material_id']}")
                    logger.info(f"  Dimensions: {embedding['embedding_dimension']}")
            else:
                logger.warning("No embeddings found for B44.pdf")
                return False
            
            # Check document chunks (if using advanced processing)
            chunks_exist = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'document_chunks'
                )
            """)
            
            if chunks_exist:
                chunks = await conn.fetch("""
                    SELECT dc.id, dc.chunk_type, dc.importance_score, dc.concepts, dc.formulas
                    FROM document_chunks dc
                    JOIN course_materials cm ON dc.material_id = cm.id
                    WHERE cm.material_name LIKE '%B44%' OR cm.file_path LIKE '%B44%'
                    ORDER BY dc.importance_score DESC
                    LIMIT 10
                """)
                
                logger.info(f"=== B44.pdf Chunks (Top 10 by importance) ===")
                for chunk in chunks:
                    logger.info(f"Chunk ID: {chunk['id']}")
                    logger.info(f"  Type: {chunk['chunk_type']}")
                    logger.info(f"  Importance: {chunk['importance_score']:.3f}")
                    logger.info(f"  Concepts: {chunk['concepts']}")
                    logger.info(f"  Formulas: {chunk['formulas']}")
                    logger.info("")
            
            await conn.close()
            return len(materials) > 0 and len(embeddings) > 0
            
        except Exception as e:
            logger.error(f"Error verifying stored data: {e}")
            return False
    
    async def test_search_b44_content(self):
        """Test searching B44.pdf content"""
        try:
            await self.processor.initialize()
            
            # Test searches that might be relevant to B44 content
            test_queries = [
                "engineering mathematics",
                "formula equation",
                "calculation method",
                "mathematical concept",
                "problem solution",
                "B44"  # Direct search for the document
            ]
            
            logger.info("=== Testing Search for B44 Content ===")
            
            for query in test_queries:
                logger.info(f"Searching for: '{query}'")
                results = await self.processor.search_similar_materials(
                    query=query,
                    limit=5
                )
                
                if results:
                    logger.info(f"  Found {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        material_name = result.get('material_name', 'Unknown')
                        distance = result.get('distance', 'N/A')
                        chunk_type = result.get('chunk_type', 'N/A')
                        importance = result.get('importance_score', 'N/A')
                        
                        logger.info(f"    {i}. {material_name}")
                        logger.info(f"       Distance: {distance:.4f if distance != 'N/A' else distance}")
                        if chunk_type != 'N/A':
                            logger.info(f"       Type: {chunk_type}, Importance: {importance:.3f if importance != 'N/A' else importance}")
                else:
                    logger.info("  No results found")
                logger.info("")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in search test: {e}")
            return False
        finally:
            await self.processor.cleanup()
    
    async def run_full_test(self):
        """Run the complete test for B44.pdf"""
        logger.info("=== Starting B44.pdf Storage Test ===")
        logger.info(f"Target file: {self.pdf_path}")
        
        try:
            # Step 1: Verify PDF file exists
            logger.info("\n1. Verifying PDF file...")
            if not await self.verify_pdf_exists():
                logger.error("PDF file verification failed. Exiting.")
                return False
            
            # Step 2: Test PDF text extraction
            logger.info("\n2. Testing PDF text extraction...")
            if not await self.test_pdf_extraction():
                logger.error("PDF text extraction failed. Exiting.")
                return False
            
            # Step 3: Set up test data
            logger.info("\n3. Setting up test data...")
            user_id, course_id, topic_id = await self.setup_test_data()
            
            # Step 4: Test simple processing
            logger.info("\n4. Testing simple processing...")
            simple_success = await self.test_simple_processing(course_id, topic_id, user_id)
            
            # Step 5: Test advanced processing
            logger.info("\n5. Testing advanced processing...")
            advanced_success = await self.test_advanced_processing(course_id, topic_id, user_id)
            
            # Step 6: Verify stored data
            logger.info("\n6. Verifying stored data...")
            if not await self.verify_stored_data():
                logger.error("Data verification failed.")
                return False
            
            # Step 7: Test search functionality
            logger.info("\n7. Testing search functionality...")
            if not await self.test_search_b44_content():
                logger.error("Search test failed.")
                return False
            
            logger.info("\n=== B44.pdf test completed successfully! ===")
            return True
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            return False

async def main():
    """Main function to run the B44.pdf test"""
    test = PDFTest()
    
    success = await test.run_full_test()
    
    if success:
        print("\npdf storage test completed successfully!")
        print("\nYour PDF has been processed and stored in the database.")
        print("You can now:")
        print("1. Use the similarity search to find relevant content")
        print("2. Generate questions based on this material")
        print("3. Add more PDF files to your knowledge base")
    else:
        print("\nB44.pdf storage test failed. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())