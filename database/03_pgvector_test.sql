-- 02_pgvector_test.sql
-- Test the pgvector installation

-- Create a test table with a smaller dimension for testing
CREATE TABLE test_vectors (
  id SERIAL PRIMARY KEY,
  embedding vector(5)  -- Using 5 dimensions for the test
);

-- Insert a test vector with 5 dimensions
INSERT INTO test_vectors (embedding) VALUES ('[0.1, 0.2, 0.3, 0.4, 0.5]');

-- Verify it worked
SELECT * FROM test_vectors;

-- Add sample users
INSERT INTO users (name, email, password_hash, role) 
VALUES ('Admin User', 'admin@example.com', 'hashed_password', 'admin');

-- Add sample courses
INSERT INTO courses (course_name, course_code, department, created_by)
VALUES ('Calculus I', 'MATH101', 'Mathematics', 1);

-- Add sample topics
INSERT INTO topics (course_id, topic_name, description)
VALUES (1, 'Derivatives', 'Introduction to derivatives and differentiation');

-- Add sample question formats
INSERT INTO question_formats (format_name, description)
VALUES ('Multiple Choice', 'Questions with multiple answer options');

-- Add sample difficulty levels
INSERT INTO difficulty_levels (level_name, description)
VALUES ('Easy', 'Basic concepts and straightforward applications');

-- Add sample course materials
INSERT INTO course_materials (
    course_id, 
    topic_id, 
    material_name, 
    file_path, 
    file_type, 
    uploaded_by
)
VALUES (
    1, 
    1, 
    'Derivatives Lecture Notes', 
    '/files/derivatives_notes.pdf', 
    'pdf', 
    1
);

-- For testing the main schema, we can use a zero vector of the correct dimension
CREATE OR REPLACE FUNCTION create_zero_vector(dim INTEGER) 
RETURNS vector AS $$
DECLARE
    result text := '[';
    i INTEGER;
BEGIN
    FOR i IN 1..dim-1 LOOP
        result := result || '0, ';
    END LOOP;
    result := result || '0]';
    RETURN result::vector;
END;
$$ LANGUAGE plpgsql;

-- Insert a zero vector into embeddings table
INSERT INTO embeddings (material_id, embedding)
VALUES (1, create_zero_vector(1024));

-- Update course_materials to reference this embedding
UPDATE course_materials SET embedding_id = 1 WHERE id = 1;

-- Print verification data
SELECT 'Database initialized successfully!' AS status;