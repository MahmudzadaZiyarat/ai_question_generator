--run the command -> docker exec -it math_rag_db psql -U rag_user -d db_math_rag
-- View all users
SELECT * FROM users;

-- View all courses
SELECT * FROM courses;

-- View all topics
SELECT * FROM topics;

-- View all question formats
SELECT * FROM question_formats;

-- View all difficulty levels
SELECT * FROM difficulty_levels;

-- View all course materials
SELECT * FROM course_materials;

-- View all embeddings
SELECT * FROM embeddings;