-- Grant privileges to rag_user
GRANT ALL PRIVILEGES ON DATABASE db_math_rag TO rag_user;
GRANT USAGE ON SCHEMA public TO rag_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO rag_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO rag_user;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO rag_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO rag_user;

-- Add pgvector extension for vector support
CREATE EXTENSION IF NOT EXISTS vector;

-- USERS
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- COURSES
CREATE TABLE courses (
  id SERIAL PRIMARY KEY,
  course_name TEXT NOT NULL,
  course_code TEXT NOT NULL,
  department TEXT,
  created_by INTEGER REFERENCES users(id),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- TOPICS
CREATE TABLE topics (
  id SERIAL PRIMARY KEY,
  course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
  topic_name TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- QUESTION FORMATS
CREATE TABLE question_formats (
  id SERIAL PRIMARY KEY,
  format_name TEXT NOT NULL,
  description TEXT
);

-- DIFFICULTY LEVELS
CREATE TABLE difficulty_levels (
  id SERIAL PRIMARY KEY,
  level_name TEXT NOT NULL,
  description TEXT
);

-- COURSE MATERIALS (with embedding_id added but constraint deferred)
CREATE TABLE course_materials (
  id SERIAL PRIMARY KEY,
  course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
  topic_id INTEGER REFERENCES topics(id),
  material_name TEXT NOT NULL,
  file_path TEXT NOT NULL,
  file_type TEXT,
  embedding_id INTEGER,  -- Added the missing column
  uploaded_at TIMESTAMP DEFAULT NOW(),
  uploaded_by INTEGER REFERENCES users(id)
);

-- EMBEDDINGS
CREATE TABLE embeddings (
  id SERIAL PRIMARY KEY,
  material_id INTEGER REFERENCES course_materials(id) ON DELETE CASCADE,
  embedding vector(1024),  -- For Mistral AI mistral-embed (1024 dimensions)
  created_at TIMESTAMP DEFAULT NOW()
);

-- Add foreign key constraint for embedding_id after embeddings table is created
ALTER TABLE course_materials
ADD CONSTRAINT fk_embedding_id
FOREIGN KEY (embedding_id) REFERENCES embeddings(id);

-- GENERATION REQUESTS
CREATE TABLE generation_requests (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id),
  course_id INTEGER REFERENCES courses(id),
  topic_id INTEGER REFERENCES topics(id),
  format_id INTEGER REFERENCES question_formats(id),
  difficulty_id INTEGER REFERENCES difficulty_levels(id),
  custom_instructions TEXT,
  status TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  completed_at TIMESTAMP
);

-- REQUEST MATERIALS
CREATE TABLE request_materials (
  id SERIAL PRIMARY KEY,
  request_id INTEGER REFERENCES generation_requests(id),
  material_id INTEGER REFERENCES course_materials(id)
);

-- QUESTIONS TABLE
CREATE TABLE questions (
  id SERIAL PRIMARY KEY,
  course_id INTEGER REFERENCES courses(id),
  topic_id INTEGER REFERENCES topics(id),
  format_id INTEGER REFERENCES question_formats(id),
  difficulty_id INTEGER REFERENCES difficulty_levels(id),
  question_text TEXT NOT NULL,
  created_by INTEGER REFERENCES users(id),
  approved BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- ANSWERS TABLE (includes correct + optional distractors)
CREATE TABLE answers (
  id SERIAL PRIMARY KEY,
  question_id INTEGER REFERENCES questions(id) ON DELETE CASCADE,
  answer_text TEXT NOT NULL,
  is_correct BOOLEAN DEFAULT FALSE
);

-- GENERATED QUESTIONS (linked to generation request)
CREATE TABLE generated_questions (
  id SERIAL PRIMARY KEY,
  request_id INTEGER REFERENCES generation_requests(id),
  question_id INTEGER REFERENCES questions(id),
  correct_answer TEXT NOT NULL,
  solution_steps TEXT,
  answer_options JSONB,
  validation_status TEXT NOT NULL,
  feedback TEXT,
  validation_attempts INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- QUESTION VALIDATIONS
CREATE TABLE question_validations (
  id SERIAL PRIMARY KEY,
  question_id INTEGER REFERENCES questions(id),
  validation_result TEXT NOT NULL,
  validation_feedback TEXT,
  validated_at TIMESTAMP DEFAULT NOW()
);

-- APPROVED QUESTIONS
CREATE TABLE approved_questions (
  id SERIAL PRIMARY KEY,
  question_id INTEGER REFERENCES questions(id),
  approved_by INTEGER REFERENCES users(id),
  modifications TEXT,
  approved_at TIMESTAMP DEFAULT NOW()
);

-- AGENT LOGS
CREATE TABLE agent_logs (
  id SERIAL PRIMARY KEY,
  request_id INTEGER REFERENCES generation_requests(id),
  question_id INTEGER REFERENCES questions(id),
  agent_type TEXT NOT NULL,
  agent_input TEXT,
  agent_output TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for tables with updated_at column
CREATE TRIGGER update_users_updated_at
BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_courses_updated_at
BEFORE UPDATE ON courses
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_topics_updated_at
BEFORE UPDATE ON topics
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_questions_updated_at
BEFORE UPDATE ON questions
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_generated_questions_updated_at
BEFORE UPDATE ON generated_questions
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();