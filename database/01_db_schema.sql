-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- USERS
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- COURSES
CREATE TABLE courses (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  course_name TEXT NOT NULL,
  course_code TEXT NOT NULL,
  department TEXT,
  created_by UUID REFERENCES users(id),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- TOPICS
CREATE TABLE topics (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  course_id UUID REFERENCES courses(id) ON DELETE CASCADE,
  topic_name TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- QUESTION FORMATS
CREATE TABLE question_formats (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  format_name TEXT NOT NULL,
  description TEXT
);

-- DIFFICULTY LEVELS
CREATE TABLE difficulty_levels (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  level_name TEXT NOT NULL,
  description TEXT
);

-- COURSE MATERIALS (embedding_id defined without foreign key initially)
CREATE TABLE course_materials (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  course_id UUID REFERENCES courses(id) ON DELETE CASCADE,
  topic_id UUID REFERENCES topics(id),
  material_name TEXT NOT NULL,
  file_path TEXT NOT NULL,
  file_type TEXT,
  embedding_id UUID, -- No REFERENCES clause here yet
  uploaded_at TIMESTAMP DEFAULT NOW(),
  uploaded_by UUID REFERENCES users(id)
);

-- EMBEDDINGS
CREATE TABLE embeddings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  material_id UUID REFERENCES course_materials(id) ON DELETE CASCADE,
  embedding VECTOR(1024),  -- For Mistral AI mistral-embed (1024 dimensions)
  created_at TIMESTAMP DEFAULT NOW()
);

-- Add foreign key constraint for embedding_id after embeddings table is created
ALTER TABLE course_materials
ADD CONSTRAINT fk_embedding_id
FOREIGN KEY (embedding_id) REFERENCES embeddings(id);

-- GENERATION REQUESTS
CREATE TABLE generation_requests (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES users(id),
  course_id UUID REFERENCES courses(id),
  topic_id UUID REFERENCES topics(id),
  format_id UUID REFERENCES question_formats(id),
  difficulty_id UUID REFERENCES difficulty_levels(id),
  custom_instructions TEXT,
  status TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  completed_at TIMESTAMP
);

-- REQUEST MATERIALS
CREATE TABLE request_materials (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  request_id UUID REFERENCES generation_requests(id),
  material_id UUID REFERENCES course_materials(id)
);

-- QUESTIONS TABLE
CREATE TABLE questions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  course_id UUID REFERENCES courses(id),
  topic_id UUID REFERENCES topics(id),
  format_id UUID REFERENCES question_formats(id),
  difficulty_id UUID REFERENCES difficulty_levels(id),
  question_text TEXT NOT NULL,
  created_by UUID REFERENCES users(id),
  approved BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- ANSWERS TABLE (includes correct + optional distractors)
CREATE TABLE answers (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  question_id UUID REFERENCES questions(id) ON DELETE CASCADE,
  answer_text TEXT NOT NULL,
  is_correct BOOLEAN DEFAULT FALSE
);

-- GENERATED QUESTIONS (linked to generation request)
CREATE TABLE generated_questions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  request_id UUID REFERENCES generation_requests(id),
  question_id UUID REFERENCES questions(id),
  correct_answer TEXT NOT NULL,
  solution_steps TEXT,
  answer_options JSONB,
  validation_status TEXT NOT NULL,
  feedback TEXT,
  validation_attempts INT DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- QUESTION VALIDATIONS
CREATE TABLE question_validations (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  question_id UUID REFERENCES questions(id),
  validation_result TEXT NOT NULL,
  validation_feedback TEXT,
  validated_at TIMESTAMP DEFAULT NOW()
);

-- APPROVED QUESTIONS
CREATE TABLE approved_questions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  question_id UUID REFERENCES questions(id),
  approved_by UUID REFERENCES users(id),
  modifications TEXT,
  approved_at TIMESTAMP DEFAULT NOW()
);

-- AGENT LOGS
CREATE TABLE agent_logs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  request_id UUID REFERENCES generation_requests(id),
  question_id UUID REFERENCES questions(id),
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

-- Indexes for performance
CREATE INDEX idx_courses_created_by ON courses (created_by);
CREATE INDEX idx_topics_course_id ON topics (course_id);
CREATE INDEX idx_course_materials_course_id ON course_materials (course_id);
CREATE INDEX idx_course_materials_topic_id ON course_materials (topic_id);
CREATE INDEX idx_course_materials_embedding_id ON course_materials (embedding_id);
CREATE INDEX idx_course_materials_uploaded_by ON course_materials (uploaded_by);
CREATE INDEX idx_embeddings_material_id ON embeddings (material_id);
CREATE INDEX idx_embeddings_embedding ON embeddings USING ivfflat (embedding vector_l2_ops);
CREATE INDEX idx_generation_requests_user_id ON generation_requests (user_id);
CREATE INDEX idx_generation_requests_course_id ON generation_requests (course_id);
CREATE INDEX idx_generation_requests_topic_id ON generation_requests (topic_id);
CREATE INDEX idx_generation_requests_format_id ON generation_requests (format_id);
CREATE INDEX idx_generation_requests_difficulty_id ON generation_requests (difficulty_id);
CREATE INDEX idx_request_materials_request_id ON request_materials (request_id);
CREATE INDEX idx_request_materials_material_id ON request_materials (material_id);
CREATE INDEX idx_questions_course_id ON questions (course_id);
CREATE INDEX idx_questions_topic_id ON questions (topic_id);
CREATE INDEX idx_questions_format_id ON questions (format_id);
CREATE INDEX idx_questions_difficulty_id ON questions (difficulty_id);
CREATE INDEX idx_questions_created_by ON questions (created_by);
CREATE INDEX idx_answers_question_id ON answers (question_id);
CREATE INDEX idx_generated_questions_request_id ON generated_questions (request_id);
CREATE INDEX idx_generated_questions_question_id ON generated_questions (question_id);
CREATE INDEX idx_question_validations_question_id ON question_validations (question_id);
CREATE INDEX idx_approved_questions_question_id ON approved_questions (question_id);
CREATE INDEX idx_approved_questions_approved_by ON approved_questions (approved_by);
CREATE INDEX idx_agent_logs_request_id ON agent_logs (request_id);
CREATE INDEX idx_agent_logs_question_id ON agent_logs (question_id);