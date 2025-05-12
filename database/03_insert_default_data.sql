
-- Insert default data
INSERT INTO question_formats (format_name, description) VALUES
('Multiple Choice', 'Questions with multiple options and one correct answer'),
('Short Answer', 'Questions requiring brief textual responses'),
('Computational', 'Questions requiring mathematical calculations'),
('Proof', 'Questions requiring mathematical proofs');

INSERT INTO difficulty_levels (level_name, description) VALUES
('Easy', 'Basic concepts and straightforward applications'),
('Medium', 'More complex applications requiring deeper understanding'),
('Hard', 'Advanced concepts requiring synthesis of multiple ideas');

-- Insert default system configs
INSERT INTO system_configs (config_key, config_value, description) VALUES
('retriever_config', 
  '{
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 5,
    "similarity_threshold": 0.75
  }',
  'Configuration for Retriever Agent'),
('generator_config', 
  '{
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.95,
    "prompt_template": "Generate a {{format}} question about {{topic}} at {{difficulty}} level..."
  }',
  'Configuration for Generator Agent'),
('validator_config', 
  '{
    "math_validation": true,
    "context_validation": true,
    "max_validation_attempts": 3
  }',
  'Configuration for Validator Agent'),
('orchestrator_config', 
  '{
    "max_retries": 3,
    "timeout_seconds": 120
  }',
  'Configuration for Orchestrator');