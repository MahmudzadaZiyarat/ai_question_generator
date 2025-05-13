-- Connect to postgres first to enable extension
\c postgres;
CREATE EXTENSION IF NOT EXISTS vector;

-- Connect to the target database
\c db_math_rag;

-- Create the extension in this database too
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant privileges to rag_user
GRANT ALL PRIVILEGES ON DATABASE db_math_rag TO rag_user;
GRANT USAGE ON SCHEMA public TO rag_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO rag_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO rag_user;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO rag_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO rag_user;