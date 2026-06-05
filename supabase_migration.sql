-- Supabase SQL migration: run this in the Supabase SQL Editor
-- Creates the assessment_results table and sets up RLS policies

CREATE TABLE IF NOT EXISTS public.assessment_results (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id  TEXT NOT NULL,
    style       TEXT NOT NULL CHECK (style IN ('Visual', 'Auditory', 'Reading/Writing', 'Kinesthetic')),
    confidence  NUMERIC(5, 2) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now() NOT NULL
);

-- Index for looking up results by session
CREATE INDEX IF NOT EXISTS idx_assessment_session ON public.assessment_results (session_id);

-- Enable Row Level Security
ALTER TABLE public.assessment_results ENABLE ROW LEVEL SECURITY;

-- Allow anonymous inserts (your backend uses the anon key)
CREATE POLICY "Allow anon insert" ON public.assessment_results
    FOR INSERT WITH CHECK (true);

-- Allow read by session_id
CREATE POLICY "Allow read by session" ON public.assessment_results
    FOR SELECT USING (true);
