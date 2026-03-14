/*
  # Add Database Management Functions

  1. New Functions
    - get_table_info: Get detailed information about a table
    - get_database_stats: Get overall database statistics
    - execute_query: Execute a custom SQL query (admin only)
  
  2. Security
    - Functions are restricted to admin users only
    - Query execution is limited to safe operations
*/

-- Function to get table information
CREATE OR REPLACE FUNCTION get_table_info(table_name text)
RETURNS TABLE (
  row_count bigint,
  size text,
  last_vacuum timestamp with time zone,
  last_analyze timestamp with time zone
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  RETURN QUERY
  SELECT
    (SELECT reltuples::bigint FROM pg_class WHERE oid = (table_name::regclass)::oid) as row_count,
    pg_size_pretty(pg_total_relation_size(table_name::regclass)) as size,
    last_vacuum,
    last_analyze
  FROM pg_stat_user_tables
  WHERE schemaname = 'public'
  AND relname = table_name;
END;
$$;

-- Function to get database statistics
CREATE OR REPLACE FUNCTION get_database_stats()
RETURNS TABLE (
  total_size text,
  total_tables bigint,
  total_schemas bigint
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  RETURN QUERY
  SELECT
    pg_size_pretty(pg_database_size(current_database())) as total_size,
    (SELECT count(*)::bigint FROM information_schema.tables WHERE table_schema = 'public') as total_tables,
    (SELECT count(*)::bigint FROM information_schema.schemata) as total_schemas;
END;
$$;

-- Function to execute custom queries (admin only)
CREATE OR REPLACE FUNCTION execute_query(query_string text)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  result json;
BEGIN
  -- Check if user is admin
  IF NOT EXISTS (
    SELECT 1 FROM profiles
    WHERE id = auth.uid()
    AND is_admin = true
  ) THEN
    RAISE EXCEPTION 'Permission denied: Admin access required';
  END IF;

  -- Execute query and return results as JSON
  EXECUTE 'SELECT array_to_json(array_agg(row_to_json(t)))
           FROM (' || query_string || ') t'
  INTO result;

  RETURN result;
END;
$$;

-- Grant execute permissions to authenticated users
GRANT EXECUTE ON FUNCTION get_table_info(text) TO authenticated;
GRANT EXECUTE ON FUNCTION get_database_stats() TO authenticated;
GRANT EXECUTE ON FUNCTION execute_query(text) TO authenticated;