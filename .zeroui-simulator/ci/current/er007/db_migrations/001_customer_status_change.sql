-- ER-007 blocker migration: destructive schema change.
-- Real-world meaning: a release migration tries to drop a production column.
ALTER TABLE customer_accounts DROP COLUMN legacy_status;
