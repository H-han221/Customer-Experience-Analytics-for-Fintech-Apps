-- banks table
CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name TEXT NOT NULL UNIQUE,
    app_package TEXT NOT NULL UNIQUE
);

-- reviews table
CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY,
    bank_id INTEGER NOT NULL REFERENCES banks(bank_id),
    review_text TEXT NOT NULL,
    rating SMALLINT,
    review_date DATE,
    sentiment_label TEXT,
    sentiment_score NUMERIC,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
