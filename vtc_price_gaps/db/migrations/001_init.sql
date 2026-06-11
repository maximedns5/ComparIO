CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS h3;
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- M1: Données brutes nettoyées
CREATE TABLE IF NOT EXISTS trips_clean (
    trip_id              TEXT PRIMARY KEY,
    city_id              TEXT NOT NULL,
    provider             TEXT NOT NULL,
    pickup_lat           DOUBLE PRECISION NOT NULL,
    pickup_lon           DOUBLE PRECISION NOT NULL,
    dropoff_lat          DOUBLE PRECISION NOT NULL,
    dropoff_lon          DOUBLE PRECISION NOT NULL,
    pickup_geom          GEOMETRY(Point, 4326),
    dropoff_geom         GEOMETRY(Point, 4326),
    pickup_ts            TIMESTAMPTZ NOT NULL,
    trip_duration_s      INTEGER,
    distance_km          DOUBLE PRECISION,
    price                DOUBLE PRECISION NOT NULL,
    is_special_day       BOOLEAN DEFAULT FALSE,
    data_quality_score   REAL DEFAULT 1.0,
    ingested_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_trips_pickup_geom ON trips_clean USING GIST(pickup_geom);
CREATE INDEX IF NOT EXISTS idx_trips_ts ON trips_clean(pickup_ts DESC);
CREATE INDEX IF NOT EXISTS idx_trips_city ON trips_clean(city_id);

-- M2: Features par trajet
CREATE TABLE IF NOT EXISTS trip_features (
    trip_id                  TEXT PRIMARY KEY REFERENCES trips_clean(trip_id),
    hour_sin REAL, hour_cos REAL, dow_sin REAL, dow_cos REAL,
    month_sin REAL, month_cos REAL,
    is_weekend BOOLEAN, is_rush_am BOOLEAN, is_rush_pm BOOLEAN, is_night BOOLEAN,
    minutes_since_midnight INTEGER,
    distance_km REAL, bearing_sin REAL, bearing_cos REAL,
    delta_lat REAL, delta_lon REAL,
    h3_pickup_r6 TEXT, h3_pickup_r7 TEXT, h3_pickup_r8 TEXT, h3_pickup_r9 TEXT,
    h3_dropoff_r7 TEXT, h3_dropoff_r8 TEXT,
    h3r8_price_median_7d REAL, h3r8_price_std_7d REAL, h3r8_trip_count_7d INTEGER,
    h3r8_duration_med_7d REAL, h3r8_price_median_30d REAL, h3r8_price_p10_30d REAL,
    h3r8_neighbor_mean REAL, h3r8_price_vs_neighbors REAL, h3r8_is_minimum BOOLEAN,
    price REAL NOT NULL,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

-- M2: Agrégats H3 par cellule
CREATE TABLE IF NOT EXISTS h3_price_stats (
    h3_cell_id            TEXT NOT NULL,
    resolution            SMALLINT NOT NULL,
    city_id               TEXT NOT NULL,
    window_days           SMALLINT NOT NULL,
    computed_at           TIMESTAMPTZ NOT NULL,
    price_median          DOUBLE PRECISION,
    price_mean            DOUBLE PRECISION,
    price_std             DOUBLE PRECISION,
    price_p10             DOUBLE PRECISION,
    price_p90             DOUBLE PRECISION,
    trip_count            INTEGER,
    duration_median       DOUBLE PRECISION,
    neighbor_price_mean   DOUBLE PRECISION,
    price_vs_neighbors    DOUBLE PRECISION,
    is_price_minimum      BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (h3_cell_id, resolution, window_days, computed_at)
);
CREATE INDEX IF NOT EXISTS idx_h3stats_cell ON h3_price_stats(h3_cell_id, resolution, window_days);

-- M4: Cache des opportunités de prix
CREATE TABLE IF NOT EXISTS price_opportunities (
    h3_source_cell   TEXT NOT NULL,
    h3_target_cell   TEXT NOT NULL,
    resolution       SMALLINT NOT NULL,
    city_id          TEXT NOT NULL,
    hour_bucket      SMALLINT NOT NULL,
    dow_bucket       SMALLINT NOT NULL,
    is_weekend       BOOLEAN,
    gain_median_eur  REAL,
    gain_pct         REAL,
    walk_dist_m      REAL,
    stability_score  REAL,
    confidence       REAL,
    direction_deg    REAL,
    valid_from       TIMESTAMPTZ DEFAULT NOW(),
    valid_until      TIMESTAMPTZ,
    PRIMARY KEY (h3_source_cell, h3_target_cell, hour_bucket, dow_bucket)
);
CREATE INDEX IF NOT EXISTS idx_opp_source ON price_opportunities(h3_source_cell);

SELECT cron.schedule(
  'invalidate-opp-cache', '0 * * * *',
  $$DELETE FROM price_opportunities WHERE valid_until < NOW();$$
);