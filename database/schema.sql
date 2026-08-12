-- =====================================================================
-- ClaimVision — Healthcare Claims Star Schema DDL
-- Dialect: ANSI SQL / SQLite / PostgreSQL / DuckDB compatible
-- =====================================================================

DROP TABLE IF EXISTS fact_claims_unified;
DROP TABLE IF EXISTS fact_inpatient_claims;
DROP TABLE IF EXISTS fact_outpatient_claims;
DROP TABLE IF EXISTS dim_beneficiaries;
DROP TABLE IF EXISTS dim_providers;

-- ---------------------------------------------------------------------
-- 1. dim_providers: Healthcare institutions/practitioners
-- ---------------------------------------------------------------------
CREATE TABLE dim_providers (
    provider_id             VARCHAR(30) PRIMARY KEY,
    potential_fraud         BOOLEAN NOT NULL,
    fraud_label             VARCHAR(5) NOT NULL CHECK (fraud_label IN ('Yes', 'No')),
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------
-- 2. dim_beneficiaries: Medicare patients and clinical demographics
-- ---------------------------------------------------------------------
CREATE TABLE dim_beneficiaries (
    bene_id                     VARCHAR(30) PRIMARY KEY,
    dob                         DATE NOT NULL,
    dod                         DATE,
    gender                      INTEGER NOT NULL CHECK (gender IN (1, 2)),
    race                        INTEGER NOT NULL,
    state_id                    INTEGER NOT NULL,
    county_id                   INTEGER NOT NULL,
    renal_disease               BOOLEAN NOT NULL DEFAULT 0,
    chronic_alzheimer           BOOLEAN NOT NULL DEFAULT 0,
    chronic_heartfailure        BOOLEAN NOT NULL DEFAULT 0,
    chronic_kidneydisease       BOOLEAN NOT NULL DEFAULT 0,
    chronic_cancer              BOOLEAN NOT NULL DEFAULT 0,
    chronic_copd                BOOLEAN NOT NULL DEFAULT 0,
    chronic_depression          BOOLEAN NOT NULL DEFAULT 0,
    chronic_diabetes            BOOLEAN NOT NULL DEFAULT 0,
    chronic_ischemicheart       BOOLEAN NOT NULL DEFAULT 0,
    chronic_osteoporosis        BOOLEAN NOT NULL DEFAULT 0,
    chronic_rheumatoidarthritis BOOLEAN NOT NULL DEFAULT 0,
    chronic_stroke              BOOLEAN NOT NULL DEFAULT 0,
    ip_annual_reimbursement     NUMERIC(12,2) DEFAULT 0.0,
    ip_annual_deductible        NUMERIC(12,2) DEFAULT 0.0,
    op_annual_reimbursement     NUMERIC(12,2) DEFAULT 0.0,
    op_annual_deductible        NUMERIC(12,2) DEFAULT 0.0,
    created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------
-- 3. fact_inpatient_claims: Hospital admission claims
-- ---------------------------------------------------------------------
CREATE TABLE fact_inpatient_claims (
    claim_id                VARCHAR(30) PRIMARY KEY,
    bene_id                 VARCHAR(30) NOT NULL REFERENCES dim_beneficiaries(bene_id),
    provider_id             VARCHAR(30) NOT NULL REFERENCES dim_providers(provider_id),
    claim_start_date        DATE NOT NULL,
    claim_end_date          DATE NOT NULL,
    admission_date          DATE NOT NULL,
    discharge_date          DATE NOT NULL,
    length_of_stay          INTEGER NOT NULL CHECK (length_of_stay >= 0),
    reimbursed_amount       NUMERIC(12,2) NOT NULL CHECK (reimbursed_amount >= 0),
    deductible_amount       NUMERIC(12,2) NOT NULL CHECK (deductible_amount >= 0),
    attending_physician     VARCHAR(30),
    operating_physician     VARCHAR(30),
    other_physician         VARCHAR(30),
    admit_diagnosis_code    VARCHAR(20),
    diagnosis_group_code    VARCHAR(20),
    primary_diagnosis_code  VARCHAR(20),
    primary_procedure_code  VARCHAR(20),
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------
-- 4. fact_outpatient_claims: Clinic & outpatient hospital claims
-- ---------------------------------------------------------------------
CREATE TABLE fact_outpatient_claims (
    claim_id                VARCHAR(30) PRIMARY KEY,
    bene_id                 VARCHAR(30) NOT NULL REFERENCES dim_beneficiaries(bene_id),
    provider_id             VARCHAR(30) NOT NULL REFERENCES dim_providers(provider_id),
    claim_start_date        DATE NOT NULL,
    claim_end_date          DATE NOT NULL,
    reimbursed_amount       NUMERIC(12,2) NOT NULL CHECK (reimbursed_amount >= 0),
    deductible_amount       NUMERIC(12,2) NOT NULL CHECK (deductible_amount >= 0),
    attending_physician     VARCHAR(30),
    operating_physician     VARCHAR(30),
    other_physician         VARCHAR(30),
    primary_diagnosis_code  VARCHAR(20),
    primary_procedure_code  VARCHAR(20),
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------
-- 5. fact_claims_unified: Consolidated view/table for fast analytics
-- ---------------------------------------------------------------------
CREATE TABLE fact_claims_unified (
    claim_id                VARCHAR(30) PRIMARY KEY,
    claim_type              VARCHAR(15) NOT NULL CHECK (claim_type IN ('Inpatient', 'Outpatient')),
    bene_id                 VARCHAR(30) NOT NULL REFERENCES dim_beneficiaries(bene_id),
    provider_id             VARCHAR(30) NOT NULL REFERENCES dim_providers(provider_id),
    claim_start_date        DATE NOT NULL,
    claim_end_date          DATE NOT NULL,
    reimbursed_amount       NUMERIC(12,2) NOT NULL CHECK (reimbursed_amount >= 0),
    deductible_amount       NUMERIC(12,2) NOT NULL CHECK (deductible_amount >= 0),
    length_of_stay          INTEGER NOT NULL DEFAULT 0,
    attending_physician     VARCHAR(30),
    operating_physician     VARCHAR(30),
    primary_diagnosis_code  VARCHAR(20),
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================================
-- Performance Indexes
-- =====================================================================
CREATE INDEX idx_inpatient_provider      ON fact_inpatient_claims(provider_id);
CREATE INDEX idx_inpatient_bene          ON fact_inpatient_claims(bene_id);
CREATE INDEX idx_inpatient_start_dt      ON fact_inpatient_claims(claim_start_date);
CREATE INDEX idx_outpatient_provider     ON fact_outpatient_claims(provider_id);
CREATE INDEX idx_outpatient_bene         ON fact_outpatient_claims(bene_id);
CREATE INDEX idx_unified_provider        ON fact_claims_unified(provider_id);
CREATE INDEX idx_unified_bene            ON fact_claims_unified(bene_id);
CREATE INDEX idx_unified_start_dt        ON fact_claims_unified(claim_start_date);
CREATE INDEX idx_unified_provider_amount ON fact_claims_unified(provider_id, reimbursed_amount);
CREATE INDEX idx_beneficiary_state       ON dim_beneficiaries(state_id);
