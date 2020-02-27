
--query diagnosis codes for sepsis
CREATE TABLE sepsis_icd9_codes AS
    SELECT
        DISTINCT ICD9_CODE,
        LONG_TITLE
    FROM d_icd_diagnoses
    WHERE lower(LONG_TITLE) LIKE 'seps%' OR lower(LONG_TITLE) LIKE 'septi%';


--create new table that includes patients for whom the initial free-form text diagnosis given was of sepsis.
CREATE TABLE sepsis_on_arrival AS
    SELECT admissions.SUBJECT_ID,
           admissions.HADM_ID,
           ADMITTIME,
           DISCHTIME,
           DIAGNOSIS,
           HOSPITAL_EXPIRE_FLAG,
           GENDER,
           DOB,
           DOD_HOSP
    FROM admissions
    INNER JOIN patients ON admissions.SUBJECT_ID = patients.SUBJECT_ID
    WHERE lower(DIAGNOSIS) LIKE '%seps%' OR lower(DIAGNOSIS) LIKE 'sept%';


SELECT 
    count(DISTINCT SUBJECT_ID)
FROM 
    sepsis_on_arrival;
--1,709

SELECT 
    count(DISTINCT HADM_ID)
FROM 
    sepsis_on_arrival;
--1,864

--create new table that includes patients for whom the initial free-form text diagnosis given was not of sepsis
CREATE TABLE no_sepsis_on_arrival AS
    SELECT admissions.SUBJECT_ID,
           admissions.HADM_ID,
           ADMITTIME,
           DISCHTIME,
           DIAGNOSIS,
           HOSPITAL_EXPIRE_FLAG,
           GENDER,
           DOB,
           DOD_HOSP
    FROM admissions
    INNER JOIN patients ON admissions.SUBJECT_ID = patients.SUBJECT_ID
    WHERE admissions.SUBJECT_ID NOT IN (
        SELECT SUBJECT_ID
        FROM sepsis_on_arrival);

SELECT 
    count(DISTINCT SUBJECT_ID)
FROM 
    no_sepsis_on_arrival;
--44,811

SELECT 
    count(DISTINCT HADM_ID)
FROM 
    no_sepsis_on_arrival;
--55,368

--PATIENTS WITH NO SEPSIS ON ARRIVAL AND WITH SEPSIS DURING HOSPITAL STAY

--create table of patients who did not have sepsis on arrival, but acquired it during their hospital stay.
CREATE TABLE hai_sepsis_patients AS
    SELECT DISTINCT SUBJECT_ID
    FROM diagnoses_icd
    WHERE ICD9_CODE IN (
        SELECT ICD9_CODE
        FROM sepsis_icd9_codes)
    AND SUBJECT_ID IN (
            SELECT SUBJECT_ID
            FROM no_sepsis_on_arrival);

SELECT 
    count(*)
FROM 
    hai_sepsis_patients;
--2,898 patients who acquired sepsis while in the hospital.


--create table with chart data for hai_sepsis_patients.
CREATE TABLE hai_sepsis_charts AS
    SELECT hai_sepsis_patients.SUBJECT_ID,
           admissions.HADM_ID,
           ADMITTIME,
           DISCHTIME,
           chartevents.ITEMID,
           CHARTTIME,
           VALUENUM,
           VALUEUOM,
           ERROR,
           LABEL,
           CATEGORY,
           GENDER,
           DOB,
           DOD_HOSP,
           DEATHTIME,
           HOSPITAL_EXPIRE_FLAG
    FROM admissions
    INNER JOIN hai_sepsis_patients ON hai_sepsis_patients.SUBJECT_ID = admissions.SUBJECT_ID
    INNER JOIN chartevents on chartevents.SUBJECT_ID = hai_sepsis_patients.SUBJECT_ID
    INNER JOIN d_items ON chartevents.ITEMID = d_items.ITEMID
    INNER JOIN patients ON patients.SUBJECT_ID = hai_sepsis_patients.SUBJECT_ID;

--create new column CAT_GROUP
ALTER TABLE hai_sepsis_charts
ADD COLUMN CAT_GROUP integer;

--create indices to make querying faster.
CREATE INDEX
    idx_haisepsis_subjectid
ON
   hai_sepsis_charts(SUBJECT_ID);

CREATE INDEX
    idx_haisepsis_label
ON
    hai_sepsis_charts(LABEL);--only took 7 m 43 s 663 ms

--and then update the cat_group column by assigning values based on the column LABEL because there were too many labels
--to be useful in further analysis.
UPDATE hai_sepsis_charts
SET CAT_GROUP = 1 WHERE LABEL IN ('Heart Rate');

UPDATE hai_sepsis_charts
SET CAT_GROUP = 2 WHERE LABEL IN ('Non Invasive Blood Pressure diastolic', 'Manual Blood Pressure Diastolic Left', 'Manual Blood Pressure Diastolic Right');

UPDATE hai_sepsis_charts
SET CAT_GROUP = 3 WHERE LABEL IN ('Arterial Blood Pressure diastolic', 'ART BP Diastolic');

UPDATE hai_sepsis_charts
SET CAT_GROUP = 4 WHERE LABEL IN ('Non Invasive Blood Pressure systolic', 'Manual Blood Pressure Systolic Left', 'Manual Blood Pressure Systolic Right');
--took 1 m 46 s 278 ms, so maybe indexing on LABEL was a good idea.

UPDATE hai_sepsis_charts
SET CAT_GROUP = 5 WHERE LABEL IN ('Arterial Blood Pressure systolic', 'ART BP Systolic');
--took 1 m 32 s 632 ms

UPDATE hai_sepsis_charts
SET CAT_GROUP = 6 WHERE LABEL IN ('Respiratory Rate');
--9+ minutes

UPDATE hai_sepsis_charts
SET CAT_GROUP = 7 WHERE LABEL IN ('Temperature Fahrenheit');
--51 s 593 ms

SELECT count(*)
FROM hai_sepsis_charts;
--120,226,989 rows

--remove rows of the table that don't include a label in one of the 7 categories above.
CREATE TABLE hai_sepsis_charts2 AS
    SELECT *
    FROM hai_sepsis_charts
    WHERE CAT_GROUP IS NOT NULL;
--took 5 m 54 s 671 ms
--8,109,506 rows

CREATE INDEX idx_haisepsis2_subjectid
ON hai_sepsis_charts1b (SUBJECT_ID);
--28 s 604 ms

SELECT count(*)
FROM hai_sepsis_charts2;

SELECT count(DISTINCT SUBJECT_ID)
FROM hai_sepsis_charts2;

--add column with simple category name
ALTER TABLE hai_sepsis_charts2
    ADD COLUMN CAT_NAME text;

UPDATE hai_sepsis_charts2
    SET CAT_NAME = 'Heart Rate'
    WHERE CAT_GROUP = 1

UPDATE hai_sepsis_charts2
    SET CAT_NAME = 'Diastolic Blood Pressure'
    WHERE CAT_GROUP = 2;

UPDATE hai_sepsis_charts2
    SET CAT_NAME = 'Arterial Diastolic Blood Pressure'
    WHERE CAT_GROUP = 3;

UPDATE hai_sepsis_charts2
    SET CAT_NAME = 'Systolic Blood Pressure'
    WHERE CAT_GROUP = 4;

UPDATE hai_sepsis_charts2
    SET CAT_NAME = 'Arterial Systolic Blood Pressure'
    WHERE CAT_GROUP = 5;

UPDATE hai_sepsis_charts2
    SET CAT_NAME = 'Respiratory Rate'
    WHERE CAT_GROUP = 6;

UPDATE hai_sepsis_charts2
    SET CAT_NAME = 'Temperature Farenheit'
    WHERE CAT_GROUP = 7;

ALTER TABLE hai_sepsis_charts2
    ADD COLUMN CLASS integer;

UPDATE hai_sepsis_charts2
    SET CLASS = 1
    WHERE CLASS IS NULL;

CREATE TABLE hai_sepsis_charts3
AS
    SELECT SUBJECT_ID,
           HADM_ID,
           CHARTTIME,
           VALUENUM,
           VALUEUOM,
           CAT_GROUP,
           CAT_NAME,
           DOB,
           ADMITTIME,
           DISCHTIME,
           GENDER,
           HOSPITAL_EXPIRE_FLAG,
           CLASS
    FROM
         hai_sepsis_charts2;

SELECT count(DISTINCT SUBJECT_ID)
    FROM hai_sepsis_charts3;

--PATIENTS WITH NO SEPSIS ON ARRIVAL AND NO SEPSIS THROUGHOUT HOSPITAL STAY

--query patients who DID NOT have sepsis on arrival and DID NOT acquire sepsis during their hospital stay.
CREATE TABLE no_hai_sepsis_patients AS
    SELECT DISTINCT SUBJECT_ID
    FROM diagnoses_icd
    WHERE ICD9_CODE NOT IN (
        SELECT ICD9_CODE
        FROM sepsis_icd9_codes)
      AND SUBJECT_ID IN (
          SELECT SUBJECT_ID
          FROM no_sepsis_on_arrival);

SELECT count(*)
FROM no_hai_sepsis_patients;
--44,808

--create table like above but for no_hai_sepsis_patients.
CREATE TABLE no_hai_sepsis_charts AS
    SELECT no_hai_sepsis_patients.SUBJECT_ID,
           admissions.HADM_ID,
           ADMITTIME,
           DISCHTIME,
           chartevents.ITEMID,
           CHARTTIME,
           VALUENUM,
           VALUEUOM,
           ERROR,
           LABEL,
           CATEGORY,
           GENDER,
           DOB,
           DOD_HOSP,
           DEATHTIME,
           HOSPITAL_EXPIRE_FLAG
    FROM admissions
    INNER JOIN no_hai_sepsis_patients ON no_hai_sepsis_patients.SUBJECT_ID = admissions.SUBJECT_ID
    INNER JOIN chartevents on chartevents.SUBJECT_ID = no_hai_sepsis_patients.SUBJECT_ID
    INNER JOIN d_items ON chartevents.ITEMID = d_items.ITEMID
    INNER JOIN patients ON patients.SUBJECT_ID = no_hai_sepsis_patients.SUBJECT_ID;

--create new column CAT_GROUP
ALTER TABLE no_hai_sepsis_charts
ADD COLUMN CAT_GROUP integer;

CREATE INDEX
    idx_nohaisepsis_subjectid
ON
   no_hai_sepsis_charts(SUBJECT_ID);

CREATE INDEX
    idx_nohaisepsis_label
ON
    no_hai_sepsis_charts(LABEL);--was this a really bad idea? took 40 m 39 s 853 ms

--and then update the column by assigning values based on the column CATEGORY because there were too many categories to
-- be useful in further analysis.

UPDATE no_hai_sepsis_charts
SET CAT_GROUP = 1 WHERE LABEL IN ('Heart Rate');

UPDATE no_hai_sepsis_charts
SET CAT_GROUP = 2 WHERE LABEL IN ('Non Invasive Blood Pressure diastolic', 'Manual Blood Pressure Diastolic Left', 'Manual Blood Pressure Diastolic Right');

UPDATE no_hai_sepsis_charts
SET CAT_GROUP = 3 WHERE LABEL IN ('Arterial Blood Pressure diastolic', 'ART BP Diastolic');

UPDATE no_hai_sepsis_charts
SET CAT_GROUP = 4 WHERE LABEL IN ('Non Invasive Blood Pressure systolic', 'Manual Blood Pressure Systolic Left', 'Manual Blood Pressure Systolic Right');
--took 1 m 46 s 278 ms, so maybe indexing on LABEL was a good idea.

UPDATE no_hai_sepsis_charts
SET CAT_GROUP = 5 WHERE LABEL IN ('Arterial Blood Pressure systolic', 'ART BP Systolic');
--took 1 m 32 s 632 ms

UPDATE no_hai_sepsis_charts
SET CAT_GROUP = 6 WHERE LABEL IN ('Respiratory Rate');
--9+ minutes

UPDATE no_hai_sepsis_charts
SET CAT_GROUP = 7 WHERE LABEL IN ('Temperature Fahrenheit');
--51 s 593 ms


SELECT count(*)
FROM no_hai_sepsis_charts;
--505,812,695 rows

CREATE TABLE no_hai_sepsis_charts2 AS
    SELECT *
    FROM no_hai_sepsis_charts
    WHERE CAT_GROUP IS NOT NULL;
--took 33 m 19 s 579 ms
--31,691,518 rows

CREATE INDEX idx_nohaisepsis2_subjectid
ON no_hai_sepsis_charts2(SUBJECT_ID);
--21 s 388 ms

SELECT count(*)
FROM no_hai_sepsis_charts2;

SELECT count(DISTINCT SUBJECT_ID)
FROM no_hai_sepsis_charts2;
--44,282

--create table with 1000 subject_ids to reduce the processing time.
CREATE TABLE no_hai_sepsis_charts3
    AS
    SELECT
           *
    FROM
         no_hai_sepsis_charts2
    WHERE
          SUBJECT_ID in
          (SELECT *
          FROM not_septic_1000_people);

SELECT count(DISTINCT SUBJECT_ID)
    FROM no_hai_sepsis_charts3;
--982

--add column with simple category name
ALTER TABLE no_hai_sepsis_charts3
    ADD COLUMN CAT_NAME text;

UPDATE no_hai_sepsis_charts3
    SET CAT_NAME = 'Heart Rate'
    WHERE CAT_GROUP = 1

UPDATE no_hai_sepsis_charts3
    SET CAT_NAME = 'Diastolic Blood Pressure'
    WHERE CAT_GROUP = 2;

UPDATE no_hai_sepsis_charts3
    SET CAT_NAME = 'Arterial Diastolic Blood Pressure'
    WHERE CAT_GROUP = 3;

UPDATE no_hai_sepsis_charts3
    SET CAT_NAME = 'Systolic Blood Pressure'
    WHERE CAT_GROUP = 4;

UPDATE no_hai_sepsis_charts3
    SET CAT_NAME = 'Arterial Systolic Blood Pressure'
    WHERE CAT_GROUP = 5;

UPDATE no_hai_sepsis_charts3
    SET CAT_NAME = 'Respiratory Rate'
    WHERE CAT_GROUP = 6;

UPDATE no_hai_sepsis_charts3
    SET CAT_NAME = 'Temperature Farenheit'
    WHERE CAT_GROUP = 7;


ALTER TABLE no_hai_sepsis_charts3
    ADD COLUMN CLASS integer;

UPDATE no_hai_sepsis_charts3
    SET CLASS = 0
    WHERE CLASS IS NULL;

SELECT count(DISTINCT SUBJECT_ID)
FROM no_hai_sepsis_charts2;

SELECT count(DISTINCT SUBJECT_ID)
FROM no_hai_sepsis_charts3;

CREATE TABLE no_hai_sepsis_charts4
AS
    SELECT SUBJECT_ID,
           HADM_ID,
           CHARTTIME,
           VALUENUM,
           VALUEUOM,
           CAT_GROUP,
           CAT_NAME,
           DOB,
           ADMITTIME,
           DISCHTIME,
           GENDER,
           HOSPITAL_EXPIRE_FLAG,
           CLASS
    FROM
         no_hai_sepsis_charts3;

SELECT count(DISTINCT SUBJECT_ID)
FROM no_hai_sepsis_charts4;
