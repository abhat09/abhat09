-- Q2: create external hive table

CREATE DATABASE chicago_crimes;



-- Q3: load data into table


CREATE EXTERNAL TABLE IF NOT EXISTS chicago_crimes (
    ID STRING,
    case_num STRING,
    crime_date STRING,
    block STRING,
    IUCR STRING,
    primary_type STRING,
    description STRING,
    location_desc STRING,
    arrest STRING,
    domestic STRING,
    beat DATE,
    district STRING,
    ward STRING,
    community STRING,
    fbi_code STRING,
    x_coord STRING,
    y_coord STRING,
    year_occured STRING,
    updated STRING,
    lat STRING,
    long STRING,
    location_coord STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
    "separatorChar" = ",",
    "quoteChar"     = '\"',
    "escapeChar"    = '\\'
)
STORED AS TEXTFILE
LOCATION 's3://anusha-assign2/crimedata'
TBLPROPERTIES ("skip.header.line.count"="1");

-- view table:

SELECT * FROM "chicago_crimes"."chicago_crimes" limit 10;

-- Q4: 

-- earliest and most recent crime dates


SELECT 
    MIN(DATE_PARSE(crime_date, '%m/%d/%Y %I:%i:%s %p')),
    MAX(DATE_PARSE(crime_date, '%m/%d/%Y %I:%i:%s %p'))
FROM 
    chicago_crimes;



-- crimes on early date

SELECT DISTINCT primary_type
FROM chicago_crimes
WHERE 
    DATE_PARSE(crime_date, '%m/%d/%Y %I:%i:%s %p') IN (
        SELECT MIN(DATE_PARSE(crime_date, '%m/%d/%Y %I:%i:%s %p')) 
        FROM chicago_crimes
    );



-- crimes on latest date
SELECT DISTINCT primary_type
FROM chicago_crimes
WHERE 
    DATE_PARSE(crime_date, '%m/%d/%Y %I:%i:%s %p') IN (
        SELECT MAX(DATE_PARSE(crime_date, '%m/%d/%Y %I:%i:%s %p')) 
        FROM chicago_crimes
    );




-- Q5: top 5 and bottom 5 crimes

-- top 5:
SELECT primary_type, COUNT(*) AS occurrence_count
FROM chicago_crimes
GROUP BY primary_type
ORDER BY occurrence_count DESC
LIMIT 5;


-- bottom 5:
SELECT primary_type, COUNT(*) AS occurrence_count
FROM chicago_crimes
GROUP BY primary_type
ORDER BY occurrence_count ASC
LIMIT 5;


-- Q6: location w/ most homicides

SELECT location_desc, COUNT(*) AS homicide_count
FROM chicago_crimes
WHERE primary_type = 'HOMICIDE'
GROUP BY location_desc
ORDER BY homicide_count DESC
LIMIT 1;


-- Q7: most dangerous and least dangerous

-- top 5 most dangerous districts

SELECT district, COUNT(*) AS count
FROM chicago_crimes
GROUP BY district
ORDER BY count DESC
LIMIT 5;


-- bottom 45 (least dangerous, all with count of 1 crime)

SELECT district, COUNT(*) AS count
FROM chicago_crimes
GROUP BY district
ORDER BY count ASC;


-- Q8: avg. assualts/month 

-- avg for 2021

SELECT 
    COUNT(*) / 12 AS average
FROM 
    chicago_crimes
WHERE 
    primary_type = 'ASSAULT'
    AND YEAR(DATE_PARSE(crime_date, '%m/%d/%Y %I:%i:%s %p')) = 2021;


-- avg for 2020

SELECT 
    COUNT(*) / 12 AS average
FROM 
    chicago_crimes
WHERE 
    primary_type = 'ASSAULT'
    AND YEAR(DATE_PARSE(crime_date, '%m/%d/%Y %I:%i:%s %p')) = 2020;


-- Q9: parquet table
CREATE TABLE chicago_crimes_parquet
WITH (
    format = 'PARQUET',
    external_location = 's3://anusha-assign2/chicago_crimes_parquet/'
) AS
SELECT *
FROM chicago_crimes;


-- view table:

SELECT * FROM "chicago_crimes"."chicago_crimes_parquet" limit 10;



-- Q10: EXPLAIN for q6 query 

-- for original table

EXPLAIN
SELECT location_desc, COUNT(*) AS homicide_count
FROM chicago_crimes
WHERE primary_type = 'HOMICIDE'
GROUP BY location_desc
ORDER BY homicide_count DESC
LIMIT 1;


-- for parquet table

EXPLAIN
SELECT location_desc, COUNT(*) AS homicide_count
FROM chicago_crimes_parquet
WHERE primary_type = 'HOMICIDE'
GROUP BY location_desc
ORDER BY homicide_count DESC
LIMIT 1;


-- Q11: summarized table 

CREATE TABLE chicago_crimes_summarized AS
SELECT community, primary_type, COUNT(*) AS offense_count
FROM chicago_crimes
GROUP BY community, primary_type;

-- view table:
SELECT * FROM "chicago_crimes"."chicago_crimes_summarized" limit 10;

-- get community names from wikipedia page and turn into csv
-- load csv into s3
-- create community table in Athena (no header in csv so don't skip 1st line)

CREATE EXTERNAL TABLE IF NOT EXISTS chicago_communities (
    community STRING,
    area_name STRING
)
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n' 
STORED AS TEXTFILE
LOCATION 's3://anusha-assign2/community_data';

-- preview table

SELECT * FROM "chicago_crimes"."chicago_communities" limit 10;

-- create joined table 

CREATE TABLE chicago_crimes_summarized_final AS
SELECT 
    chicago_crimes_summarized.*,
    chicago_communities.area_name
FROM 
    chicago_crimes_summarized
JOIN 
    chicago_communities
ON 
    chicago_crimes_summarized.community = chicago_communities.community;


-- preview table and download results

SELECT * FROM "chicago_crimes"."chicago_crimes_summarized_final";