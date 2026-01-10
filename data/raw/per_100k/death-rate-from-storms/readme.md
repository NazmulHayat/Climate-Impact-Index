# Death rate from storms - Data package

This data package contains the data that powers the chart ["Death rate from storms"](https://ourworldindata.org/explorers/natural-disasters?Disaster+Type=Storms&Impact=Deaths&Timespan=Annual&Per+capita=true&country=~OWID_WRL) on the Our World in Data website.

## CSV Structure

The high level structure of the CSV file is that each row is an observation for an entity (usually a country or region) and a timepoint (usually a year).

The first two columns in the CSV file are "Entity" and "Code". "Entity" is the name of the entity (e.g. "United States"). "Code" is the OWID internal entity code that we use if the entity is a country or region. For normal countries, this is the same as the [iso alpha-3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) code of the entity (e.g. "USA") - for non-standard countries like historical countries these are custom codes.

The third column is either "Year" or "Day". If the data is annual, this is "Year" and contains only the year as an integer. If the column is "Day", the column contains a date string in the form "YYYY-MM-DD".

The remaining columns are the data columns, each of which is a time series. If the CSV data is downloaded using the "full data" option, then each column corresponds to one time series below. If the CSV data is downloaded using the "only selected data visible in the chart" option then the data columns are transformed depending on the chart type and thus the association with the time series might not be as straightforward.

## Metadata.json structure

The .metadata.json file contains metadata about the data package. The "charts" key contains information to recreate the chart, like the title, subtitle etc.. The "columns" key contains information about each of the columns in the csv, like the unit, timespan covered, citation for the data etc..

## About the data

Our World in Data is almost never the original producer of the data - almost all of the data we use has been compiled by others. If you want to re-use data, it is your responsibility to ensure that you adhere to the sources' license and to credit them correctly. Please note that a single time series may have more than one source - e.g. when we stich together data from different time periods by different producers or when we calculate per capita metrics using population data from a second source.

### How we process data at Our World In Data
All data and visualizations on Our World in Data rely on data sourced from one or several original data providers. Preparing this original data involves several processing steps. Depending on the data, this can include standardizing country names and world region definitions, converting units, calculating derived indicators such as per capita measures, as well as adding or adapting metadata such as the name or the description given to an indicator.
[Read about our data pipeline](https://docs.owid.io/projects/etl/)

## Detailed information about each time series


## Death rates from storms
Unit: per 100,000  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
EM-DAT, CRED / UCLouvain – processed by Our World in Data

#### Full citation
EM-DAT, CRED / UCLouvain – processed by Our World in Data. “Death rates from storms” [dataset]. EM-DAT, CRED / UCLouvain [original data].
Source: EM-DAT, CRED / UCLouvain – processed by Our World In Data

### Additional information about this data
This dataset has been calculated and compiled by Our World in Data based on raw disaster data published by EM-DAT, CRED / UCLouvain, Brussels, Belgium – www.emdat.be (D. Guha-Sapir).
 EM-DAT publishes comprehensive, global data on each individual disaster event – estimating the number of deaths; people affected; and economic damages, from UN reports; government records; expert opinion; and additional sources.
 Our World in Data have calculated annual aggregates, and decadal averages, for each country based on this raw event-by-event dataset.
 Decadal figures are measured as the annual average over the subsequent ten-year period. This means figures for ‘1900’ represent the average from 1900 to 1909; ‘1910’ is the average from 1910 to 1919 etc.
 We have calculated per capita rates using population figures from Gapminder (gapminder.org) and the UN World Population Prospects (https://population.un.org/wpp/).
 Economic damages data is provided by EM-DAT in current US$. We have calculated this as a share of gross domestic product (GDP) using the World Bank’s GDP figures (also in current US$) (https://data.worldbank.org/indicator).
 Definitions of specific metrics are as follows:
 – ‘All disasters’ includes all geophysical, meteorological and climate events including earthquakes, volcanic activity, landslides, drought, wildfires, storms, and flooding.
 – The total number of people affected is the sum of injured, requiring assistance and homeless.


    