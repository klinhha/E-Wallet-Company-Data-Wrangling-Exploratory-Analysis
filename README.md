# [PYTHON] E-Wallet Transaction Analysis

## I. Introduction

This project analyzes payment and transaction data for a hypothetical e-wallet company using Python and pandas. It focuses on data cleaning, data integration, transaction classification, and business analysis to identify product performance patterns, team contributions, and refund distribution.

## II. Objectives

- Assess data quality, including missing values, duplicates, and incorrect data types.
- Clean and integrate datasets using pandas.
- Analyze product payment volumes, team performance, and refund contributions.
- Classify transactions using predefined business rules.
- Generate business insights and recommendations.

## III. Tools & Technologies
- Language: Python
- Libraries: pandas, NumPy, Matplotlib, Seaborn
- Environment: Google Colab
- Data sources: Internal company database (e-wallet transaction records)

## IV. Dataset Overview
Three datasets are used in this project for analysis.
| Dataset | Records | Description |
|---|---:|---|
| `payment_report.csv` | 919 | Monthly payment volume by product and source. |
| `product.csv` | 492 | Product information, category, and owning team. |
| `transactions.csv` | 1,324,002 | Transaction details and participant information. |

### 1. `payment_report.csv`

| Column | Data Type | Description |
|---|---|---|
| `report_month` | `object` | Reporting month of the payment volume. Converted to `datetime` during data cleaning. |
| `payment_group` | `object` | Classification of payment activity, such as payments or refunds. |
| `product_id` | `int64` | Unique identifier used to associate payment records with products. |
| `source_id` | `int64` | Identifier of the payment source. |
| `volume` | `int64` | Reported payment volume for the corresponding month, product, source, and payment group. |

### 2. `product.csv`

| Column | Data Type | Description |
|---|---|---|
| `product_id` | `int64` | Product identifier used to link product information with payment records. |
| `category` | `object` | Category to which the product belongs. |
| `team_own` | `object` | Team responsible for the product. |

### 3. `transactions.csv`

| Column | Data Type | Description |
|---|---|---|
| `transaction_id` | `int64` | Identifier of the transaction record. |
| `merchant_id` | `int64` | Identifier of the merchant associated with the transaction. |
| `volume` | `int64` | Recorded transaction volume. |
| `transType` | `int64` | Transaction type code used to classify transaction activity. |
| `transStatus` | `int64` | Transaction status code. The meaning of each code requires a business definition. |
| `sender_id` | `float64` | Identifier of the sender. Missing values were present in the original dataset. |
| `receiver_id` | `float64` | Identifier of the receiver. Missing values were present in the original dataset. |
| `extra_info` | `object` | Additional transaction information; most values were missing. Removed during data preparation. |
| `timeStamp` | `int64` | Original transaction timestamp, stored as an integer and converted to `datetime` during data cleaning. |

## V. Exploratory Data Analysis (EDA)
### 1. Import library & Load Dataset
```python
# Import necessary libraries
import numpy as np
import pandas as pd
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```
```python
# Mount Google Drive to Colab
from google.colab import drive
drive.mount('/content/drive')

# Import CSV files into Colab
transactions = pd.read_csv('/content/drive/MyDrive/Python - Data_Wrangling_project/transactions.csv')
product = pd.read_csv('/content/drive/MyDrive/Python - Data_Wrangling_project/product.csv')
payment = pd.read_csv('/content/drive/MyDrive/Python - Data_Wrangling_project/payment_report.csv')
```
### 2. Display Datasets
- Product:

|         | **product_id** | **category** | **team_own** |
| ------: | -------------: | -----------: | -----------: |
|   **0** |             17 |      PXXXXXB |          ASD |
|   **1** |             18 |      PXXXXXB |          ASD |
|   **2** |             20 |      PXXXXXB |          ASD |
|   **3** |            287 |      PXXXXXB |          ASD |
|   **4** |            372 |      PXXXXXB |          ASD |
| **...** |            ... |          ... |          ... |
| **487** |            321 |      PXXXXXV |          ASD |
| **488** |            322 |      PXXXXXV |          ASD |
| **489** |            341 |      PXXXXXV |          ASD |
| **490** |            342 |      PXXXXXV |          ASD |
| **491** |            367 |      PXXXXXV |          ASD |

492 rows × 3 columns

- Payment:

|     | report_month | payment_group | product_id | source_id | volume | category | team_own |
|----:|--------------|---------------|-----------:|----------:|-------:|----------|----------|
| 0   | 2023-01      | payment       | 12         | 45        | 624110375 | PXXXXXT | ASD |
| 1   | 2023-01      | payment       | 17         | 45        | 335715113 | PXXXXXB | ASD |
| 2   | 2023-01      | payment       | 18         | 45        | 737784466 | PXXXXXB | ASD |
| 3   | 2023-01      | payment       | 19         | 45        | 120963069 | PXXXXXM2 | ASD |
| 4   | 2023-01      | payment       | 20         | 45        | 319653158 | PXXXXXB | ASD |
| ... | ...          | ...           | ...        | ...       | ...       | ...     | ... |
| 914 | 2023-04      | payment       | 15067      | 45        | 1504000 | PXXXXXR | ASL |
| 915 | 2023-04      | refund        | 1976       | 37        | 3542271587 | NaN | NaN |
| 916 | 2023-04      | refund        | 1976       | 38        | 13831708189 | NaN | NaN |
| 917 | 2023-04      | refund        | 1976       | 39        | 1905435543 | NaN | NaN |
| 918 | 2023-04      | refund        | 1976       | 39        | 3679922071 | NaN | NaN |

919 rows × 5 columns

- Transactions:

|             | **transaction_id** | **merchant_id** | **volume** | **transType** | **transStatus** | **sender_id** | **receiver_id** | **extra_info** | **timeStamp** |
| ----------: | -----------------: | --------------: | ---------: | ------------: | --------------: | ------------: | --------------: | -------------: | ------------: |
|       **0** |         3002692434 |               5 |     100000 |            24 |               1 |    10199794.0 |        199794.0 |            NaN | 1682932054455 |
|       **1** |         3002692437 |             305 |      20000 |             2 |               1 |    14022211.0 |      14022211.0 |            NaN | 1682932054912 |
|       **2** |         3001960110 |            7255 |      48605 |            22 |               1 |           NaN |      10530940.0 |            NaN | 1682932055000 |
|       **3** |         3002680710 |            2270 |    1500000 |             2 |               1 |    10059206.0 |         59206.0 |            NaN | 1682932055622 |
|       **4** |         3002680713 |            2275 |      90000 |             2 |               1 |    10004711.0 |          4711.0 |            NaN | 1682932056197 |
|     **...** |                ... |             ... |        ... |           ... |             ... |           ... |             ... |            ... |           ... |
| **1323997** |         3003723030 |             305 |      20000 |             2 |               1 |    24524311.0 |             NaN |            NaN | 1683035672634 |
| **1323998** |         3003723033 |            2270 |     100000 |             2 |               1 |    10277242.0 |        277242.0 |            NaN | 1683035672876 |
| **1323999** |         3003723036 |            2270 |     100000 |             2 |               1 |    10144599.0 |        144599.0 |            NaN | 1683035672892 |
| **1324000** |         3003723039 |               5 |        400 |            22 |               1 |    10028007.0 |      21013762.0 |            NaN | 1683035672896 |
| **1324001** |         3003602967 |            2250 |          1 |             8 |               1 |    38559843.0 |      24501638.0 |            NaN | 1683035673053 |

1324002 rows × 9 columns

### 3. Merge payment_report.csv with product.csv
```python
payment_enriched = payment.merge(
    product,
    on='product_id',
    how='left')
```
- Result:

|     | report_month | payment_group | product_id | source_id | volume | category | team_own |
|----:|--------------|---------------|-----------:|----------:|-------:|----------|----------|
| 0   | 2023-01      | payment       | 12         | 45        | 624110375 | PXXXXXT | ASD |
| 1   | 2023-01      | payment       | 17         | 45        | 335715113 | PXXXXXB | ASD |
| 2   | 2023-01      | payment       | 18         | 45        | 737784466 | PXXXXXB | ASD |
| 3   | 2023-01      | payment       | 19         | 45        | 120963069 | PXXXXXM2 | ASD |
| 4   | 2023-01      | payment       | 20         | 45        | 319653158 | PXXXXXB | ASD |
| ... | ...          | ...           | ...        | ...       | ...       | ...     | ... |
| 914 | 2023-04      | payment       | 15067      | 45        | 1504000 | PXXXXXR | ASL |
| 915 | 2023-04      | refund        | 1976       | 37        | 3542271587 | NaN | NaN |
| 916 | 2023-04      | refund        | 1976       | 38        | 13831708189 | NaN | NaN |
| 917 | 2023-04      | refund        | 1976       | 39        | 1905435543 | NaN | NaN |
| 918 | 2023-04      | refund        | 1976       | 39        | 3679922071 | NaN | NaN |

### 4. Check and convert Data Type
#### a. Product
- No incorrect Data type

#### b. Payment and Payment_enriched
- Incorrect Data type in column "report_month"

  => Convert to the correct data type
```python
# Convert the report_month column from object to datetime
payment['report_month'] = pd.to_datetime(payment['report_month'])
payment_enriched['report_month'] = pd.to_datetime(payment_enriched['report_month'])
```
#### c. Transactions
- Incorrect Data type in column "timeStamp" 

=> Convert to the correct data type
```python
# Convert 'timeStamp' from int64 to datetime
transactions['timeStamp'] = pd.to_datetime(transactions['timeStamp'])
```
### 5. Check for missing values
- Missing values in transaction table:
  - **49059 rows** in column 'sender_id'
  - **164795 rows** in column 'receiver_id'
  - **1317907 rows** in column 'extra_info'

=> **Replaced** missing sender and receiver IDs with Unknow and removed extra_info from the working dataset:
```python
# The sender_id and receiver_id columns in the Transaction table are not important, so missing values are replaced with 'Unknow' for consistency
transactions['sender_id'] = transactions['sender_id'].fillna(value='Unknow')
transactions['receiver_id'] = transactions['receiver_id'].fillna(value='Unknow')

# Drop the 'extra_info' column from the Transaction table
transactions.drop('extra_info', axis=1, inplace=True)
```

- **No missing values** in payment and product tables

### 6. Check for Duplicates
```python
# Check for duplicates
duplicate_payment = payment.duplicated().sum()
duplicate_product = product.duplicated().sum()
duplicate_transactions = transactions.duplicated().sum()
```
- **No duplicates** are found in payment and product table
- Identified **28 duplicate rows** in  transaction table

=> **Removed** duplicates to maintain data integrity:
``` python
#Drop duplicate
transactions.drop_duplicates(subset=None, keep='first')
```

## VI. Data Wrangling
### Using payment_report.csv & product.csv
### Q1. What are the top three products by total payment volume?
```python
payment_enriched.groupby("product_id",
                         as_index = False).agg({'volume':'sum'}).sort_values('volume',
                                                                              ascending = False).head(3)
```
- Result

|     | **product_id** | **volume** |
| --: | -------------: | ---------: |
| **279** | 1976 | 61797583647 |
| **43**  | 429  | 14667676567 |
| **38**  | 372  | 13713658515 |

=> Product 1976 dominates the total volume among the three products. This suggests it **plays a key role** in the E-wallet company's business and **should be prioritized** for promotion and strategic focus.

### Q2. Given that 1 product_id is only owed by 1 team, are there any abnormal products against this rule?
```python
# Group by product_id and count the number of unique team_own values
abnormal = payment_enriched.groupby("product_id",
                         as_index = False).agg({'team_own': 'nunique'}) #nunique is similar to DISTINCT COUNT

# Identify product_id entries managed by more than one team
abnormal_products = abnormal[abnormal["team_own"]>1]

# Print the results
if abnormal_products.empty:
  print('No abnormal products against this rule')
else:
  print('There are abnormal products against this rule:')
  print(abnormal_products)
```
- Result

**No abnormal products** against this rule

=> No abnormal products were identified. Each product is associated with at most one team, satisfying the product ownership rule for all products with assigned team information.

### Q3. Find the team has had the lowest performance (lowest volume) since Q2.2023. Find the category that contributes the least to that team.

a. Find the team has had the lowest performance (lowest volume) since Q2.2023.
```python
# Filter data from Q2.2023 onwards
Q2_2023 = payment_enriched[payment_enriched['report_month'] >= '2023-04-01']

# Find the team with the lowest total volume
lowest_volume = (
    Q2_2023.groupby('team_own', as_index=False)
    .agg({'volume': 'sum'})
    .sort_values('volume')
    .head(1)
)
```
- Result


|       | **team_own** | **volume** |
| ----: | -----------: | ---------: |
| **0** |          APS |   51141753 |

The lowest performance team in Q2.2023 is **APS** with **volume = 51141753**

b. Find the category that contributes the least to that team.
```python
# Filter data for the lowest-performing team since Q2.2023
team_APS = Q2_2023[Q2_2023['team_own'] == 'APS']

# Find the category with the lowest total volume
least_contribution = (team_APS.groupby('category', as_index=False)
    .agg({'volume': 'sum'})
    .sort_values('volume')
    .head(1)
)
```
- Result

|       | **category** | **volume** |
| ----: | -----------: | ---------: |
| **0** |      PXXXXXE |   25232438 |

The lowest contributing category to the team APS is **PXXXXXE** with **volume = 25232438**


=> APS’s performance **requires improvement**, particularly in category **PXXXXXE**, which has the lowest contribution to the team’s total volume. Further investigation is needed to **identify the root causes** and **determine appropriate actions** to improve performance.

### Q4. Find the contribution of source_ids of refund transactions (payment_group = ‘refund’), what is the source_id with the highest contribution?
```python
# Filter transactions that are refunds
refund = payment_enriched[payment_enriched['payment_group'] == 'refund']

# Calculate total volume by each source_id
refund_volume = refund.groupby("source_id",
                         as_index = False).agg({'volume': 'sum'}).sort_values('volume',
                                                                              ascending = False)

# Calculate the percentage contribution of each source_id
refund_volume['contribution'] = (refund_volume["volume"]/ refund_volume["volume"].sum() * 100)

print(refund_volume)

# Get the source_id with the highest contribution
top_source = refund_volume.loc[
    refund_volume['volume'] == refund_volume['volume'].max()].iloc[0]

source_id = top_source['source_id']
contribute = top_source['contribution']

print(
    f"{source_id} is the highest contributor to refund transactions "
    f"with {contribute:.2f}% contribution."
)
```

- Result
  
| **source_id** | **volume** | **contribution** |
| ------------: | ---------: | ---------------: |
|            38 | 36527454759 | 59.108225 |
|            39 | 16119059662 | 26.083641 |
|            37 |  9151069226 | 14.808134 |


=> Source_id **38** accounts for the **largest share of refund volume**, highlighting a potential area of concern. It is neccesary to investigate into possible causes, such as product quality issues or customer dissatisfaction, could help reduce refunds and improve the overall customer experience.

### Using transactions.csv
### Q5. Define type of transactions (‘transaction_type’) for each row, given:
- transType = 2 & merchant_id = 1205: Bank Transfer Transaction
- transType = 2 & merchant_id = 2260: Withdraw Money Transaction
- transType = 2 & merchant_id = 2270: Top Up Money Transaction
- transType = 2 & others merchant_id: Payment Transaction
- transType = 8, merchant_id = 2250: Transfer Money Transaction
- transType = 8 & others merchant_id: Split Bill Transaction
- Remained cases are invalid transactions
```python
# Define conditions for transaction types
def clasify_transaction(row):
  if row["transType"] == 2 and row["merchant_id"] == 1205:
    return "Bank Transfer Transaction"

  elif row["transType"] == 2 and row["merchant_id"] == 2260:
    return "Withdraw Money Transaction"

  elif row["transType"] == 2 and row["merchant_id"] == 2270:
    return "Top Up Money Transaction"

  elif row["transType"] == 2:
    return "Payment Transaction"

  elif row["transType"] == 8 and row["merchant_id"] == 2250:
    return "Transfer Money Transaction"

  elif row["transType"] == 8:
    return "Split Bill Transaction"

  else:
    return "Invalid Transactions"

# Assign transaction types based on conditions
transactions["transaction_type"] = transactions.apply(clasify_transaction, axis = 1)
```
- Result

|  | **transaction_id** | **merchant_id** | **volume** | **transType** | **transStatus** | **sender_id** | **receiver_id** | **timeStamp** | **transaction_type** |
| --: | --: | --: | --: | --: | --: | --: | --: | -- | -- |
| **0** | 3002692434 | 5 | 100000 | 24 | 1 | 10199794.0 | 199794.0 | 1970-01-01 00:28:02.932054455 | Invalid Transactions |
| **1** | 3002692437 | 305 | 20000 | 2 | 1 | 14022211.0 | 14022211.0 | 1970-01-01 00:28:02.932054912 | Payment Transaction |
| **2** | 3001960110 | 7255 | 48605 | 22 | 1 | Unknow | 10530940.0 | 1970-01-01 00:28:02.932055000 | Invalid Transactions |
| **3** | 3002680710 | 2270 | 1500000 | 2 | 1 | 10059206.0 | 59206.0 | 1970-01-01 00:28:02.932055622 | Top Up Money Transaction |
| **4** | 3002680713 | 2275 | 90000 | 2 | 1 | 10004711.0 | 4711.0 | 1970-01-01 00:28:02.932056197 | Payment Transaction |
| **...** | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| **1323997** | 3003723030 | 305 | 20000 | 2 | 1 | 24524311.0 | Unknow | 1970-01-01 00:28:03.035672634 | Payment Transaction |
| **1323998** | 3003723033 | 2270 | 100000 | 2 | 1 | 10277242.0 | 277242.0 | 1970-01-01 00:28:03.035672876 | Top Up Money Transaction |
| **1323999** | 3003723036 | 2270 | 100000 | 2 | 1 | 10144599.0 | 144599.0 | 1970-01-01 00:28:03.035672892 | Top Up Money Transaction |
| **1324000** | 3003723039 | 5 | 400 | 22 | 1 | 10028007.0 | 21013762.0 | 1970-01-01 00:28:03.035672896 | Invalid Transactions |
| **1324001** | 3003602967 | 2250 | 1 | 8 | 1 | 38559843.0 | 24501638.0 | 1970-01-01 00:28:03.035673053 | Transfer Money Transaction |

=> This classification **helps** analyze transaction patterns and **identify** potentially invalid records for further investigation.

### Q6. Of each transaction type (excluding invalid transactions): find the number of transactions, volume, senders and receivers.
```python
#excluding invalid transactions
transactions_valid = transactions[transactions["transaction_type"] != "Invalid Transactions"]

# Group by transaction type and calculate the required metrics
transaction_number = transactions_valid.groupby("transaction_type",
                                                as_index = False).agg({"transaction_id": "count",
                                                                       "volume" : "sum",
                                                                       "sender_id" : "nunique",
                                                                       "receiver_id" : "nunique"})
```

- Result

| **transaction_type** | **transaction_id** | **volume** | **sender_id** | **receiver_id** |
| :--- | ---: | ---: | ---: | ---: |
| Bank Transfer Transaction | 37,879 | 50,605,806,190 | 23,156 | 9,272 |
| Payment Transaction | 398,677 | 71,851,515,181 | 139,583 | 113,299 |
| Split Bill Transaction | 1,376 | 4,901,464 | 1,323 | 572 |
| Top Up Money Transaction | 290,502 | 108,606,478,829 | 110,409 | 110,409 |
| Transfer Money Transaction | 341,177 | 37,033,171,492 | 39,021 | 34,585 |
| Withdraw Money Transaction | 33,725 | 23,418,181,420 | 24,814 | 24,814 |

=> **Top-up transactions** recorded the **highest total volume** (108.61 billion), while **payment transactions** had the **highest number of transactions** (398,677) and the **largest number of unique senders** (139,583). **Split-bill transactions** had the **lowest** volume and transaction count. 
These findings suggest that **top-ups** and **payments** are the **main transaction activities** by volume and frequency, respectively, while split-bill transactions represent a relatively small share of activity.

## VII. Conclusions & Recommendations
### 1. Conclusions
- **Transaction volume**: Top-up transactions generated the highest total volume at 108.61 billion, highlighting their importance in overall transaction value.
- **Transaction activity**: Payments recorded the highest transaction count (398,677) and the largest number of unique senders (139,583), indicating strong user engagement.
- **Split-bill transactions**: Split-bill activity had the lowest total volume and transaction count, suggesting relatively limited usage compared with other transaction types.

### 2. Recommendations
- **Strengthen top-up services**: Investigate user behavior and popular top-up products to identify more opportunities to increase transaction volume and retention.
- **Optimize payment services**: Improve payment reliability and convenience to maintain user engagement and encourage repeat transactions.
- **Increase split-bill adoption**: Explore targeted promotions and simplify the user experience to encourage more users to try split-bill transactions.
- **Monitor transaction performance**: Track transaction volume, transaction count, unique users, and growth trends by transaction type to support data-driven business decisions.







