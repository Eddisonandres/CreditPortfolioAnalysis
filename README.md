# 💳 Credit Portfolio Simulation & Analysis

This project provides a full pipeline to **simulate a synthetic credit portfolio** and analyze it using an **interactive Power BI dashboard**. It was created to replicate common loan performance behaviors (such as default rates, delinquency buckets, and disbursements) and visualize them clearly for portfolio monitoring and business insights.

---

## 📊 1. Credit Simulator Dataset Generator

This Python script simulates the behavior of 1,000 fictional loans over time, generating both detailed historical data and a summarized view suitable for analysis in Power BI or other BI tools.

### 📌 Features
✔ Generates realistic loan data with attributes like:
- Loan amount
- Interest rate
- Loan term
- Days past due
- Product type
- Agency location

✔ Simulates customer behavior, including late and missed payments

✔ Categorizes credit quality based on days past due

✔ The credit simulator uses Python to generate realistic loan portfolio data, including:

- Credit disbursement by month
- Product type (General, Education, Taxes Loan, Travel)
- Office location (Toronto, Mississauga, etc.)
- Delinquency classification (Regular, Doubtful, Bad Debt)
- Payment behavior over time

### ⚙️ How It Works

✔ Randomly assigns:
- Product types and interest rates
- Loan amounts and terms
- Disbursement dates between Jan 2022 and Dec 2024

✔ Calculates:
- Installments using amortization formula
- Monthly behavior including potential non-payment scenarios
- Days past due and outstanding balances

✔ Simulates both good and bad payers using weighted probabilities

✔ Labels loan status (e.g., Excellent, Doubtful, Bad debt) for each monthly snapshot

### 📁 Output

- `loans_data.csv`: Full loan history per product per month
- `loans_data_filtered.csv`: final dataset used in the Power BI dashboard.

## 📊 2. Power BI Dashboard: Loan Portfolio Analysis

This Power BI report was built using the loans_data_filtered.csv dataset. It provides an interactive and comprehensive analysis of a credit portfolio, including vintage tracking, outstanding balances, delinquency trends, and product disbursement insights. The goal is to monitor loan performance over time and identify areas of concern across offices and loan types.

### 📌Features

✔ Vintage Analysis Tab:
- Tracks the performance of different loan cohorts (semesters) over time.
- Visualizes how the delinquency rate increases as loans age.
- Allows filtering by product, office, and time unit (semester/year).
- Includes donut charts for disbursement totals and product breakdown.

✔ General Analysis Tab:
- Displays the current outstanding balance and its evolution over time.
- Visualizes overdue balances by category (e.g., bad debt, doubtful 31–60 days, regular).
- Shows the distribution of outstanding amounts by product and office.
- Includes a pie chart and scatter plot for comparative analysis.

✔ File Used
- loans_data_filtered.csv: Contains anonymized loan data used for all visuals.

## Screenshot:

![Vintage Analysis](Screenshot_Vintage_Analysis.PNG)

![General Analysis](Screenshot_Gral_Analysis.PNG)

## 🔧 Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt