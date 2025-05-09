import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import pandas as pd
import numpy as np
# variables
credit_data = []
payer_good = False
loan_amount = 0
loan_term = 0
installment = 0
product_selection = 0
product_code = 0
interest_rate = 0
int_rate_year = 0
principal_payment = 0
interest_payment = 0
random_days = 0
customer_paid = 0
agency_code = 0
outstanding_balance = 0
credit_data_all = []

# constants
NUM_CREDITS = 1000
PRODUCTS = {
    1: {"name": "Education", "rate": 0.05},
    2: {"name": "General", "rate": 0.08},
    3: {"name": "Taxes loan", "rate": 0.09},
    4: {"name": "Travel", "rate": 0.10}
}
PRODUCT_WEIGHTS = [0.3, 0.4, 0.2, 0.1]
MIN_LOAN = 500
MAX_LOAN = 10000
RATE_PRODUCTS = [0.05, 0.08, 0.09, 0.1]
LOAN_TERMS = [12, 24, 36, 48, 60]
START_DATE_LOAN = datetime(2022, 1, 1)
END_DATE_LOAN = datetime(2025, 4, 30)
DELTA = END_DATE_LOAN - START_DATE_LOAN
CUT_DATE_DATA = '202503'
IMPAIRED = 180
AGENCIES = {
    10: 'Toronto', 
    20: 'North York', 
    30: 'Missisauga', 
    40: 'Pickering', 
    50: 'Whitby'
}
AMORTIZATION = 30
DETAIL_FILE_NAME = "loans_data"
SUMMARY_FILE_NAME = "loans_data_filtered"

# function to add the record of each loan
def add_record(
    p_cut_month, 
    p_pid,
    p_disbursement_date,
    p_end_date,
    p_days_past_due,
    p_loan_term,
    p_amortization,
    p_loan_amount,
    p_interest_rate,
    p_installment_amount,
    p_outstanding_balance,
    p_agency_code,
    p_agency_name,
    p_product_code,
    p_product_name
    ):
    record = {
        'cut_month': p_cut_month,
        'num_product': p_pid,
        'disbursement_date': p_disbursement_date,
        'end_date': p_end_date,
        'days_past_due': p_days_past_due,
        'loan_term': p_loan_term,
        'amortization': p_amortization,
        'loan_amount': p_loan_amount,
        'interest_rate': p_interest_rate,
        'installment_amount': p_installment_amount,
        'outstanding_balance': int(p_outstanding_balance),
        'agency_code': p_agency_code,
        'agency_name': p_agency_name,
        'product_code': p_product_code,
        'product_name': p_product_name
    }
    #print(record)
    credit_data_all.append(record)

# function to create the summary file CSV
def build_summary_data():
    # filter the data
    df_filter_credit = df_all_credits[df_all_credits['cut_month'] <= CUT_DATE_DATA].copy()
    # conditions to add a column
    conditions = [
        df_filter_credit['days_past_due'] == 0,
        df_filter_credit['days_past_due'] <= 30,
        df_filter_credit['days_past_due'] <= 40,
        df_filter_credit['days_past_due'] <= 50,
        df_filter_credit['days_past_due'] <= 60
    ]
    condition_values = [
        'Excellent',
        'Regular',
        'Doubtful 31-40',
        'Doubtful 41-50',
        'Doubtful 51-60',
    ]
    # add the column credit status acording to the days_past_due
    df_filter_credit['credit_status'] = np.select(conditions, condition_values, default = 'Bad debt')
    # convert disbursement_date to date time
    df_filter_credit['disbursement_date'] = pd.to_datetime(df_filter_credit['disbursement_date'], dayfirst=True)
    # add column disbursement_month
    df_filter_credit['disbursement_month'] = df_filter_credit['disbursement_date'].dt.strftime('%Y%m').astype(int)
    # add column to control it is a new credit
    df_filter_credit['is_new_loan'] = df_filter_credit['disbursement_month'] == df_filter_credit['cut_month'].astype(int)
    # add new column to count rows
    df_filter_credit['count_records'] = 0
    summary_data = df_filter_credit.groupby([
        'cut_month',
        'disbursement_month',
        'loan_term',
        'interest_rate',
        'agency_code',
        'agency_name',
        'product_code',
        'product_name',
        'credit_status'
    ], as_index=False).agg({
        'loan_amount': lambda x: x[df_filter_credit.loc[x.index, 'is_new_loan']].sum(),  # sum when disbursement_month == cut_month
        'disbursement_date': lambda x: x[df_filter_credit.loc[x.index, 'is_new_loan']].count(),  # count when disbursement_month == cut_month
        'outstanding_balance': 'sum',
        'count_records': 'count'  # count the rows
    }).rename(columns={
        'disbursement_date': 'loan_amount_count'
    })
    # export the file
    output_path_multi = os.path.join(script_dir, SUMMARY_FILE_NAME + ".csv")
    summary_data.to_csv(output_path_multi, index=False)
    print("The " + SUMMARY_FILE_NAME + " has been exported with " + str('{:,}'.format(len(summary_data))) + " records")
# create the ids for the credits
products_id = [f'{1000 + i}' for i in range(NUM_CREDITS)]
random_date = START_DATE_LOAN
# simulation for each credit
for pid in products_id:
    # give the kind of payer, weight of 8% to bad payer
    payer_good = random.choices([False, True], weights = [0.08, 0.92], k=1) [0]
    # give an agency code
    agency_code = random.choice(list(AGENCIES.keys()))
    agency_name = AGENCIES[agency_code]
    # give a random loan amount
    loan_amount = random.randrange(MIN_LOAN, MAX_LOAN, 10)
    # give a random product code and interest rate
    product_code = random.choices(list(PRODUCTS.keys()), weights = PRODUCT_WEIGHTS, k=1) [0]
    product_name = PRODUCTS[product_code]["name"]
    interest_rate = PRODUCTS[product_code]["rate"]
    int_rate_year = interest_rate / 12 # it takes 12 as months in the year
    # give a loan term
    loan_term = random.choice(LOAN_TERMS)
    # installment = [P * r * (1 + r)^n] / [(1 + r)^n - 1] 
    # calculate installment
    installment = round(loan_amount * ((int_rate_year) * (1 + int_rate_year)** loan_term) / (((1 + int_rate_year)** loan_term) -1), 4)
    interest_payment = loan_amount * int_rate_year
    principal_payment = installment - interest_payment
    # create the disbursement date
    random_days = random.randint(0, DELTA.days)
    disbursement_date = START_DATE_LOAN + timedelta(days=random_days)
    end_date = disbursement_date + relativedelta(months=loan_term)
    #print(random_date)
    # add the record for the disbursement to the dictionary
    add_record(disbursement_date.strftime('%Y%m'),
        pid,
        disbursement_date.strftime('%d/%m/%Y'),
        end_date.strftime('%d/%m/%Y'),
        0,
        loan_term,
        AMORTIZATION,
        loan_amount,
        interest_rate,
        installment,
        loan_amount,
        agency_code,
        agency_name,
        product_code,
        product_name
    )
    # create historical month
    date_next_add = disbursement_date
    outstanding_balance = loan_amount
    days_past_due = 0
    past_due_count = 1
    cut_date_next = 0
    loan_term_count = loan_term
    #for mon in range(1, loan_term + 1):
    while loan_term_count > 0 and int(cut_date_next) <= int(CUT_DATE_DATA):
        if not payer_good:
            # when the loan is more than impaired parameter, customer_paid=false
            if days_past_due > IMPAIRED:
                customer_paid = False
            else:
                # create random payment with 80% of weight no payment when is not good payer
                customer_paid = random.choices([False, True], weights= [0.8, 0.2], k=1) [0]
            # validate if customer paid
            if not customer_paid:
                past_due_count += 1
                # calculate the days from the payment day when starts delinquency
                if days_past_due == 0:
                    dayn = datetime.strptime(disbursement_date.strftime('%d/%m/%y'), '%d/%m/%y')
                    days_past_due = 30 - dayn.day
                else:
                    days_past_due = days_past_due + 30
            else:
                days_past_due = 0
                past_due_count = 1
        # validate if the customer paid
        if days_past_due == 0:
            loan_term_count -= 1
            outstanding_balance -= (principal_payment * past_due_count)
            # calculate the distribution of the next installment
            interest_payment = outstanding_balance * int_rate_year
            principal_payment = installment - interest_payment
        # add 1 month to the date to create the cut month
        date_next_add = date_next_add + relativedelta(months=1)
        cut_date_next = date_next_add.strftime('%Y%m')
        
        # add the record for the disbursement to the dictionary
        add_record(cut_date_next,
            pid,
            disbursement_date.strftime('%d/%m/%Y'),
            end_date.strftime('%d/%m/%Y'),
            days_past_due,
            loan_term,
            AMORTIZATION,
            loan_amount,
            interest_rate,
            installment,
            outstanding_balance,
            agency_code,
            agency_name,
            product_code,
            product_name
        )
# convert the data to a dataframe
df_all_credits = pd.DataFrame(credit_data_all)
# path to save the data
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path_multi = os.path.join(script_dir, DETAIL_FILE_NAME + ".csv")
# export the data file
df_all_credits.to_csv(output_path_multi, index=False)
print("The " + DETAIL_FILE_NAME + " has been exported with " + str('{:,}'.format(len(df_all_credits))) + " records")
# go to the function to create the summary file to use in PBI    
build_summary_data()
print("Process finished successfully!!")