import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import pandas as pd
import numpy as np

class CreditSimulation:
    """
    Class to simulate the credit data for a financial institution
    """
    # Initialize the class with the necessary parameters
    def __init__(self):
        # set the random seed for reproducibility
        random.seed(42)
        ### Configration
        self.NUM_CREDITS = 1000 # number of credits to create
        self.PRODUCTS = { 
            1: {"name": "Education", "rate": 0.05, "weight": 0.3},
            2: {"name": "General", "rate": 0.08, "weight": 0.4},
            3: {"name": "Taxes loan", "rate": 0.09, "weight": 0.2},
            4: {"name": "Travel", "rate": 0.10, "weight": 0.1}
        } # products available to create the data
        self.MIN_LOAN = 500 # minimum amount of loan in the simulation
        self.MAX_LOAN = 10000 # maximum amount of loan in the simulation
        self.LOAN_TERMS = [12, 24, 36, 48, 60] # terms available to create the loans
        self.START_DATE_LOAN = datetime(2022, 1, 1) # minimum date to create a disbursement (year, month, day)
        self.END_DATE_LOAN = datetime(2025, 4, 30) # maximum date to create a disbursement (year, month, day)
        self.DELTA = self.END_DATE_LOAN - self.START_DATE_LOAN # delta to create the random date for disbursement
        self.CUT_DATE_DATA = 202503 # last cut month of the data simulation (yyyymm)
        self.IMPAIRED = 180 # limit to credit be impaired
        self.BAD_PAYER_WEIGHT = 0.08 # bad payer weight to simulate bad loans
        self.BAD_PAYER_PAY_WEIGHT = 0.08 # weight to a bad payer pay
        self.DAYS_MONTH = 30 # days in the month to calculate the days past due
        self.AGENCIES = {
            # offices of the simulated company
            10: 'Toronto', 
            20: 'North York', 
            30: 'Missisauga', 
            40: 'Pickering', 
            50: 'Whitby'
        }
        self.PRODUCT_COD_INIT = 1000 # initial code of the product
        self.AMORTIZATION = 30 # all the credits created with monthly amortization
        self.DETAIL_FILE_NAME = "loans_data" # name of the detail file data
        self.SUMMARY_FILE_NAME = "loans_data_filtered" # name of the summary file data
        ### variables
        # path to save the data
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.payer_good = False
        self.loan_amount = 0
        self.loan_term = 0
        self.installment = 0
        self.product_selection = 0
        self.product_code = 0
        self.interest_rate = 0
        self.int_rate_year = 0
        self.principal_payment = 0
        self.interest_payment = 0
        self.random_days = 0
        self.customer_paid = 0
        self.agency_code = 0
        self.outstanding_balance = 0
        self.credit_data_all = []

    # function to add the record of each loan
    def add_record(
        self,
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
        self.credit_data_all.append(record)

    # function to create the summary file CSV
    def build_summary_data(self, df_all_credits):
        # filter the data
        df_filter_credit = df_all_credits[df_all_credits['cut_month'] <= str(self.CUT_DATE_DATA)].copy()
        # add the column credit status acording to the days_past_due
        df_filter_credit['credit_status'] = np.select(
            [
                df_filter_credit['days_past_due'] == 0,
                df_filter_credit['days_past_due'].between(1, 30),
                df_filter_credit['days_past_due'].between(31, 40),
                df_filter_credit['days_past_due'].between(41, 50),
                df_filter_credit['days_past_due'].between(51, 60)
            ],
            ['Excellent', 'Regular', 'Doubtful 31-40', 'Doubtful 41-50', 'Doubtful 51-60'],
            default='Bad debt'
        )
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
        output_path_multi = os.path.join(self.script_dir, self.SUMMARY_FILE_NAME + ".csv")
        summary_data.to_csv(output_path_multi, index=False)
        print("The " + self.SUMMARY_FILE_NAME + " has been exported with " + str('{:,}'.format(len(summary_data))) + " records")
    
    # function to create the detail file CSV
    def build_detail_data(self, df_all_credits):
        output_path_multi = os.path.join(self.script_dir, self.DETAIL_FILE_NAME + ".csv")
        # export the data file
        df_all_credits.to_csv(output_path_multi, index=False)
        print("The " + self.DETAIL_FILE_NAME + " has been exported with " + str('{:,}'.format(len(df_all_credits))) + " records")
    
    # function to simulate the credit data.
    def simulate(self):
        # create the ids for the credits
        products_id = [f'{self.PRODUCT_COD_INIT + i}' for i in range(self.NUM_CREDITS)]
        # simulation for each credit
        for pid in products_id:
            # give the kind of payer, weight of 8% to bad payer
            payer_good = random.choices([False, True], weights = [self.BAD_PAYER_WEIGHT, 1 - self.BAD_PAYER_WEIGHT], k=1) [0]
            # give an agency code
            agency_code = random.choice(list(self.AGENCIES.keys()))
            agency_name = self.AGENCIES[agency_code]
            # give a random loan amount
            loan_amount = random.randrange(self.MIN_LOAN, self.MAX_LOAN, 10)
            # give a random product code and interest rate
            product_code_list = list(self.PRODUCTS.keys())
            # create a list of weights for the products
            product_weights = [self.PRODUCTS[code]["weight"] for code in product_code_list]
            product_code = random.choices(list(self.PRODUCTS.keys()), weights = product_weights, k=1) [0]
            product_name = self.PRODUCTS[product_code]["name"]
            interest_rate = self.PRODUCTS[product_code]["rate"]
            int_rate_year = interest_rate / 12 # it takes 12 as months in the year
            # give a loan term
            loan_term = random.choice(self.LOAN_TERMS)
            # installment = [P * r * (1 + r)^n] / [(1 + r)^n - 1] 
            # calculate installment
            installment = round(loan_amount * ((int_rate_year) * (1 + int_rate_year)** loan_term) / (((1 + int_rate_year)** loan_term) -1), 4)
            interest_payment = loan_amount * int_rate_year
            principal_payment = installment - interest_payment
            # create the disbursement date
            random_days = random.randint(0, self.DELTA.days)
            disbursement_date = self.START_DATE_LOAN + timedelta(days=random_days)
            end_date = disbursement_date + relativedelta(months=loan_term)
            #print(random_date)
            # add the record for the disbursement to the dictionary
            self.add_record(disbursement_date.strftime('%Y%m'),
                pid,
                disbursement_date.strftime('%d/%m/%Y'),
                end_date.strftime('%d/%m/%Y'),
                0,
                loan_term,
                self.AMORTIZATION,
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
            while loan_term_count > 0 and int(cut_date_next) <= int(self.CUT_DATE_DATA):
                if not payer_good:
                    # when the loan is more than impaired parameter, customer_paid=false
                    if days_past_due > self.IMPAIRED:
                        customer_paid = False
                    else:
                        # create random payment with 80% of weight no payment when is not good payer
                        customer_paid = random.choices([False, True], weights= [self.BAD_PAYER_PAY_WEIGHT, 1 - self.BAD_PAYER_PAY_WEIGHT], k=1) [0]
                    # validate if customer paid
                    if not customer_paid:
                        past_due_count += 1
                        # calculate the days from the payment day when starts delinquency
                        if days_past_due == 0:
                            dayn = datetime.strptime(disbursement_date.strftime('%d/%m/%y'), '%d/%m/%y')
                            days_past_due = self.DAYS_MONTH - dayn.day
                        else:
                            days_past_due = days_past_due + self.DAYS_MONTH
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
                self.add_record(cut_date_next,
                    pid,
                    disbursement_date.strftime('%d/%m/%Y'),
                    end_date.strftime('%d/%m/%Y'),
                    days_past_due,
                    loan_term,
                    self.AMORTIZATION,
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
        df_all_credits = pd.DataFrame(self.credit_data_all)
        # go to the function to create the detailed file to use in PBI    
        self.build_detail_data(df_all_credits)
        # go to the function to create the summary file to use in PBI    
        self.build_summary_data(df_all_credits)
        print("Process finished successfully!!")
    
if __name__ == "__main__":
    # create the object
    credit_simulation = CreditSimulation()
    # call the function to simulate the data
    credit_simulation.simulate()