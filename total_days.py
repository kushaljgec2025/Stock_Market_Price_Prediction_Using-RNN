from datetime import datetime

def calculate_days_between_dates(start_date, end_date):
    # Convert the date strings to datetime objects
   
    start_date_obj = datetime.strptime(start_date, '%d-%m-%Y')
    end_date_obj = datetime.strptime(end_date, '%d-%m-%Y')

    # Calculate the difference between the two dates
    delta = end_date_obj - start_date_obj

    # Return the difference in days
    return delta.days

# start_date='02-12-2023'
# end_date='20-07-2025'
# print(calculate_days_between_dates(start_date, end_date))