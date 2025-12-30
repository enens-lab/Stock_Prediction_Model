
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta

def get_market_calendar(start_date, end_date):
    '''
    Returns a list of valid NYSE trading days between start and end date.
    Handles weekends and US market holidays.
    '''
    nyse = mcal.get_calendar('NYSE')
    
    # Get the schedule
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    
    # Return index as strings or timestamps depending on need
    # optimizing for the backtester format
    return schedule.index.strftime('%Y-%m-%d').tolist()

def is_trading_day(date_str):
    '''
    Checks if a specific date string (YYYY-MM-DD) is a market day.
    '''
    try:
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=date_str, end_date=date_str)
        return not schedule.empty
    except:
        return False

if __name__ == "__main__":
    # Test
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"Is {today} a trading day? {is_trading_day(today)}")
    
    past_days = get_market_calendar('2024-01-01', '2024-01-10')
    print(f"First trading days of 2024: {past_days}")
