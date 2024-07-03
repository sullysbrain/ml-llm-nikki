"""
File: report_generator.py
Author: Scott Sullivan
Created: 2024-07-01
Description:
    This module will generate a formatted report

Functions:
"""

# Variable Loaders
import streamlit as st
from datetime import datetime, time
import threading
import time as time_module

def generate_daily_report():
    st.write("Daily report generated at", datetime.now())

def generate_post_show_report():
    st.write("Post-show report generated at", datetime.now())


def is_time_for_report(target_hour, target_minute):
    now = datetime.now()
    return now.hour == target_hour and now.minute == target_minute


def check_and_run_reports():
    while True:
        if is_time_for_report(21, 27):
            generate_daily_report()
    
        if is_time_for_report(21, 28):
            generate_post_show_report()    
        
        # Sleep for 60 seconds before checking again
        time_module.sleep(60)


# Start the report checker in a separate thread
threading.Thread(target=check_and_run_reports, daemon=True).start()
# Your main Streamlit app code here
st.title("Theatre Production Management")

while True:
    st.write("This is the main Streamlit app running in the main thread.")
    check_and_run_reports()
    time_module.sleep(60)
