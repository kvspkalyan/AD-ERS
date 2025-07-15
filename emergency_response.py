from twilio.rest import Client
import tkinter as tk
from tkinter import messagebox

def trigger_emergency():
    print("Emergency Triggered! Sending SMS...")

    # Twilio credentials (replace with real ones)
    account_sid = 'YOUR_ACCOUNT_SID'
    auth_token = 'YOUR_AUTH_TOKEN'
    from_phone = '+1234567890'
    to_phone = '+91XXXXXXXXXX'

    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body="Accident Detected! Immediate assistance required.",
            from_=from_phone,
            to=to_phone
        )
        print("SMS sent! SID:", message.sid)
    except Exception as e:
        print("Failed to send SMS:", e)

    # Pop-up window alert
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning("Accident Alert", " Accident Detected! Emergency alert sent.")
    root.destroy()
