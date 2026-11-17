
import smtplib
import os
from email.message import EmailMessage


def send_mail(recipient_email, Subject, body):

    # Define email details
    sender_email = os.getenv("GMAIL_EMAIL")
    sender_password = os.getenv("GMAIL_APP_PASSWORD")

    if not sender_email or not sender_password:
        return "Failed: Gmail credentials are missing."

    # Create the message
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = Subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    # Connect to the SMTP server and send
    try:

        with smtplib.SMTP("smtp.gmail.com", 587) as server:

            server.ehlo()
            server.starttls()
            server.ehlo()

            server.login(sender_email, sender_password)

            server.send_message(msg)

        return "Email sent successfully!"

    except Exception as e:

        return f"Failed to send email: {e}"
