"""Notification utility module for training status updates via email.

This module provides functions to notify the user when training finishes
or fails using yagmail and environment-stored Gmail credentials.

Functions:
    notify: Sends a success notification with a message.
"""

import os
from typing import Optional
import yagmail
from dotenv import load_dotenv

load_dotenv()

GMAIL_USERNAME: Optional[str] = os.getenv("GMAIL_USERNAME")
APP_PASSWORD: Optional[str] = os.getenv("APP_PASSWORD")

if not GMAIL_USERNAME or not APP_PASSWORD:
    raise EnvironmentError("GMAIL_USERNAME or APP_PASSWORD missing from .env")

yag = yagmail.SMTP(GMAIL_USERNAME, APP_PASSWORD)


def notify(subject: str, message_or_trace: str) -> None:
    """Sends an email notification for successful completion.

    Args:
        subject (str): Subject line of the email.
        message (str): Message or trace to be sent
    """
    if isinstance(message_or_trace, str):
        yag.send(GMAIL_USERNAME, subject, [message_or_trace])
    else:
        yag.send(GMAIL_USERNAME, subject, message_or_trace)
