"""
layer8_email.py — Layer 8: Email Integration
Sends rejection and shortlist emails to candidates via Gmail SMTP.

Setup:
  1. Gmail → Security → Enable 2-Step Verification
  2. Search "App Passwords" → create one for "Mail"
  3. Add to .env file:
       SENDER_EMAIL=yourname@gmail.com
       SENDER_APP_PASSWORD=xxxx xxxx xxxx xxxx
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()


class EmailSender:
    def __init__(self):
        self.sender_email    = os.getenv("SENDER_EMAIL", "")
        self.sender_password = os.getenv("SENDER_APP_PASSWORD", "")
        self.smtp_host       = "smtp.gmail.com"
        self.smtp_port       = 587

    def _is_configured(self) -> bool:
        return bool(self.sender_email and self.sender_password)

    def _send(self, to_email: str, subject: str, html_body: str) -> dict:
        if not self._is_configured():
            return {"status": "not_configured", "message": "Set SENDER_EMAIL and SENDER_APP_PASSWORD in .env"}

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = self.sender_email
        msg["To"]      = to_email
        msg.attach(MIMEText(html_body, "html"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, to_email, msg.as_string())
            return {"status": "sent", "message": f"Email sent to {to_email}"}
        except smtplib.SMTPAuthenticationError:
            return {"status": "error", "message": "Authentication failed. Check your App Password in .env"}
        except smtplib.SMTPException as e:
            return {"status": "error", "message": str(e)}

    def send_rejection(self, candidate_name: str, candidate_email: str,
                       job_title: str, company_name: str = "Our Company") -> dict:
        subject = f"Your Application for {job_title} — Update"
        return self._send(candidate_email, subject,
                          _rejection_template(candidate_name, job_title, company_name))

    def send_shortlist(self, candidate_name: str, candidate_email: str, job_title: str,
                       company_name: str = "Our Company",
                       next_steps: str = "We will contact you shortly to schedule an interview.") -> dict:
        subject = f"Good News — {job_title} Application Update"
        return self._send(candidate_email, subject,
                          _shortlist_template(candidate_name, job_title, company_name, next_steps))


def _rejection_template(name: str, job_title: str, company: str) -> str:
    first_name = (name or "Candidate").split()[0]
    return f"""
    <html><body style="font-family:Arial,sans-serif;color:#333;max-width:600px;margin:auto;padding:24px;">
      <h2 style="color:#1a1a2e;">Application Update</h2>
      <p>Dear {first_name},</p>
      <p>Thank you for applying for the <strong>{job_title}</strong> position at <strong>{company}</strong>.</p>
      <p>After carefully reviewing your application, we regret to inform you that we will not be moving
      forward with your candidacy at this time.</p>
      <p>We encourage you to apply for future openings that match your skills and experience.</p>
      <p>We wish you all the best in your job search.</p>
      <br><p>Warm regards,<br><strong>HR Team</strong><br>{company}</p>
      <hr style="border:none;border-top:1px solid #eee;margin-top:32px;">
      <p style="font-size:11px;color:#999;">This is an automated message. Please do not reply.</p>
    </body></html>"""


def _shortlist_template(name: str, job_title: str, company: str, next_steps: str) -> str:
    first_name = (name or "Candidate").split()[0]
    return f"""
    <html><body style="font-family:Arial,sans-serif;color:#333;max-width:600px;margin:auto;padding:24px;">
      <h2 style="color:#1a1a2e;">Congratulations!</h2>
      <p>Dear {first_name},</p>
      <p>We are pleased to inform you that your application for the <strong>{job_title}</strong> position
      at <strong>{company}</strong> has been shortlisted.</p>
      <p>{next_steps}</p>
      <p>Please feel free to reach out if you have any questions.</p>
      <br><p>Best regards,<br><strong>HR Team</strong><br>{company}</p>
      <hr style="border:none;border-top:1px solid #eee;margin-top:32px;">
      <p style="font-size:11px;color:#999;">This is an automated message from our recruitment system.</p>
    </body></html>"""