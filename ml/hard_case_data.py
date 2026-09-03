from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUGMENTATION = PROJECT_ROOT / "data" / "augmentation" / "hard_cases.csv"
DEFAULT_CHALLENGE_DIR = PROJECT_ROOT / "evaluation" / "challenge_emails"
SEED = 240826

SOURCE_BASIS = {
    "google_security": "https://support.google.com/accounts/answer/6063333?hl=en",
    "gift_card": "https://consumer.ftc.gov/consumer-alerts/2026/01/no-thats-not-your-boss-asking-you-buy-gift-cards",
    "email_authentication": "https://www.rfc-editor.org/info/rfc9989/",
}


def _record(
    record_id: str,
    label: int,
    scenario: str,
    subject: str,
    body: str,
    purpose: str,
) -> dict[str, object]:
    text = f"Subject: {subject}\n\nBody: {body}"
    return {
        "id": record_id,
        "example_id": hashlib.sha256(text.lower().encode("utf-8")).hexdigest()[:16],
        "expected_label": "phishing" if label else "legitimate",
        "label": label,
        "scenario": scenario,
        "subject": subject,
        "body": body,
        "text": text,
        "purpose": purpose,
        "provenance": "controlled-synthetic",
    }


def build_augmentation() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    services = ["mail", "student portal", "cloud storage", "payroll", "library", "travel account"]
    devices = ["Windows laptop", "Android phone", "Mac browser", "new tablet"]
    locations = ["Bristol", "Cardiff", "London", "Manchester"]
    amounts = ["2.50 GBP", "18.40 GBP", "42.00 GBP", "76.25 GBP"]
    references = ["PR-1042", "TX-8821", "RC-3107", "INV-6240"]
    deadlines = ["today", "before 4 PM", "within two hours", "before the office closes"]
    roles = ["department head", "finance manager", "course leader", "operations director"]
    brands = ["Example Mail", "Example Cloud", "Example Campus", "Example Payments"]

    legitimate_templates = [
        (
            "security_notice",
            "New sign-in to your {service}",
            "A sign-in from a {device} near {location} was recorded. If this was you, no action is required. If it was not you, open the official {service} application directly and review recent activity. We will never ask for your password by email.",
        ),
        (
            "security_notice",
            "Security settings changed",
            "Two-step verification settings were updated for your {service}. You can review the change from the security page after opening the service normally. Do not reply with credentials or verification codes.",
        ),
        (
            "receipt",
            "Payment receipt {reference}",
            "Your payment of {amount} was processed successfully. Reference {reference}. This receipt is for your records and no response is needed.",
        ),
        (
            "invoice",
            "Invoice approved for scheduled payment",
            "Invoice {reference} has passed the normal approval workflow and is scheduled for payment. Bank details have not changed. Questions should be raised through the finance portal.",
        ),
        (
            "urgent_work",
            "Room change for today's meeting",
            "The meeting has moved to room 2B14 and still starts at 10:00. Please tell the project group before the session begins. No purchase or account action is required.",
        ),
        (
            "account_update",
            "Your monthly {brand} statement is ready",
            "Your monthly statement is available in the {brand} application. Open the application directly to view it. This notification does not request payment details or a password.",
        ),
        (
            "delivery",
            "Parcel delivered to reception",
            "Your parcel was delivered to the main reception desk at 13:20. Bring your identification when collecting it during opening hours. There is no delivery fee outstanding.",
        ),
        (
            "newsletter",
            "Service maintenance this weekend",
            "Planned maintenance will affect the {service} between 01:00 and 03:00 on Sunday. Existing files and account settings will remain unchanged. Status updates are available in the usual service dashboard.",
        ),
    ]
    phishing_templates = [
        (
            "credential_theft",
            "Urgent: {service} access expires {deadline}",
            "Your {service} will be disabled {deadline}. Sign in at https://{brand_slug}-verify.example.invalid and enter your password immediately to prevent closure.",
        ),
        (
            "security_impersonation",
            "Unusual sign-in requires verification",
            "We blocked a sign-in from {location}. Confirm your identity at https://security-check.example.invalid and provide your password and one-time code now.",
        ),
        (
            "gift_card_bec",
            "Confidential request from the {role}",
            "I am in a meeting and need an urgent favour. Buy four gift cards {deadline}, then reply with photographs of the codes. Keep this request confidential.",
        ),
        (
            "payment_diversion",
            "Updated bank details for invoice {reference}",
            "Do not use the bank account on our previous invoice. Transfer {amount} to the replacement account in the attached instructions {deadline} and send confirmation.",
        ),
        (
            "refund_theft",
            "Refund waiting for confirmation",
            "A refund of {amount} is waiting. Submit your card number and security code at https://refund-release.example.invalid {deadline} or the refund will be cancelled.",
        ),
        (
            "attachment_malware",
            "Unpaid invoice escalated",
            "Final notice for invoice {reference}. Open the attached macro-enabled document and enable editing to see the amount due. Payment must be made {deadline}.",
        ),
        (
            "payroll_impersonation",
            "Payroll account must be reconfirmed",
            "Payroll processing has stopped. Reply with your bank account, date of birth and employee password {deadline} so salary can be released.",
        ),
        (
            "delivery_theft",
            "Delivery held - small fee required",
            "Your parcel is on hold. Pay a redelivery fee of {amount} at https://parcel-fee.example.invalid {deadline} to prevent return to sender.",
        ),
    ]

    records: list[dict[str, object]] = []
    slots = []
    for index in range(24):
        brand = brands[index % len(brands)]
        slots.append(
            {
                "service": services[index % len(services)],
                "device": devices[(index * 3) % len(devices)],
                "location": locations[(index * 5) % len(locations)],
                "amount": amounts[(index * 7) % len(amounts)],
                "reference": references[(index * 11) % len(references)],
                "deadline": deadlines[(index * 13) % len(deadlines)],
                "role": roles[(index * 17) % len(roles)],
                "brand": brand,
                "brand_slug": brand.lower().replace(" ", "-"),
            }
        )
    rng.shuffle(slots)

    counter = 1
    for label, templates in ((0, legitimate_templates), (1, phishing_templates)):
        for scenario, subject_template, body_template in templates:
            for values in slots:
                records.append(
                    _record(
                        f"aug-{counter:04d}",
                        label,
                        scenario,
                        subject_template.format(**values),
                        body_template.format(**values),
                        "training-only",
                    )
                )
                counter += 1
    rng.shuffle(records)
    return records


def _challenge_templates(split: str) -> list[tuple[int, str, str, str]]:
    if split == "development":
        return [
            (0, "security_notice", "New sign-in on a Windows device", "A new sign-in was recorded on your account. If this was you, no action is required. If not, open the official account application directly and review your recent security activity."),
            (0, "security_notice", "Password changed successfully", "Your password was changed at 09:42. This is a confirmation only. If you did not make this change, use the security page in the application you normally use."),
            (0, "receipt", "Campus printing receipt", "Your printing balance was charged 3.20 GBP. Reference CP-4102. This automated receipt is for your records and no reply is required."),
            (0, "invoice", "Invoice INV-2204 approved", "Invoice INV-2204 completed the normal approval process. Payment will be made to the bank details already held in the supplier system."),
            (0, "urgent_work", "Urgent timetable correction", "Today's laboratory starts at 13:00 rather than 14:00. Please notify your group before travelling to campus. No account action is required."),
            (0, "account_update", "Storage limit reminder", "Your storage is approaching its plan limit. Review large files from the storage application when convenient. This notice does not request credentials or payment."),
            (1, "credential_theft", "Mailbox suspended today", "Your mailbox has been suspended. Restore access at https://mail-restore.example.invalid and enter your password within 30 minutes."),
            (1, "security_impersonation", "Suspicious login detected", "A foreign login was detected. Verify your identity at https://identity-review.example.invalid using your password and one-time code."),
            (1, "gift_card_bec", "Quick confidential favour", "I cannot speak because I am in a meeting. Purchase five gift cards now and email the codes to me. Do not discuss this request with anyone."),
            (1, "payment_diversion", "Revised account for today's transfer", "Our bank details changed this morning. Ignore the previous invoice and transfer the outstanding balance to the replacement account before 3 PM."),
            (1, "refund_theft", "Tax refund expires tonight", "Claim your tax refund at https://refund-confirm.example.invalid. Enter your card number and security code before midnight."),
            (1, "attachment_malware", "Encrypted invoice attached", "Open the attached document and enable macros to decrypt invoice INV-9381. Failure to pay today will incur a penalty."),
        ]
    if split == "test":
        return [
            (0, "security_notice", "Review of recent account activity", "We noticed a sign-in from a new browser. If you recognise it, you do not need to do anything. Otherwise, visit the service by typing its normal address yourself and check recent activity."),
            (0, "security_notice", "Two-step verification enabled", "Two-step verification was enabled on your account. This message confirms the change and will never ask you to send a password or verification code."),
            (0, "receipt", "Receipt for library replacement card", "A charge of 5.00 GBP was applied for a replacement library card. Receipt LC-9014 is retained in your library account. No response is needed."),
            (0, "invoice", "Scheduled supplier payment", "The approved payment for purchase order PO-7118 is scheduled for Friday using the supplier details already on file. Contact finance through the staff directory with questions."),
            (0, "urgent_work", "Building closes early today", "The main building will close at 17:00 today because of maintenance. Please collect belongings before closing and follow the usual access procedure tomorrow."),
            (0, "account_update", "Your annual privacy summary", "The annual privacy summary is available from the settings page of the application. No immediate action is required, and this email does not collect account information."),
            (1, "credential_theft", "Email quota exceeded", "Incoming mail is blocked. Validate your account at https://quota-release.example.invalid and provide your username and password immediately."),
            (1, "security_impersonation", "Device verification pending", "A new device is waiting for approval. Enter your email password and authentication code at https://device-approval.example.invalid within one hour."),
            (1, "gift_card_bec", "Request from senior management", "I need you to handle a sensitive purchase. Buy six gift cards and send clear photographs of every PIN. Keep this between us until the meeting ends."),
            (1, "payment_diversion", "Outstanding invoice - banking amendment", "The account on our invoice is no longer valid. Send payment to the new account listed below today and confirm the transfer by reply."),
            (1, "refund_theft", "Overpayment refund available", "You paid too much on your last bill. Confirm your bank and card details at https://billing-refund.example.invalid to receive the money."),
            (1, "attachment_malware", "Voicemail document requires access", "An encrypted voicemail is attached. Open the document, enable content and sign in with your mailbox password to listen."),
        ]
    raise ValueError(f"Unknown split: {split}")


def build_challenge(split: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    variants = {
        "development": [
            ("", ""),
            ("Reminder: ", " Please keep this notification for your records."),
            ("Notification: ", " This message was generated automatically."),
            ("Update: ", " Contact the organisation through its normal website if you have questions."),
        ],
        "test": [
            ("", ""),
            ("Important: ", " Do not reply to this automated notification."),
            ("Account notice: ", " Use a trusted bookmark if you need to contact support."),
            ("Service message: ", " Keep this email as a record of the notification."),
        ],
    }[split]
    counter = 1
    for label, scenario, subject, body in _challenge_templates(split):
        for subject_prefix, body_suffix in variants:
            records.append(
                _record(
                    f"challenge-{split[:3]}-{counter:03d}",
                    label,
                    scenario,
                    subject_prefix + subject,
                    body + body_suffix,
                    f"{split}-challenge",
                )
            )
            counter += 1
    return records


def write_outputs(augmentation_path: Path, challenge_dir: Path) -> dict[str, object]:
    augmentation_candidates = build_augmentation()
    augmentation = list(
        {
            str(row["example_id"]): row for row in augmentation_candidates
        }.values()
    )
    development = build_challenge("development")
    test = build_challenge("test")

    augmentation_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(augmentation).to_csv(augmentation_path, index=False)
    challenge_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("development", development), ("test", test)):
        (challenge_dir / f"{name}.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
    manifest = {
        "seed": SEED,
        "augmentation_candidate_rows": len(augmentation_candidates),
        "augmentation_unique_training_rows": len(augmentation),
        "development_rows": len(development),
        "test_rows": len(test),
        "class_balance": "50 percent legitimate, 50 percent phishing",
        "separation": {
            "augmentation": "training-only",
            "development": "policy-selection-only",
            "test": "untouched-final-evaluation",
        },
        "source_basis": SOURCE_BASIS,
        "safety": "No live links, attachments, personal data, or real account content.",
        "limitation": "Controlled synthetic scenarios support robustness testing but are not representative real-world prevalence data.",
    }
    (challenge_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build separated hard-case research datasets.")
    parser.add_argument("--augmentation", type=Path, default=DEFAULT_AUGMENTATION)
    parser.add_argument("--challenge-dir", type=Path, default=DEFAULT_CHALLENGE_DIR)
    args = parser.parse_args()
    print(json.dumps(write_outputs(args.augmentation, args.challenge_dir), indent=2))


if __name__ == "__main__":
    main()
