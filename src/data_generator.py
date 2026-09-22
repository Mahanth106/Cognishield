# src/data_generator.py
import os
import random
import pandas as pd
from datetime import datetime, timedelta

def generate_cert_mock_data(output_dir="data/raw", num_users=20, days=14, threat_actor=None, seed=None):
    os.makedirs(output_dir, exist_ok=True)
    if seed is not None:
        random.seed(seed)
    
    start_date = datetime(2026, 8, 1, 8, 0, 0)
    users = [f"USER_{i:03d}" for i in range(1, num_users + 1)]
    if threat_actor is None:
        threat_actor = random.choice(users)
    print(f"[*] Threat scenario injected for entity: {threat_actor}")

    scenario_users = [user for user in users if user != threat_actor][:4]
    scenario_names = [
        "after-hours access",
        "repeated USB activity",
        "suspicious text without file copying",
        "moderate file activity",
    ]
    for name, user in zip(scenario_names, scenario_users):
        print(f"[*] Secondary scenario injected for entity {user}: {name}")
    
    logon_rows = []
    device_rows = []
    file_rows = []
    text_rows = []

    normal_texts = [
        "Updated quarterly projection spreadsheet",
        "Routine weekly sync with product team",
        "Refactored login module unit tests",
        "Submitted expense reimbursement forms"
    ]
    suspicious_texts = [
        "Exported full client database to external backup",
        "Bypass administrative policy override",
        "Dumped confidential salary keys and credentials",
        "Staging files before resignation"
    ]

    for day in range(days):
        current_day = start_date + timedelta(days=day)
        is_weekend = current_day.weekday() >= 5

        for u in users:
            # Baseline behavior
            if not is_weekend or random.random() < 0.05:
                # Normal login/logout
                login_hour = random.randint(8, 10)
                logout_hour = random.randint(17, 19)
                t_login = current_day.replace(hour=login_hour, minute=random.randint(0, 59))
                t_logout = current_day.replace(hour=logout_hour, minute=random.randint(0, 59))

                logon_rows.append({"timestamp": t_login, "user": u, "activity": "Logon", "pc": f"PC_{u}"})
                logon_rows.append({"timestamp": t_logout, "user": u, "activity": "Logoff", "pc": f"PC_{u}"})

                # Standard activity
                num_files = random.randint(5, 20)
                for _ in range(num_files):
                    f_time = t_login + timedelta(minutes=random.randint(10, 400))
                    file_rows.append({
                        "timestamp": f_time,
                        "user": u,
                        "filename": f"doc_{random.randint(100, 999)}.pdf",
                        "to_removable_media": 0
                    })

                if random.random() < 0.15:
                    d_time = t_login + timedelta(minutes=random.randint(20, 300))
                    device_rows.append({"timestamp": d_time, "user": u, "activity": "Connect"})

                text_rows.append({
                    "timestamp": t_login + timedelta(minutes=60),
                    "user": u,
                    "content": random.choice(normal_texts)
                })

            # Injected insider threat behavior on the last 3 days for threat actor
            if u == threat_actor and day >= (days - 3):
                # Anomaly 1: Late-night logon (contextual/temporal deviation)
                t_night = current_day.replace(hour=23, minute=random.randint(10, 45))
                logon_rows.append({"timestamp": t_night, "user": u, "activity": "Logon", "pc": f"PC_{u}"})

                # Anomaly 2: Repeated unauthorized USB connections
                for step in range(4):
                    usb_time = t_night + timedelta(minutes=10 + step * 5)
                    device_rows.append({"timestamp": usb_time, "user": u, "activity": "Connect"})

                # Anomaly 3: Bulk file access and copy to external media
                for f_idx in range(60):
                    f_time = t_night + timedelta(minutes=15 + (f_idx // 3))
                    file_rows.append({
                        "timestamp": f_time,
                        "user": u,
                        "filename": f"classified_intel_{f_idx}.dat",
                        "to_removable_media": 1
                    })

                # Anomaly 4: Lexical/NLP red flag
                text_rows.append({
                    "timestamp": t_night + timedelta(minutes=30),
                    "user": u,
                    "content": random.choice(suspicious_texts)
                })

            # Secondary scenario 1: repeated after-hours access without exfiltration.
            if scenario_users and u == scenario_users[0] and day >= (days - 3):
                for offset in (0, 25):
                    late_time = current_day.replace(hour=22, minute=10 + offset)
                    logon_rows.append({
                        "timestamp": late_time,
                        "user": u,
                        "activity": "Logon",
                        "pc": f"PC_{u}"
                    })

            # Secondary scenario 2: repeated removable-media connections.
            if len(scenario_users) > 1 and u == scenario_users[1] and day >= (days - 3):
                for step in range(3):
                    device_rows.append({
                        "timestamp": current_day.replace(hour=18, minute=10 + step * 5),
                        "user": u,
                        "activity": "Connect"
                    })

            # Secondary scenario 3: suspicious communication with no external copies.
            if len(scenario_users) > 2 and u == scenario_users[2] and day == days - 2:
                text_rows.append({
                    "timestamp": current_day.replace(hour=16, minute=30),
                    "user": u,
                    "content": "Reviewed confidential backup policy with the administrator"
                })

            # Secondary scenario 4: moderate file volume, all kept on the workstation.
            if len(scenario_users) > 3 and u == scenario_users[3] and day >= (days - 3):
                for f_idx in range(18):
                    file_rows.append({
                        "timestamp": current_day.replace(hour=15, minute=5 + (f_idx % 40)),
                        "user": u,
                        "filename": f"project_archive_{day}_{f_idx}.pdf",
                        "to_removable_media": 0
                    })

    pd.DataFrame(logon_rows).to_csv(f"{output_dir}/logon.csv", index=False)
    pd.DataFrame(device_rows).to_csv(f"{output_dir}/device.csv", index=False)
    pd.DataFrame(file_rows).to_csv(f"{output_dir}/file.csv", index=False)
    pd.DataFrame(text_rows).to_csv(f"{output_dir}/text.csv", index=False)
    print(f"Data generation complete. Files saved in {output_dir}")

if __name__ == "__main__":
    generate_cert_mock_data(seed=42)