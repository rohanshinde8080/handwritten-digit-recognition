import os

# Bind to 0.0.0.0 on the port assigned by Render / Cloud host
port = os.environ.get("PORT", "5000")
bind = f"0.0.0.0:{port}"

# Worker configuration
workers = 1
threads = 2
timeout = 120
