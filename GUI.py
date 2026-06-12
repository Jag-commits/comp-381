import tkinter as tk
import tkinter.messagebox as mb
from tkinter.filedialog import askopenfilename

import customtkinter as ctk
import pandas as pd

from MLRunner import MLRunner

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Fraud Detector")
root.geometry("480x620")

file_path     = tk.StringVar()
selected_algo = tk.StringVar()

# ── Headline ──────────────────────────────────────────────────────────────────
ctk.CTkLabel(root, text="Fraud Detector", font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(20, 10))

# ── File selector ─────────────────────────────────────────────────────────────
file_label = ctk.CTkLabel(root, text="No file selected", text_color="gray60")
file_label.pack()

def browse():
    path = askopenfilename(filetypes=[("CSV files", "*.csv")])
    if path:
        file_path.set(path)
        file_label.configure(text=path, text_color="#4CAF50")

ctk.CTkButton(root, text="Browse CSV", command=browse).pack(pady=(6, 16))

# ── Algorithm radio buttons ───────────────────────────────────────────────────
ctk.CTkLabel(root, text="Algorithm", font=ctk.CTkFont(size=14, weight="bold")).pack()
ctk.CTkRadioButton(root, text="Logistic Regression", variable=selected_algo, value="option_2").pack(pady=4)
ctk.CTkRadioButton(root, text="Random Forest",       variable=selected_algo, value="option_1").pack(pady=4)

# ── Results ───────────────────────────────────────────────────────────────────
results_label = ctk.CTkLabel(root, text="", justify="left", wraplength=440)
results_label.pack(pady=(16, 4))

sample_box = ctk.CTkTextbox(root, height=160, state="disabled")
sample_box.pack(fill="x", padx=16, pady=(0, 16))

# ── Run ───────────────────────────────────────────────────────────────────────
def on_run():
    if not file_path.get():
        mb.showwarning("No file", "Please select a CSV file first.")
        return
    if not selected_algo.get():
        mb.showwarning("No algorithm", "Please choose an algorithm.")
        return

    df = pd.read_csv(file_path.get())
    result = MLRunner.run("Finance Dataset", selected_algo.get(), df)

    if result["error"]:
        results_label.configure(text=f"Error: {result['error']}", text_color="#F44336")
        return

    results_label.configure(text=(
        f"Algorithm: {result['algo_label']}\n"
        f"Anomalies: {result['anomaly_count']}  |  Rate: {result['anomaly_pct']}%\n"
        f"Saved to:  {result['output_path']}"
    ), text_color="white")

    sample_box.configure(state="normal")
    sample_box.delete("0.0", "end")
    sample_box.insert("0.0", result["sample_rows"] or "No anomalies found.")
    sample_box.configure(state="disabled")

ctk.CTkButton(root, text="Run Detection", font=ctk.CTkFont(size=14, weight="bold"),
              height=42, command=on_run).pack(pady=8)

root.mainloop()