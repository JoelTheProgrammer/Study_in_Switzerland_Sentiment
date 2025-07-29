import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os

# Load data
CSV_PATH = "data/preprocessed/sentiment_posts.csv"
data = pd.read_csv(CSV_PATH)

MAIN_LANGS = ["de", "en", "fr", "it"]

# --- Filtering logic ---
def filter_data(df, allow_duplicates, allow_multiple_per_author, lang_order, sort_order, prioritize, include_comments):
    df = df[df["is_about_study"] == True]

    if not include_comments:
        df = df[df["type"] == "post"]

    if not allow_multiple_per_author:
        df = df.drop_duplicates(subset=['author'])

    if not allow_duplicates:
        df = df.drop_duplicates(subset=['title', 'selftext'])

    if prioritize == "Recency":
        df = df.sort_values("created_utc", ascending=(sort_order == "Oldest first"))
    else:
        df["lang_priority"] = df["lang"].apply(lambda x: lang_order.index(x) if x in lang_order else len(lang_order))
        df = df.sort_values("lang_priority")

    return df

# --- Plotting ---
def plot_pie_chart(title, labels, sizes):
    plt.figure()
    plt.title(title)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.show()

# --- GUI ---
class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Dashboard")

        self.allow_dupes = tk.BooleanVar(value=True)
        self.allow_multi_author = tk.BooleanVar(value=True)
        self.include_comments = tk.BooleanVar(value=True)

        tk.Checkbutton(root, text="Allow duplicate posts", variable=self.allow_dupes).pack(anchor="w")
        tk.Checkbutton(root, text="Allow multiple posts from same author", variable=self.allow_multi_author).pack(anchor="w")
        tk.Checkbutton(root, text="Include comments", variable=self.include_comments).pack(anchor="w")

        tk.Label(root, text="Comment Importance:").pack(anchor="w")
        self.comment_weight = ttk.Combobox(root, values=["Lower", "Equal", "Higher"])
        self.comment_weight.current(1)
        self.comment_weight.pack(fill="x")

        tk.Label(root, text="Language Preference Order (comma-separated ISO)").pack(anchor="w")
        self.lang_entry = tk.Entry(root)
        self.lang_entry.insert(0, "de,en,fr,it")
        self.lang_entry.pack(fill="x")

        tk.Label(root, text="Sort by:").pack(anchor="w")
        self.sort_by = ttk.Combobox(root, values=["Newest first", "Oldest first"])
        self.sort_by.current(0)
        self.sort_by.pack(fill="x")

        tk.Label(root, text="Prioritize:").pack(anchor="w")
        self.priority = ttk.Combobox(root, values=["Recency", "Language preference"])
        self.priority.current(0)
        self.priority.pack(fill="x")

        tk.Button(root, text="Generate Pie Charts", command=self.run_analysis).pack(pady=10)

    def run_analysis(self):
        try:
            lang_order = [l.strip() for l in self.lang_entry.get().split(",") if l.strip()]
            filtered = filter_data(
                data.copy(),
                allow_duplicates=self.allow_dupes.get(),
                allow_multiple_per_author=self.allow_multi_author.get(),
                lang_order=lang_order,
                sort_order=self.sort_by.get(),
                prioritize=self.priority.get(),
                include_comments=self.include_comments.get()
            )

            if filtered.empty:
                messagebox.showwarning("No Data", "No posts match the selected filters.")
                return

            # Determine comment weight
            if self.comment_weight.get() == "Lower":
                comment_weight = 0.5
            elif self.comment_weight.get() == "Higher":
                comment_weight = 1.5
            else:
                comment_weight = 1.0

            def weighted_counter(df, field):
                counts = Counter()
                for _, row in df.iterrows():
                    weight = comment_weight if row["type"] == "comment" else 1.0
                    key = row.get(field, "UNKNOWN")
                    counts[key] += weight
                return counts

            # Pie 1: Language distribution
            lang_counts = weighted_counter(filtered, "lang")
            # Group others
            grouped_counts = Counter()
            for lang, count in lang_counts.items():
                grouped_key = lang if lang in MAIN_LANGS else "other"
                grouped_counts[grouped_key] += count

            plot_pie_chart(f"Language Distribution (n = {len(filtered)})", list(grouped_counts.keys()), list(grouped_counts.values()))

            # Pies 2–6: Sentiment per language
            for lang in MAIN_LANGS + ["other"]:
                if lang == "other":
                    subset = filtered[(~filtered["lang"].isin(MAIN_LANGS)) & (filtered["lang"] != "unknown")]
                else:
                    subset = filtered[filtered["lang"] == lang]

                if not subset.empty:
                    sent_counts = weighted_counter(subset, "sentiment_majority")
                    plot_pie_chart(f"Sentiment in '{lang}' (n = {len(subset)})", list(sent_counts.keys()), list(sent_counts.values()))

        except Exception as e:
            messagebox.showerror("Error", str(e))

# --- Run App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()
