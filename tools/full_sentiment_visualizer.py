import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import os

CSV_PATH = "data/final/final_posts.csv"
data = pd.read_csv(CSV_PATH)

MAIN_LANGS = ["de", "en", "fr", "it"]

def filter_data(df, allow_duplicates, allow_multiple_per_author, lang_order, sort_order, prioritize, include_comments):
    df = df[df["is_about_study"] == True]

    if not include_comments:
        df = df[df["type"] == "post"]

    if not allow_multiple_per_author:
        df = df.drop_duplicates(subset=["author"])

    if not allow_duplicates:
        df = df.drop_duplicates(subset=["title", "selftext"])

    if prioritize == "Recency":
        df = df.sort_values("created_utc", ascending=(sort_order == "Oldest first"))
    else:
        df["lang_priority"] = df["lang"].apply(lambda x: lang_order.index(x) if x in lang_order else len(lang_order))
        df = df.sort_values("lang_priority")

    return df

def plot_pie_chart(title, labels, sizes):
    plt.figure()
    plt.title(title)
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.axis("equal")
    plt.show()

def plot_stacked_bar(title, breakdown):
    """Draws a stacked bar chart for aspect × sentiment for one language."""
    aspects = list(breakdown.keys())
    sentiments = set()
    for asp in breakdown.values():
        sentiments.update(asp.keys())
    sentiments = sorted(sentiments)

    # Prepare data
    values = {s: [breakdown[a].get(s, 0) for a in aspects] for s in sentiments}

    plt.figure(figsize=(10, 6))
    bottom = [0] * len(aspects)
    for s in sentiments:
        plt.bar(aspects, values[s], label=s, bottom=bottom)
        bottom = [b + v for b, v in zip(bottom, values[s])]

    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.show()

class FullSentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Full Sentiment Dashboard")

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

        tk.Button(root, text="Generate Charts", command=self.run_analysis).pack(pady=10)

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

            comment_weight = 0.5 if self.comment_weight.get() == "Lower" else (1.5 if self.comment_weight.get() == "Higher" else 1.0)

            def weighted_counter(df, field):
                counts = Counter()
                for _, row in df.iterrows():
                    weight = comment_weight if row["type"] == "comment" else 1.0
                    counts[row.get(field, "UNKNOWN")] += weight
                return counts

            # === Existing Charts ===
            lang_counts = weighted_counter(filtered, "lang")
            grouped_counts = Counter()
            for lang, count in lang_counts.items():
                grouped_counts[lang if lang in MAIN_LANGS else "other"] += count
            plot_pie_chart(f"Language Distribution (n={len(filtered)})", grouped_counts.keys(), grouped_counts.values())

            for lang in MAIN_LANGS + ["other"]:
                subset = filtered[(filtered["lang"] == lang)] if lang != "other" else filtered[(~filtered["lang"].isin(MAIN_LANGS)) & (filtered["lang"] != "unknown")]
                if not subset.empty:
                    sent_counts = weighted_counter(subset, "sentiment_majority")
                    plot_pie_chart(f"Sentiment in {lang} (n={len(subset)})", sent_counts.keys(), sent_counts.values())

            # === New Pie Charts ===
            reason_counts = weighted_counter(filtered, "main_reason")
            plot_pie_chart(f"Main Reason Mentioned (n={len(filtered)})", reason_counts.keys(), reason_counts.values())

            degree_counts = weighted_counter(filtered, "degree_type")
            plot_pie_chart(f"Degree Type Mentioned (n={len(filtered)})", degree_counts.keys(), degree_counts.values())

            # === New Cross-Stats: Sentiment × Aspect × Language ===
            breakdown = defaultdict(lambda: defaultdict(lambda: Counter()))
            for _, row in filtered.iterrows():
                lang = row.get("lang", "unknown")
                lang = lang if lang in MAIN_LANGS else "other"
                aspect = row.get("main_reason", "unknown")
                sent = row.get("sentiment_majority", "UNKNOWN")
                weight = comment_weight if row["type"] == "comment" else 1.0
                breakdown[lang][aspect][sent] += weight

            # Create stacked bar per language
            for lang, aspects in breakdown.items():
                plot_stacked_bar(f"Aspect × Sentiment for {lang}", aspects)

            # Global stacked bar (all languages combined)
            global_aspects = defaultdict(Counter)
            for lang, aspects in breakdown.items():
                for asp, sents in aspects.items():
                    for sent, val in sents.items():
                        global_aspects[asp][sent] += val
            plot_stacked_bar("Aspect × Sentiment (All Languages Combined)", global_aspects)

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = FullSentimentApp(root)
    root.mainloop()
