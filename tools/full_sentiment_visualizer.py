import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk

MAIN_LANGS = ["de", "en", "fr", "it"]


def filter_data(df, allow_duplicates, allow_multiple_per_author, lang_order, sort_order, prioritize, source_mode):
    df = df[df["is_about_study"] == True]

    if source_mode == "Posts":
        df = df[df["type"] == "post"]
    elif source_mode == "Comments":
        df = df[df["type"] == "comment"]

    if not allow_multiple_per_author:
        df = df.drop_duplicates(subset=["author"], keep="first")

    if not allow_duplicates:
        df = df.drop_duplicates(subset=["title", "selftext"], keep="first")

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
    aspects = list(breakdown.keys())
    sentiments = set()
    for asp in breakdown.values():
        sentiments.update(asp.keys())
    sentiments = sorted(sentiments)

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
    def __init__(self, root, csv_path: Path):
        self.root = root
        self.root.title("Full Sentiment Dashboard")
        self.csv_path = csv_path

        if not self.csv_path.exists():
            messagebox.showerror("Missing file", f"Could not find:\n{self.csv_path}")
            self.root.after(100, self.root.destroy)
            return

        self.data = pd.read_csv(self.csv_path)

        self.allow_dupes = tk.BooleanVar(value=True)
        self.allow_multi_author = tk.BooleanVar(value=True)

        tk.Checkbutton(root, text="Allow duplicate posts", variable=self.allow_dupes).pack(anchor="w")
        tk.Checkbutton(root, text="Allow multiple posts from same author", variable=self.allow_multi_author).pack(anchor="w")

        tk.Label(root, text="Data source:").pack(anchor="w")
        self.source_mode = ttk.Combobox(root, values=["Posts and comments", "Posts", "Comments"], state="readonly")
        self.source_mode.current(0)
        self.source_mode.pack(fill="x")

        tk.Label(root, text="Comment Importance:").pack(anchor="w")
        self.comment_weight = ttk.Combobox(root, values=["Lower", "Equal", "Higher"], state="readonly")
        self.comment_weight.current(1)
        self.comment_weight.pack(fill="x")

        tk.Label(root, text="Language Preference Order (comma-separated ISO)").pack(anchor="w")
        self.lang_entry = tk.Entry(root)
        self.lang_entry.insert(0, "de,en,fr,it")
        self.lang_entry.pack(fill="x")

        tk.Label(root, text="Sort by:").pack(anchor="w")
        self.sort_by = ttk.Combobox(root, values=["Newest first", "Oldest first"], state="readonly")
        self.sort_by.current(0)
        self.sort_by.pack(fill="x")

        tk.Label(root, text="Prioritize:").pack(anchor="w")
        self.priority = ttk.Combobox(root, values=["Recency", "Language preference"], state="readonly")
        self.priority.current(0)
        self.priority.pack(fill="x")

        tk.Button(root, text="Generate Charts", command=self.run_analysis).pack(pady=10)

    def run_analysis(self):
        try:
            lang_order = [l.strip() for l in self.lang_entry.get().split(",") if l.strip()]
            filtered = filter_data(
                self.data.copy(),
                allow_duplicates=self.allow_dupes.get(),
                allow_multiple_per_author=self.allow_multi_author.get(),
                lang_order=lang_order,
                sort_order=self.sort_by.get(),
                prioritize=self.priority.get(),
                source_mode=self.source_mode.get()
            )

            if filtered.empty:
                messagebox.showwarning("No Data", "No posts match the selected filters.")
                return

            mode = self.source_mode.get()
            if self.comment_weight.get() == "Lower":
                comment_weight = 0.5
            elif self.comment_weight.get() == "Higher":
                comment_weight = 1.5
            else:
                comment_weight = 1.0

            def weighted_counter(df, field):
                counts = Counter()
                for _, row in df.iterrows():
                    weight = 1.0 if mode in ("Posts", "Comments") else (comment_weight if row["type"] == "comment" else 1.0)
                    counts[row.get(field, "UNKNOWN")] += weight
                return counts

            lang_counts = weighted_counter(filtered, "lang")
            grouped_counts = Counter()
            for lang, count in lang_counts.items():
                grouped_counts[lang if lang in MAIN_LANGS else "other"] += count
            plot_pie_chart(f"Language Distribution (n={len(filtered)})", list(grouped_counts.keys()), list(grouped_counts.values()))

            for lang in MAIN_LANGS + ["other"]:
                if lang == "other":
                    subset = filtered[(~filtered["lang"].isin(MAIN_LANGS)) & (filtered["lang"] != "unknown")]
                else:
                    subset = filtered[filtered["lang"] == lang]
                if not subset.empty:
                    sent_counts = weighted_counter(subset, "sentiment_majority")
                    plot_pie_chart(f"Sentiment in {lang} (n={len(subset)})", list(sent_counts.keys()), list(sent_counts.values()))

            reason_counts = weighted_counter(filtered, "main_aspect")
            plot_pie_chart(f"Main Aspect Mentioned (n={len(filtered)})", list(reason_counts.keys()), list(reason_counts.values()))

            degree_counts = weighted_counter(filtered, "degree_type")
            plot_pie_chart(f"Degree Type Mentioned (n={len(filtered)})", list(degree_counts.keys()), list(degree_counts.values()))

            breakdown = defaultdict(lambda: defaultdict(lambda: Counter()))
            for _, row in filtered.iterrows():
                lang = row.get("lang", "unknown")
                lang = lang if lang in MAIN_LANGS else "other"
                aspect = row.get("main_aspect", "unknown")
                sent = row.get("sentiment_majority", "UNKNOWN")
                w = 1.0 if mode in ("Posts", "Comments") else (comment_weight if row["type"] == "comment" else 1.0)
                breakdown[lang][aspect][sent] += w

            for lang, aspects in breakdown.items():
                plot_stacked_bar(f"Aspect × Sentiment for {lang}", aspects)

            global_aspects = defaultdict(Counter)
            for _, aspects in breakdown.items():
                for asp, sents in aspects.items():
                    for sent, val in sents.items():
                        global_aspects[asp][sent] += val
            plot_stacked_bar("Aspect × Sentiment (All Languages Combined)", global_aspects)

        except Exception as e:
            messagebox.showerror("Error", str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    csv_path = output_dir / "final" / "final_posts.csv"

    root = tk.Tk()
    app = FullSentimentApp(root, csv_path)
    root.mainloop()


if __name__ == "__main__":
    main()