#!/usr/bin/env python3
from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Study Sentiment Pipeline Launcher")
        self.root.geometry("980x720")

        self.repo_root = Path(__file__).resolve().parent

        self.input_dir_var = tk.StringVar(value=str(self.repo_root / "data_input"))
        self.output_dir_var = tk.StringVar(value=str(self.repo_root / "data_output"))
        self.torch_var = tk.StringVar(value="cu121")
        self.status_var = tk.StringVar(value="Ready")

        self.running = False
        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()

        self._build_ui()
        self.root.after(80, self._drain_log_queue)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        paths_frame = ttk.LabelFrame(main, text="Folders", padding=10)
        paths_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(paths_frame, text="Input dataset folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(paths_frame, textvariable=self.input_dir_var, width=90).grid(row=1, column=0, padx=(0, 8), sticky="we")
        ttk.Button(paths_frame, text="Browse", command=self.pick_input_dir).grid(row=1, column=1, sticky="e")

        ttk.Label(paths_frame, text="Output dataset folder").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(paths_frame, textvariable=self.output_dir_var, width=90).grid(row=3, column=0, padx=(0, 8), sticky="we")
        ttk.Button(paths_frame, text="Browse", command=self.pick_output_dir).grid(row=3, column=1, sticky="e")

        paths_frame.columnconfigure(0, weight=1)

        setup_frame = ttk.LabelFrame(main, text="Setup", padding=10)
        setup_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(setup_frame, text="Torch").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            setup_frame,
            textvariable=self.torch_var,
            values=["cpu", "cu118", "cu121", "cu124"],
            state="readonly",
            width=10,
        ).grid(row=0, column=1, sticky="w", padx=(5, 20))

        ttk.Button(setup_frame, text="Install dependencies", command=self.install_deps).grid(row=0, column=2, padx=5)
        ttk.Button(setup_frame, text="Download models", command=self.download_models).grid(row=0, column=3, padx=5)

        pipeline_frame = ttk.LabelFrame(main, text="Pipelines", padding=10)
        pipeline_frame.pack(fill="x", pady=(0, 10))

        ttk.Button(pipeline_frame, text="1) Fetch Reddit", command=self.run_fetch_reddit).grid(row=0, column=0, padx=5, pady=5, sticky="we")
        ttk.Button(pipeline_frame, text="2) Process Reddit", command=self.run_process_reddit).grid(row=0, column=1, padx=5, pady=5, sticky="we")
        ttk.Button(pipeline_frame, text="3) Analyze Sentiment", command=self.run_analyze_sentiment).grid(row=0, column=2, padx=5, pady=5, sticky="we")
        ttk.Button(pipeline_frame, text="4) Analyze Topics", command=self.run_analyze_topics).grid(row=0, column=3, padx=5, pady=5, sticky="we")
        ttk.Button(pipeline_frame, text="Run all pipelines", command=self.run_all_pipelines).grid(row=1, column=0, columnspan=4, padx=5, pady=(8, 5), sticky="we")

        for i in range(4):
            pipeline_frame.columnconfigure(i, weight=1)

        tools_frame = ttk.LabelFrame(main, text="Tools", padding=10)
        tools_frame.pack(fill="x", pady=(0, 10))
        ttk.Button(tools_frame, text="Open sentiment visualizer", command=self.open_visualizer).pack(anchor="w")

        progress_frame = ttk.LabelFrame(main, text="Progress", padding=10)
        progress_frame.pack(fill="x", pady=(0, 10))

        self.progress = ttk.Progressbar(progress_frame, mode="indeterminate")
        self.progress.pack(fill="x")

        ttk.Label(progress_frame, textvariable=self.status_var).pack(anchor="w", pady=(6, 0))

        log_frame = ttk.LabelFrame(main, text="Logs", padding=10)
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_frame, wrap="word", height=20, state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        bottom = ttk.Frame(main)
        bottom.pack(fill="x", pady=(8, 0))
        ttk.Button(bottom, text="Clear logs", command=self.clear_logs).pack(side="left")

    def _append_log_text(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def log(self, msg: str) -> None:
        self.log_queue.put(("line", msg))

    def _drain_log_queue(self) -> None:
        try:
            while True:
                kind, msg = self.log_queue.get_nowait()
                if kind == "line":
                    self._append_log_text(msg + "\n")
        except queue.Empty:
            pass
        self.root.after(80, self._drain_log_queue)

    def clear_logs(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def pick_input_dir(self) -> None:
        initial = self.input_dir_var.get()
        if not Path(initial).exists():
            initial = str(self.repo_root / "data_input")
        chosen = filedialog.askdirectory(title="Choose input dataset folder", initialdir=initial)
        if chosen:
            self.input_dir_var.set(chosen)

    def pick_output_dir(self) -> None:
        initial = self.output_dir_var.get()
        if not Path(initial).exists():
            initial = str(self.repo_root / "data_output")
        chosen = filedialog.askdirectory(title="Choose output dataset folder", initialdir=initial)
        if chosen:
            self.output_dir_var.set(chosen)

    def _validate_dirs(self) -> tuple[Path, Path] | None:
        input_dir = Path(self.input_dir_var.get()).resolve()
        output_dir = Path(self.output_dir_var.get()).resolve()

        if not input_dir.exists():
            messagebox.showerror("Input folder missing", f"Input folder does not exist:\n{input_dir}")
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        return input_dir, output_dir

    def _script_path(self, *parts: str) -> Path:
        return self.repo_root.joinpath(*parts)

    def _find_script(self, candidates: list[str]) -> Path | None:
        for rel in candidates:
            p = self._script_path(*rel.split("/"))
            if p.exists():
                return p
        messagebox.showerror("Missing script", "Could not find any of these script paths:\n\n" + "\n".join(candidates))
        return None

    def _set_running_ui(self, running: bool, title: str = "") -> None:
        if running:
            self.status_var.set(f"Running: {title}")
            self.progress.start(12)
        else:
            self.progress.stop()
            self.status_var.set("Ready")

    def _run_commands_async(self, commands: list[list[str]], title: str) -> None:
        if self.running:
            messagebox.showwarning("Busy", "Another task is already running.")
            return

        self.running = True
        self._set_running_ui(True, title)
        self.log("")
        self.log(f"=== {title} ===")

        def worker() -> None:
            try:
                total = len(commands)
                for idx, cmd in enumerate(commands, start=1):
                    step_label = f"{title} ({idx}/{total})"
                    self.root.after(0, lambda s=step_label: self.status_var.set(f"Running: {s}"))

                    self.log("> " + " ".join(cmd))
                    process = subprocess.Popen(
                        cmd,
                        cwd=str(self.repo_root),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )

                    assert process.stdout is not None
                    for line in process.stdout:
                        self.log(line.rstrip())

                    rc = process.wait()
                    if rc != 0:
                        self.log(f"[ERROR] Command failed with exit code {rc}")
                        self.root.after(
                            0,
                            lambda c=cmd, code=rc: messagebox.showerror(
                                "Command failed",
                                f"Exit code {code}\n\n{' '.join(c)}",
                            ),
                        )
                        break
                else:
                    self.log("[OK] Finished")
            except Exception as e:
                self.log(f"[ERROR] {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.running = False
                self.root.after(0, lambda: self._set_running_ui(False))

        threading.Thread(target=worker, daemon=True).start()

    def install_deps(self) -> None:
        script = self._find_script(["config/install_deps.py"])
        if not script:
            return
        cmd = [sys.executable, str(script), "--torch", self.torch_var.get()]
        self._run_commands_async([cmd], "Install dependencies")

    def download_models(self) -> None:
        script = self._find_script(["config/download_all_models.py"])
        if not script:
            return
        cmd = [sys.executable, str(script)]
        self._run_commands_async([cmd], "Download all models")

    def run_fetch_reddit(self) -> None:
        dirs = self._validate_dirs()
        if not dirs:
            return
        input_dir, output_dir = dirs

        script = self._find_script([
            "reddit/reddit_fetch_posts_with_comments.py",
            "pipelines/reddit_fetch_posts_with_comments.py",
            "reddit_fetch_posts_with_comments.py",
        ])
        if not script:
            return

        cmd = [sys.executable, str(script), "--input-dir", str(input_dir), "--output-dir", str(output_dir)]
        self._run_commands_async([cmd], "Fetch Reddit posts")

    def run_process_reddit(self) -> None:
        dirs = self._validate_dirs()
        if not dirs:
            return
        input_dir, output_dir = dirs

        script = self._find_script([
            "pipelines/process_reddit_posts.py",
            "process_reddit_posts.py",
        ])
        if not script:
            return

        cmd = [sys.executable, str(script), "--input-dir", str(input_dir), "--output-dir", str(output_dir)]
        self._run_commands_async([cmd], "Process Reddit posts")

    def run_analyze_sentiment(self) -> None:
        dirs = self._validate_dirs()
        if not dirs:
            return
        input_dir, output_dir = dirs

        script = self._find_script([
            "pipelines/analyze_sentiment.py",
            "analyze_sentiment.py",
        ])
        if not script:
            return

        cmd = [sys.executable, str(script), "--input-dir", str(input_dir), "--output-dir", str(output_dir)]
        self._run_commands_async([cmd], "Analyze sentiment")

    def run_analyze_topics(self) -> None:
        dirs = self._validate_dirs()
        if not dirs:
            return
        input_dir, output_dir = dirs

        script = self._find_script([
            "pipelines/analyze_topics.py",
            "analyze_topics.py",
        ])
        if not script:
            return

        cmd = [sys.executable, str(script), "--input-dir", str(input_dir), "--output-dir", str(output_dir)]
        self._run_commands_async([cmd], "Analyze topics")

    def run_all_pipelines(self) -> None:
        dirs = self._validate_dirs()
        if not dirs:
            return
        input_dir, output_dir = dirs

        fetch_script = self._find_script([
            "reddit/reddit_fetch_posts_with_comments.py",
            "pipelines/reddit_fetch_posts_with_comments.py",
            "reddit_fetch_posts_with_comments.py",
        ])
        process_script = self._find_script([
            "pipelines/process_reddit_posts.py",
            "process_reddit_posts.py",
        ])
        sentiment_script = self._find_script([
            "pipelines/analyze_sentiment.py",
            "analyze_sentiment.py",
        ])
        topics_script = self._find_script([
            "pipelines/analyze_topics.py",
            "analyze_topics.py",
        ])

        if not all([fetch_script, process_script, sentiment_script, topics_script]):
            return

        commands = [
            [sys.executable, str(fetch_script), "--input-dir", str(input_dir), "--output-dir", str(output_dir)],
            [sys.executable, str(process_script), "--input-dir", str(input_dir), "--output-dir", str(output_dir)],
            [sys.executable, str(sentiment_script), "--input-dir", str(input_dir), "--output-dir", str(output_dir)],
            [sys.executable, str(topics_script), "--input-dir", str(input_dir), "--output-dir", str(output_dir)],
        ]
        self._run_commands_async(commands, "Run all pipelines")

    def open_visualizer(self) -> None:
        dirs = self._validate_dirs()
        if not dirs:
            return
        _, output_dir = dirs

        script = self._find_script([
            "tools/full_sentiment_visualizer.py",
            "full_sentiment_visualizer.py",
        ])
        if not script:
            return

        cmd = [sys.executable, str(script), "--output-dir", str(output_dir)]
        self._run_commands_async([cmd], "Open sentiment visualizer")


def main() -> None:
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()