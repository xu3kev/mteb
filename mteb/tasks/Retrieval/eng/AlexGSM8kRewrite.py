from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks import AbsTaskRetrieval, TaskMetadata


class AlexGSM8kRewrite(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="AlexGSM8kRewrite",
        dataset={
            "path": "minimario/gsm8k-rewritten",
            "revision": "7f5ad21e2207d52b963bd82be2717627a8ae8061",
        },
        description="GSM8K rewrite",
        reference=None,
        type="Retrieval",
        category="p2p",
        eval_splits=["train"],
        eval_langs=["eng-Latn"],
        main_score="ncdg_at_10",
        date=("2022-01-01", "2024-12-31"),  # approximate guess
        form=["written"],
        domains=["Non-fiction"],
        task_subtypes=["Question answering"],
        license=None,
        socioeconomic_status="high",
        annotations_creators="human-annotated",
        dialect=None,
        text_creation="found",
        bibtex_citation="""""",
        n_samples={"validation": 100},
        avg_character_length=None,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        split = self.metadata_dict["eval_splits"][0]
        ds = load_dataset(**self.metadata_dict["dataset"], split=split)
        ds = ds.shuffle(seed=42)
        max_samples = min(2048, len(ds))
        ds = ds.select(
            range(max_samples)
        )  # limit the dataset size to make sure the task does not take too long to run
        question = ds["old_question"]
        similar_question = ds["reworded_new_question_2"]

        self.corpus = {split: {}}
        self.relevant_docs = {split: {}}
        self.queries = {split: {}}

        text2id = {}
        n = 0
        for q, sq in zip(question, similar_question):
            self.queries[split][str(n)] = q
            q_n = n
            n += 1
            if sq not in text2id:
                text2id[sq] = n
                self.corpus[split][str(n)] = {"title": "", "text": sq}
                n += 1

            self.relevant_docs[split][str(q_n)] = {
                str(text2id[sq]): 1,
            } 

        self.data_loaded = True
