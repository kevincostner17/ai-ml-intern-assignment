import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PIPELINE_VERSION = "1.2"

SPEAKER_PATTERN = re.compile(r"(User\s*\d+)\s*:\s*")


@dataclass
class Message:
    global_msg_id: int
    day_id: int
    row_order: int
    intra_row_order: int
    speaker: str
    text: str


def _clean_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip().strip('"').strip()


def parse_row_to_messages(row_text: str) -> List[Tuple[str, str]]:
    text = _clean_text(row_text)
    if not text:
        return []

    matches = list(SPEAKER_PATTERN.finditer(text))
    if not matches:
        return [("Unknown", text)]

    parsed: List[Tuple[str, str]] = []
    for i, match in enumerate(matches):
        speaker = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        message = text[start:end].strip()
        if message:
            parsed.append((speaker, message))
    return parsed


class ConversationRAGSystem:
    def __init__(self, chunk_size: int = 8, chunk_overlap: int = 2, persona_speaker: str = "User 1"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persona_speaker = persona_speaker
        self.messages_df: Optional[pd.DataFrame] = None
        self.topic_checkpoints: List[Dict] = []
        self.hundred_checkpoints: List[Dict] = []
        self.chunks: List[Dict] = []
        self.persona: Dict = {}

        self.message_vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
        )
        self.summary_vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=15000,
        )
        self.message_matrix = None
        self.summary_matrix = None
        self.summary_index: List[Dict] = []
        self.first_person_pattern = re.compile(
            r"\b(i am|i'm|i was|i have|i've|i like|i love|i usually|i often|i work|my)\b",
            re.I,
        )

    def load_csv(self, csv_path: Union[str, Path]) -> pd.DataFrame:
        df = pd.read_csv(csv_path, header=None, names=["conversation"], dtype=str)
        records: List[Message] = []
        gid = 1
        for row_idx, row in enumerate(df["conversation"].fillna("").tolist(), start=1):
            row_messages = parse_row_to_messages(row)
            for m_idx, (speaker, text) in enumerate(row_messages, start=1):
                records.append(
                    Message(
                        global_msg_id=gid,
                        day_id=row_idx,
                        row_order=row_idx,
                        intra_row_order=m_idx,
                        speaker=speaker,
                        text=text.strip(),
                    )
                )
                gid += 1
        self.messages_df = pd.DataFrame([m.__dict__ for m in records])
        return self.messages_df

    def _segment_summary(self, segment_df: pd.DataFrame, max_points: int = 4) -> Tuple[str, List[str]]:
        texts = segment_df["text"].tolist()
        if not texts:
            return "", []

        seg_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=2000)
        matrix = seg_vec.fit_transform(texts)
        scores = np.asarray(matrix.sum(axis=1)).ravel()
        top_idx = np.argsort(scores)[::-1][:max_points]
        top_idx = sorted(top_idx.tolist())
        points = [texts[i] for i in top_idx]
        summary = " | ".join(points)
        return summary, points

    def detect_topic_checkpoints(
        self,
        min_topic_len: int = 12,
        compare_window: int = 6,
        similarity_threshold: float = 0.22,
        cooldown: int = 6,
    ) -> List[Dict]:
        if self.messages_df is None or self.messages_df.empty:
            raise ValueError("Load messages first.")

        texts = self.messages_df["text"].tolist()
        msg_ids = self.messages_df["global_msg_id"].tolist()
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=30000)
        X = tfidf.fit_transform(texts)

        boundaries = [0]
        last_split = 0
        i = min_topic_len
        n = len(texts)

        while i < n - compare_window:
            left_start = max(last_split, i - compare_window)
            left = np.asarray(X[left_start:i].mean(axis=0))
            right = np.asarray(X[i : i + compare_window].mean(axis=0))
            sim = cosine_similarity(left, right)[0, 0]
            topic_len = i - last_split
            if sim < similarity_threshold and topic_len >= min_topic_len and (i - last_split) >= cooldown:
                boundaries.append(i)
                last_split = i
                i += cooldown
            else:
                i += 1
        boundaries.append(n)

        checkpoints: List[Dict] = []
        for t_id, (start_i, end_i) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
            segment_df = self.messages_df.iloc[start_i:end_i]
            if segment_df.empty:
                continue
            summary, points = self._segment_summary(segment_df, max_points=4)
            checkpoints.append(
                {
                    "topic_id": t_id,
                    "start_msg_id": int(segment_df["global_msg_id"].iloc[0]),
                    "end_msg_id": int(segment_df["global_msg_id"].iloc[-1]),
                    "message_count": int(len(segment_df)),
                    "summary": summary,
                    "highlights": points,
                    "evidence_message_ids": segment_df["global_msg_id"].head(6).astype(int).tolist(),
                }
            )
        self.topic_checkpoints = checkpoints
        return checkpoints

    def build_hundred_checkpoints(self) -> List[Dict]:
        if self.messages_df is None or self.messages_df.empty:
            raise ValueError("Load messages first.")

        checkpoints: List[Dict] = []
        total = len(self.messages_df)
        checkpoint_id = 1
        for start in range(0, total, 100):
            end = min(start + 100, total)
            window_df = self.messages_df.iloc[start:end]
            summary, points = self._segment_summary(window_df, max_points=4)
            checkpoints.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "start_msg_id": int(window_df["global_msg_id"].iloc[0]),
                    "end_msg_id": int(window_df["global_msg_id"].iloc[-1]),
                    "message_count": int(len(window_df)),
                    "summary": summary,
                    "salient_points": points,
                }
            )
            checkpoint_id += 1
        self.hundred_checkpoints = checkpoints
        return checkpoints

    def build_chunks(self) -> List[Dict]:
        if self.messages_df is None or self.messages_df.empty:
            raise ValueError("Load messages first.")

        items = self.messages_df.to_dict("records")
        chunks: List[Dict] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunk_id = 1
        for i in range(0, len(items), step):
            part = items[i : i + self.chunk_size]
            if not part:
                continue
            text = "\n".join([f"{p['speaker']}: {p['text']}" for p in part])
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "start_msg_id": int(part[0]["global_msg_id"]),
                    "end_msg_id": int(part[-1]["global_msg_id"]),
                    "text": text,
                }
            )
            chunk_id += 1
            if i + self.chunk_size >= len(items):
                break
        self.chunks = chunks
        return chunks

    def build_indices(self) -> None:
        if self.messages_df is None or self.messages_df.empty:
            raise ValueError("Load messages first.")
        if not self.chunks:
            self.build_chunks()

        self.message_matrix = self.message_vectorizer.fit_transform([c["text"] for c in self.chunks])
        self.summary_index = []
        for t in self.topic_checkpoints:
            self.summary_index.append(
                {
                    "kind": "topic",
                    "id": t["topic_id"],
                    "start_msg_id": t["start_msg_id"],
                    "end_msg_id": t["end_msg_id"],
                    "text": t["summary"],
                }
            )
        for h in self.hundred_checkpoints:
            self.summary_index.append(
                {
                    "kind": "hundred",
                    "id": h["checkpoint_id"],
                    "start_msg_id": h["start_msg_id"],
                    "end_msg_id": h["end_msg_id"],
                    "text": h["summary"],
                }
            )
        if self.summary_index:
            self.summary_matrix = self.summary_vectorizer.fit_transform([s["text"] for s in self.summary_index])

    def retrieve(self, query: str, k_summaries: int = 5, k_chunks: int = 6) -> Dict:
        if self.message_matrix is None:
            self.build_indices()

        q_msg = self.message_vectorizer.transform([query])
        msg_scores = cosine_similarity(q_msg, self.message_matrix)[0]
        msg_top = np.argsort(msg_scores)[::-1][:k_chunks]
        chunk_hits = [
            {
                **self.chunks[i],
                "score": float(msg_scores[i]),
            }
            for i in msg_top
        ]

        summary_hits: List[Dict] = []
        if self.summary_matrix is not None and len(self.summary_index) > 0:
            q_sum = self.summary_vectorizer.transform([query])
            sum_scores = cosine_similarity(q_sum, self.summary_matrix)[0]
            sum_top = np.argsort(sum_scores)[::-1][:k_summaries]
            summary_hits = [{**self.summary_index[i], "score": float(sum_scores[i])} for i in sum_top]

        return {
            "query": query,
            "summary_hits": summary_hits,
            "chunk_hits": chunk_hits,
        }

    def answer_query(self, query: str, k_summaries: int = 5, k_chunks: int = 6) -> Dict:
        retrieval = self.retrieve(query, k_summaries=k_summaries, k_chunks=k_chunks)
        intent = self._classify_query_intent(query)
        top_summaries = retrieval["summary_hits"][:3]
        top_chunks = retrieval["chunk_hits"][:3]

        if intent == "persona":
            speaker_tag = f"{self.persona_speaker}:"
            top_chunks = [
                c
                for c in retrieval["chunk_hits"]
                if speaker_tag in c["text"] and self.first_person_pattern.search(c["text"])
            ][:3]
            retrieval["answer"] = self._build_persona_answer(query, top_chunks, top_summaries)
            retrieval["intent"] = intent
            return retrieval

        summary_lines = [
            f"{h['kind']}#{h['id']} ({h['start_msg_id']}-{h['end_msg_id']}): {h['text'][:220]}"
            for h in top_summaries
        ]
        chunk_lines = [f"chunk {c['start_msg_id']}-{c['end_msg_id']}: {c['text'][:260]}" for c in top_chunks]
        answer = (
            "Grounded response using retrieved checkpoints and message chunks.\n\n"
            "Relevant checkpoint summaries:\n- "
            + "\n- ".join(summary_lines if summary_lines else ["No summary hit found"])
            + "\n\nRelevant message chunks:\n- "
            + "\n- ".join(chunk_lines if chunk_lines else ["No chunk hit found"])
        )
        retrieval["answer"] = answer
        retrieval["intent"] = intent
        return retrieval

    def _classify_query_intent(self, query: str) -> str:
        q = query.lower()
        persona_terms = [
            "habit",
            "person",
            "personality",
            "trait",
            "talk",
            "style",
            "communicat",
            "who is",
            "what kind of person",
        ]
        return "persona" if any(term in q for term in persona_terms) else "general"

    def _persona_category_for_query(self, query: str) -> List[str]:
        q = query.lower()
        categories: List[str] = []
        if "habit" in q:
            categories.append("habits")
        if "talk" in q or "style" in q or "communicat" in q:
            categories.append("communication_style")
        if "trait" in q or "personality" in q:
            categories.append("personality_traits")
        if "person" in q or "who is" in q:
            categories.extend(["personality_traits", "habits", "personal_facts", "communication_style"])
        if not categories:
            categories = ["personal_facts", "personality_traits", "habits", "communication_style"]
        # preserve order and remove duplicates
        return list(dict.fromkeys(categories))

    def _build_persona_answer(self, query: str, top_chunks: List[Dict], top_summaries: List[Dict]) -> str:
        categories = self._persona_category_for_query(query)
        lines: List[str] = []
        lines.append(
            f"Persona-focused response for {self.persona_speaker} using persona JSON plus supporting retrieval evidence.\n"
        )

        for category in categories:
            values = self.persona.get(category, [])
            title = category.replace("_", " ").title()
            if not values:
                lines.append(f"{title}: no strong evidence found.")
                continue
            if category == "communication_style":
                style_parts = []
                for key in ("avg_words", "punctuation_rate", "emoji_rate", "style_label"):
                    if key in values:
                        style_parts.append(f"- {key.replace('_', ' ')}: {values[key]}")
                if values.get("sample_messages"):
                    for s in values["sample_messages"][:2]:
                        style_parts.append(f"- sample (msg {s['message_id']}): {s['text']}")
                lines.append(f"{title}:\n" + "\n".join(style_parts))
            else:
                claims = [f"- {v['claim']} (msg {v['message_id']})" for v in values[:2]]
                lines.append(f"{title}:\n" + "\n".join(claims))

        if top_chunks:
            chunk_lines = [f"- msg {c['start_msg_id']}-{c['end_msg_id']}: {c['text'][:180]}" for c in top_chunks[:1]]
            lines.append("\nSupporting message chunks:\n" + "\n".join(chunk_lines))
        return "\n\n".join(lines)

    def extract_persona(self, max_items: int = 20) -> Dict:
        if self.messages_df is None or self.messages_df.empty:
            raise ValueError("Load messages first.")

        user_df = self.messages_df[self.messages_df["speaker"] == self.persona_speaker].copy()
        if user_df.empty:
            user_df = self.messages_df.copy()

        patterns = {
            "habits": re.compile(
                r"\b(i (usually|often|always|every day|daily|wake up|sleep|eat|drink|exercise|work out|read|cook|watch|listen|play)|i love to|i like to)\b",
                re.I,
            ),
            "personal_facts": re.compile(
                r"\b(i (work|study|live|moved|moving|am from|have|raised)|my (family|kids|parents|spouse|job)|single mom|single parent)\b",
                re.I,
            ),
            "personality_traits": re.compile(r"\b(excited|nervous|happy|funny|serious|emotional|positive|kind|calm)\b", re.I),
        }

        persona = {
            "habits": [],
            "personal_facts": [],
            "personality_traits": [],
            "communication_style": {},
            "meta": {
                "method": "pattern+evidence extraction from chronological conversations",
                "only_evidence_based": True,
                "pipeline_version": PIPELINE_VERSION,
                "persona_speaker": self.persona_speaker,
            },
        }

        for _, row in user_df.iterrows():
            text = str(row["text"]).strip()
            if not text:
                continue
            for key, rgx in patterns.items():
                if rgx.search(text):
                    persona[key].append(
                        {
                            "claim": text[:220],
                            "message_id": int(row["global_msg_id"]),
                            "speaker": row["speaker"],
                            "confidence": 0.65,
                        }
                    )

        # Communication style as aggregate metrics + examples.
        texts = user_df["text"].astype(str).tolist()
        word_counts = [len(t.split()) for t in texts if t.strip()]
        avg_words = round(float(np.mean(word_counts)), 2) if word_counts else 0.0
        punct_rate = round(float(np.mean([1.0 if re.search(r"[!?]", t) else 0.0 for t in texts])), 3) if texts else 0.0
        emoji_rate = round(
            float(np.mean([1.0 if re.search(r"(🙂|😊|😂|🤣|❤️|lol)", t, flags=re.I) else 0.0 for t in texts])), 3
        ) if texts else 0.0

        style_label = "balanced"
        if avg_words <= 8:
            style_label = "short and concise"
        elif avg_words >= 18:
            style_label = "long-form and detailed"

        style_samples = []
        for _, row in user_df.head(500).iterrows():
            t = str(row["text"]).strip()
            if re.search(r"[!?]|🙂|😊|😂|🤣|❤️|lol", t, flags=re.I):
                style_samples.append({"message_id": int(row["global_msg_id"]), "text": t[:140]})
            if len(style_samples) >= 6:
                break
        persona["communication_style"] = {
            "avg_words": avg_words,
            "punctuation_rate": punct_rate,
            "emoji_rate": emoji_rate,
            "style_label": style_label,
            "sample_messages": style_samples,
        }

        for key in ("habits", "personal_facts", "personality_traits"):
            persona[key] = persona[key][:max_items]
        self.persona = persona
        return persona

    def save_artifacts(self, out_dir: Union[str, Path] = "artifacts") -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.messages_df is not None:
            self.messages_df.to_parquet(out / "messages.parquet", index=False)
            self.messages_df.to_csv(out / "messages.csv", index=False)
        (out / "topic_checkpoints.json").write_text(json.dumps(self.topic_checkpoints, indent=2), encoding="utf-8")
        (out / "message_checkpoints_100.json").write_text(json.dumps(self.hundred_checkpoints, indent=2), encoding="utf-8")
        (out / "message_chunks.json").write_text(json.dumps(self.chunks, indent=2), encoding="utf-8")
        (out / "persona.json").write_text(json.dumps(self.persona, indent=2), encoding="utf-8")

    def run_pipeline(self, csv_path: Union[str, Path], out_dir: Union[str, Path] = "artifacts") -> Dict:
        self.load_csv(csv_path)
        self.detect_topic_checkpoints()
        self.build_hundred_checkpoints()
        self.build_chunks()
        self.build_indices()
        self.extract_persona()
        self.save_artifacts(out_dir)
        return {
            "messages": 0 if self.messages_df is None else int(len(self.messages_df)),
            "topics": int(len(self.topic_checkpoints)),
            "hundred_checkpoints": int(len(self.hundred_checkpoints)),
            "chunks": int(len(self.chunks)),
        }

