"""
Measuring Massive Multitask Language Understanding
https://arxiv.org/pdf/2009.03300.pdf

The Hendryck's Test is a benchmark that measured a text model’s multitask accuracy.
The test covers 57 tasks including elementary mathematics, US history, computer
science, law, and more. To attain high accuracy on this test, models must possess
extensive world knowledge and problem solving ability. By comprehensively evaluating
the breadth and depth of a model’s academic and professional understanding,
Hendryck's Test can be used to analyze models across many tasks and to identify
important shortcomings.

Homepage: https://github.com/hendrycks/test
"""
import numpy as np
from datasets import concatenate_datasets, load_dataset

from lm_eval.base import MultipleChoiceTask, rf

_CITATION = """
@article{hendryckstest2021,
    title={Measuring Massive Multitask Language Understanding},
    author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2021}
}
"""

SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

CATEGORIES = ["stem", "humanities", "social", "others"]

SUBJECT_TO_CATEGORY = {
    "abstract_algebra": "stem",
    "astronomy": "stem",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_mathematics": "stem",
    "high_school_physics": "stem",
    "high_school_statistics": "stem",
    "machine_learning": "stem",
    "formal_logic": "humanities",
    "high_school_european_history": "humanities",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_law": "humanities",
    "world_religions": "humanities",
    "econometrics": "social",
    "high_school_geography": "social",
    "high_school_government_and_politics": "social",
    "high_school_macroeconomics": "social",
    "high_school_microeconomics": "social",
    "high_school_psychology": "social",
    "human_sexuality": "social",
    "professional_psychology": "social",
    "public_relations": "social",
    "security_studies": "social",
    "sociology": "social",
    "us_foreign_policy": "social",
    "anatomy": "others",
    "business_ethics": "others",
    "clinical_knowledge": "others",
    "college_medicine": "others",
    "global_facts": "others",
    "human_aging": "others",
    "management": "others",
    "marketing": "others",
    "medical_genetics": "others",
    "miscellaneous": "others",
    "nutrition": "others",
    "professional_accounting": "others",
    "professional_medicine": "others",
    "virology": "others",
}


class MMLUTemplate:
    def __init__(
        self,
        description,
        sample,
        choice,
        delimiter,
        choices_names,
        shuffle_choices=False,
        use_choice_text=False,
    ):
        self.description = description
        self.sample = sample
        self.choice = choice
        self.delimiter = delimiter
        assert len(choices_names) == 4
        self.choices_names = choices_names
        self.shuffle_choices = shuffle_choices
        self.use_choice_text = use_choice_text

    def render(self, doc, fewshot_docs=None):
        description = self.description.format(subject=self.format_subject(doc["subject"]))
        doc = self.prepare_choices(doc)
        doc["answer_choice"] = "{answer_choice}"
        sample = self.sample.format(**doc)
        fewshots = self.render_fewshots(fewshot_docs)
        return f"{description}{self.delimiter}{fewshots}{sample}", doc

    def render_fewshots(self, fewshot_docs):
        if fewshot_docs is None:
            return ""
        fewshot_samples = []
        for fewshot in fewshot_docs:
            fewshot = self.prepare_choices(fewshot)
            fewshot_sample = self.sample.format(**fewshot)
            fewshot_samples.append(fewshot_sample)
        return self.delimiter.join(fewshot_samples) + self.delimiter

    def prepare_choices(self, doc):
        choices = doc["choices"]
        answer = doc["answer"]
        if self.shuffle_choices:
            indexes = np.random.shuffle(list(range(len(choices))))
            choices = [choices[i] for i in indexes]
            answer = indexes.index(answer)
        for i in range(len(choices)):
            doc[f"choice{i}"] = self.choice.format(choice_name=self.choices_names[i], choice_text=choices[i])
        doc["possible_answers"] = choices if self.use_choice_text else self.choices_names
        doc["answer_choice"] = doc["possible_answers"][answer]
        doc["gold"] = answer
        return doc

    def format_subject(self, subject):
        words = subject.split("_")
        return " ".join(words)
        


class MultiMMLU(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "cais/mmlu"

    def __init__(self):
        super().__init__()
        self.templates = {
            "default": MMLUTemplate(
                description="The following are multiple choice questions (with answers) about {subject}.",
                sample="{question}\n{choice0}\n{choice1}\n{choice2}\n{choice3}\nAnswer: {answer_choice}",
                choice="{choice_name}. {choice_text}",
                delimiter="\n\n",
                choices_names=["A", "B", "C", "D"],
                shuffle_choices=False,
            )
        }

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.fewshot_docs = {}
        datasets = {"validation": [], "test": []}
        i = 0
        for subject in SUBJECTS:
            dataset = load_dataset(
                path=self.DATASET_PATH,
                name=subject,
                data_dir=data_dir,
                cache_dir=cache_dir,
                download_mode=download_mode,
            )
            self.fewshot_docs[subject] = dataset["dev"].add_column(
                "subject", [subject] * len(dataset["dev"])
            )
            for split in datasets.keys():
                split_dataset = dataset[split]
                split_dataset = split_dataset.add_column(
                    "subject", [subject] * len(split_dataset)
                )
                datasets[split].append(split_dataset)
            if i >= 0:
                break
            i += 1

        self.dataset = concatenate_datasets(datasets["test"])
        print("Test set size:", len(self.dataset))
        self.val_dataset = concatenate_datasets(datasets["validation"])
        print("Validation set size:", len(self.val_dataset))

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return map(self._process_doc, self.val_dataset)

    def test_docs(self):
        return map(self._process_doc, self.dataset)

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        template_id = "default"
        subject = doc["subject"]
        template = self.templates[template_id]
        fewshot_docs = list(self.fewshot_docs[subject])[:num_fewshot]
        text, doc = template.render(doc, fewshot_docs)
        return text, doc

    def construct_requests(self, _, ctx):
        text, updated_doc = ctx
        context, continuation = text.split("{answer_choice}", maxsplit=1)
        variants = [(context, answer_choice + continuation) for answer_choice in updated_doc["possible_answers"]]
        return [rf.loglikelihood(*v)[0] for v in variants]

    def process_results(self, doc, results):
        gold = doc["gold"]
        acc = 1 if np.argmax(results) == gold else 0
        category = SUBJECT_TO_CATEGORY[doc["subject"]]
        metrics = {"avg_acc": acc}
        for c in CATEGORIES:
            metrics[f"{c}_acc"] = acc if c == category else -1
        return metrics

    def aggregation(self):
        def calc_avg(items):
            values = [item for item in items if item != -1]
            return np.mean(values) if len(values) else 0.0

        metrics = {"avg_acc": calc_avg}
        for c in CATEGORIES:
            metrics[f"{c}_acc"] = calc_avg

        return metrics

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
