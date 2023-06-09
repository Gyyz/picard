# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.

import json
import os
from typing import List, Generator, Any, Dict, Tuple
from third_party.spider.preprocess.get_tables import dump_db_json_schema
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """None"""

_DESCRIPTION = """None"""

class Gptsql(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="gptsql",
            version=VERSION,
            description="gptsql, a sql dataset generated by chatgpt",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs) -> None:
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "query": datasets.Value("string"),
                "question": datasets.Value("string"),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
                    }
                ),
                "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                "sant_schema_input": datasets.features.Value("string")
                "db_foreign_keys": datasets.features.Sequence(
                    {
                        "column_id": datasets.Value("int32"),
                        "other_column_id": datasets.Value("int32"),
                    }
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=None,
            license=None,
            citation=None,
        )

    def _split_generators(self) -> List[datasets.SplitGenerator]:
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepaths": ['../../gptsql/data_with_schema.json'],
                    "db_path": '../../gptsql/data_with_schema.json',
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepaths": ['../../gptsql/data_with_schema.json'],
                    "db_path": '../../gptsql/data_with_schema.json',
                },)
        ]

    def _generate_examples(
        self, data_filepaths: List[str], db_path: str
    ) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        """This function returns the examples in the raw (text) form."""
        for data_filepath in data_filepaths:
            logger.info("generating examples from = %s", data_filepath)
            with open(data_filepath, encoding="utf-8") as f:
                gptsql = json.loads(f)
                for idx, sample in enumerate(gptsql):
                    db_id = sample["db_name"]
                    schema = sample['schema']
                    yield idx, {
                        "query": sample["sql"],
                        "question": sample["text"],
                        "db_id": db_id,
                        "db_path": db_path,
                        "db_table_names": sample["table_names"],
                        "db_column_names": [
                            {"table_id": table_id, "column_name": column_name}
                            for table_id, column_name in zip(sample["col_tab_mapping"], sample["column_names"])
                        ],
                        "db_column_types": sample["col_types"],
                        "sant_schema_input":sample['schema_input']
                        "db_primary_keys": [],
                        "db_foreign_keys": [],
                    }
