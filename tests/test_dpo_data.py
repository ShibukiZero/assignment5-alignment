from .adapters import run_load_hh_preference_data as load_hh_preference_data


def test_load_hh_preference_data_keeps_single_turn_examples(tmp_path):
    hh_path = tmp_path / "helpful-base.jsonl"
    hh_path.write_text(
        "\n".join(
            [
                (
                    '{"chosen":"\\n\\nHuman: How do I bake bread?\\n\\nAssistant: '
                    'Use flour, water, yeast, and salt.",'
                    '"rejected":"\\n\\nHuman: How do I bake bread?\\n\\nAssistant: '
                    'I will not answer."}'
                ),
                (
                    '{"chosen":"\\n\\nHuman: First?\\n\\nAssistant: Answer.'
                    '\\n\\nHuman: Follow up?\\n\\nAssistant: More.",'
                    '"rejected":"\\n\\nHuman: First?\\n\\nAssistant: Bad."}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_hh_preference_data(hh_path)

    assert records == [
        {
            "instruction": "How do I bake bread?",
            "chosen": "Use flour, water, yeast, and salt.",
            "rejected": "I will not answer.",
            "source_file": "helpful-base.jsonl",
            "source_path": str(hh_path),
            "line_number": 1,
            "split": "helpful",
        }
    ]


def test_load_hh_preference_data_discovers_hf_style_train_files(tmp_path):
    collection_dir = tmp_path / "hh_rlhf" / "harmless-base"
    collection_dir.mkdir(parents=True)
    (collection_dir / "train.jsonl").write_text(
        (
            '{"chosen":"\\n\\nHuman: Can you help?\\n\\nAssistant: Safe answer.",'
            '"rejected":"\\n\\nHuman: Can you help?\\n\\nAssistant: Unsafe answer."}'
            "\n"
        ),
        encoding="utf-8",
    )
    for collection in [
        "helpful-base",
        "helpful-online",
        "helpful-rejection-sampled",
    ]:
        empty_dir = tmp_path / "hh_rlhf" / collection
        empty_dir.mkdir()
        (empty_dir / "train.jsonl").write_text(
            (
                '{"chosen":"\\n\\nHuman: Prompt?\\n\\nAssistant: Chosen.",'
                '"rejected":"\\n\\nHuman: Prompt?\\n\\nAssistant: Rejected."}'
                "\n"
            ),
            encoding="utf-8",
        )

    records = load_hh_preference_data(tmp_path / "hh_rlhf", max_records=1)

    assert records[0]["source_path"].endswith("harmless-base/train.jsonl")
    assert records[0]["instruction"] == "Can you help?"
