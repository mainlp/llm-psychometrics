# %%
import json
import zipfile

# %%
with (
    open("output/cmcqrd-reading-items.jsonl", "w") as items_file,
    open(
        "output/cmcqrd-reading-response-distributions.jsonl", "w"
    ) as response_distributions_file,
    open("output/cmcqrd-parameters.jsonl", "w") as parameters_file,
):
    with zipfile.ZipFile(
        "Cambridge Multiple-Choice Questions Reading Dataset.zip", "r"
    ) as archive:
        lines = archive.read(
            "Cambridge Multiple-Choice Questions Reading Dataset.jsonl"
        ).splitlines()

        for line in lines:
            task = json.loads(line)
            passage = task["title"] + "\n" + task["text"]
            for question_id, question in task["questions"].items():
                level = task["level"]
                stem = question["text"]
                correct_option_index = "abcd".index(question["answer"])
                options = [question["options"][letter]["text"] for letter in "abcd"]
                response_distribution = [
                    question["options"][letter]["fac"] for letter in "abcd"
                ]
                if all(prob is None for prob in response_distribution):
                    # Skip items without response distribution
                    continue
                b = question["diff"]
                cmcqrd_id = f"{task['id']}-{question_id}"
                item_id = f"CMCQRD-{cmcqrd_id}"

                items_file.write(
                    json.dumps(
                        {
                            "item_id": item_id,
                            "passage": passage,
                            "stem": stem,
                            "options": options,
                            "correct_option_index": correct_option_index,
                        }
                    )
                    + "\n"
                )

                response_distributions_file.write(
                    json.dumps(
                        {
                            "item_id": item_id,
                            "level": level,
                            "response_distribution": response_distribution,
                        }
                    )
                    + "\n"
                )

                parameters_file.write(
                    json.dumps(
                        {
                            "subject": "reading",
                            "year": None,
                            "level": level,
                            "scale": None,
                            "item_id": item_id,
                            "b": b,
                            "irt_model": "1pl",
                        }
                    )
                    + "\n"
                )
