import json
import re
import string
import sys

import bs4
import requests
from tqdm import tqdm


BAD_ITEMS = [
    "H054401",  # Map not sufficiently described in alt text
    "H054501",  # Map not sufficiently described in alt text
    "H055001",  # Map not sufficiently described in alt text
    "H058401",  # Chart not sufficiently described in alt text
    "H058501",  # Chart not sufficiently described in alt text
    "H032401",  # Photo not sufficiently described in alt text
    "H037501",  # Ad text not available in alt text
    "H040801",  # Text not available due to copyright
    "H040802",  # Text not available due to copyright
    "H040803",  # Text not available due to copyright
]


def is_garbage(text: str) -> bool:
    # Page number
    if re.match(r"^Page \d+$", text):
        return True
    # Line number
    if re.match(r"^\d+$", text):
        return True
    # Copyright notice
    if (
        "©" in text
        or "copyright" in text.lower()
        or "by permission" in text.lower()
        or "permission granted" in text.lower()
        or text.startswith("Photos: ")
    ):
        return True
    return False


def get_texts(element: bs4.Tag, remove_garbage: bool = True) -> tuple[list[str], bool]:
    if "attrib" in element.get("class", []):
        return [], False
    is_paragraph = False
    if "paragraph" in element.get("class", []):
        is_paragraph = True
    texts = []
    for child in element.children:
        if isinstance(child, str):
            child = child.strip()
            if not remove_garbage or not is_garbage(child):
                if len(texts) == 0:
                    texts.append(child.strip())
                else:
                    texts[-1] += " " + child.strip()

        elif child.name == "img":
            alt = child.get("alt", "").strip()
            if alt != "" and "_image" not in alt:
                texts.append(f"[IMAGE: {alt}]")

        else:
            child_texts, child_is_paragraph = get_texts(child, remove_garbage)
            if len(child_texts) == 0:
                continue
            if child_is_paragraph:
                is_paragraph = True
                texts.extend(child_texts)
            else:
                if len(texts) == 0:
                    texts.extend(child_texts)
                else:
                    texts[-1] += " " + child_texts[0]
                    texts.extend(child_texts[1:])

    texts = [text.strip() for text in texts if text.strip() != ""]
    if remove_garbage:
        texts = [text for text in texts if not is_garbage(text)]
    # # Hack to re-merge paragraphs at column breaks
    # # Breaks poems
    # if len(texts) > 0 and texts[0][0].islower() and not texts[0].startswith("by "):
    #     is_paragraph = False

    return texts, is_paragraph


def scrape_item(table_id: int, has_passage: bool) -> dict | None:
    response = requests.get(
        "https://www.nationsreportcard.gov/nqt/api/queryresults/GetItem",
        params={"tableID": table_id},
    ).json()
    html = response["itemHTML"]

    if re.search(
        r"Passage Description|Description of ([Pp]assage|[Ss]tory|[Pp]oem)|\bPDF\b",
        html,
    ):
        # Full text is not (easily) available
        return None

    if "Type: MC" not in response["titleLine2"]:
        # Not a multiple choice item
        return None

    soup = bs4.BeautifulSoup(html, "html.parser")

    # Stem and options
    options = None
    if has_passage:
        stem = soup.select_one(".questionwrapper > .question > .stimulus + .paragraph")
        if stem is None:
            stem = list(soup.select_one(".questionwrapper > .question").children)[-1]
    else:
        stem = soup.select_one(
            ".questionwrapper > .question > .stdaln_img"
        ) or soup.select_one(".questionwrapper > .question")
    if isinstance(stem, str):
        pass
    elif "stdaln_img" in stem.get("class", []):
        item_text = stem.find("img")["alt"].strip()
        stem_match = re.match(
            r"([\s\S]*)(?:The image shows this question|Question [Tt]ext): (.*)",
            item_text,
        )
        assert stem_match is not None, item_text
        stem = ""
        if stem_match.group(1).strip() != "":
            stem += stem_match.group(1).strip() + "\n"
        stem += stem_match.group(2).strip()
        options_match = re.search(
            r"(?:The following answer options are shown|Answer [Oo]ptions): (.+)",
            item_text,
        )
        assert options_match is not None, item_text
        options = options_match.group(1)
        options = re.findall(r"\([ABCD]\)\s*([^;]+)", options)
        if len(options) == 0:
            options = re.findall(r"[ABCD][.:]\s*([^;.]+)", item_text)
        if len(options) == 0:
            options = re.findall(r"\b[ABCD]\s+([^;.]+)", item_text)
        assert len(stem) > 0, item_text
        assert len(options) == 4, item_text
    else:
        texts, _ = get_texts(stem, remove_garbage=False)
        stem = ""
        for text in texts:
            text = re.sub(r"[\n\r]+", "\n", text).strip()
            stem += text + "\n"
    if options is None:
        options = soup.select(".questionwrapper > .distractors > div > .itemtext")
        options = [option.text for option in options]
        assert len(options) == 4, options

    # Clean up
    stem = re.sub(
        r"^Questions? (?:\d+ (?:and \d+ )?)?(?:refers? to|is about|is based on) the.+?\.",
        "",
        stem,
    )
    stem = re.sub(r"\s*[\n\r]+\s*", "\n", stem)
    stem = re.sub(r"[ \xa0]+", " ", stem)
    stem = stem.strip()
    assert stem != "", table_id
    options = [re.sub(r"[ \xa0]+", " ", option) for option in options]
    options = [option.strip() for option in options]

    if not has_passage:
        return {
            "stem": stem,
            "options": options,
        }

    # Passage
    paragraphs = soup.select(".questionwrapper > .question > .stimulus > .paragraph")
    if len(paragraphs) == 0:
        paragraphs = soup.select(
            ".questionwrapper > .question > .stimulus > .stdaln_img"
        )
    if len(paragraphs) == 0:
        paragraphs = soup.select(".questionwrapper > .question > .paragraph")
        assert len(paragraphs) == 2
        paragraphs = [paragraphs[0]]
    passage = ""
    for paragraph in paragraphs:
        texts, _ = get_texts(paragraph)
        for text in texts:
            text = re.sub(r"\s*[\n\r]+\s*", "\n", text).strip()
            passage += text + "\n"

    # Clean up
    passage = passage.replace("&amp;amp;", "&")
    passage = re.sub(r"[ \xa0]+", " ", passage)
    passage = passage.strip()

    return {
        "passage": passage,
        "stem": stem,
        "options": options,
    }


def scrape_response_distribution(table_id: int) -> tuple[list[float], int]:
    response = requests.post(
        "https://www.nationsreportcard.gov/nqt/api/queryresults/GetItemPerformanceData",
        json={
            "itemTableID": table_id,
            "ndeSystemId": "1",
            "subjectCode": "RED",
            "jurisdictionsSelected": "NT",
            "output": 0,
            "showStandardError": False,
            "statistics": ["MN", "RP"],
            "variablesSelected": "TOTAL",
        },
    )
    response = response.json()
    answer_percentages = {}
    correct_answer_label = None
    for data_point in response["result"]["series"][0]["dataPoints"]:
        answer_label = data_point["AxisLabel"].strip()
        if match := re.match(r"([A-Z])\s*\*", answer_label):
            answer_label = match.group(1)
            assert correct_answer_label is None
            correct_answer_label = answer_label
        if answer_label in string.ascii_uppercase:
            answer_percentages[answer_label] = data_point["YValues"][0]

    labels = string.ascii_uppercase[: len(answer_percentages)]
    response_distribution = [answer_percentages[label] / 100 for label in labels]
    correct_option_index = labels.index(correct_answer_label)

    return response_distribution, correct_option_index


def main(args):
    with open(f"output/naep-{args.subject}-metadata.jsonl") as f:
        item_metadata = [json.loads(line) for line in f]

    items = {}
    response_distributions = []
    has_passage = args.subject == "reading"

    for metadata in tqdm(item_metadata, "Scraping items"):
        table_id = metadata["table_id"]
        naep_id = metadata["naep_id"]
        if naep_id in BAD_ITEMS:
            print(f"Skipping {naep_id} due to known issues", file=sys.stderr)
            continue

        item = scrape_item(table_id, has_passage)
        if item is None:
            print(
                f"Skipping {naep_id} due to item type or missing text", file=sys.stderr
            )
            continue
        item["item_id"] = f"NAEP-{naep_id}"

        response_distribution, correct_option_index = scrape_response_distribution(
            table_id
        )
        item["correct_option_index"] = correct_option_index

        if naep_id in items:
            known_mismatches = {
                "R060310",  # Newlines in stem
                "H058201",  # Different wording in image alt text
                "H034701",  # Added source for speech excerpt
                "H034801",  # Different wording in image alt text
                "H061001",  # Different wording in stem
                "H033801",  # Different wording in image alt text
            }
            assert (
                items[naep_id] == item or naep_id in known_mismatches
            ), f"Mismatch:\n\n{items[naep_id]}\n\n{item}"
        else:
            items[naep_id] = item

        response_distributions.append(
            {
                "item_id": f"NAEP-{naep_id}",
                "grade": metadata["grade"],
                "response_distribution": response_distribution,
            }
        )

    with open(f"output/naep-{args.subject}-items.jsonl", "w") as f:
        for item in items.values():
            f.write(json.dumps(item) + "\n")

    with open(f"output/naep-{args.subject}-response-distributions.jsonl", "w") as f:
        for response_distribution in response_distributions:
            f.write(json.dumps(response_distribution) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("subject", help="Subject to scrape NAEP items for (e.g. reading)")
    args = parser.parse_args()

    main(args)
