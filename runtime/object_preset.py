ENVIRONMENT_PRESETS = {
    "generic": {
        "description": "Balanced default for mixed indoor scenes.",
        "object_labels": ["backpack", "handbag", "suitcase", "cell phone", "bottle"],
    },
    "office": {
        "description": "Office and campus hallways.",
        "object_labels": ["backpack", "handbag", "laptop", "cell phone", "cup"],
    },
    "retail": {
        "description": "Retail aisle or store checkout areas.",
        "object_labels": ["backpack", "handbag", "bottle", "cup", "cell phone"],
    },
    "airport": {
        "description": "Terminal, lobby, and travel scenes.",
        "object_labels": ["backpack", "handbag", "suitcase", "laptop", "cell phone"],
    },
    "street": {
        "description": "Outdoor curbside or walkway scenes.",
        "object_labels": ["backpack", "handbag", "suitcase", "cell phone", "bottle"],
    },
}


def available_environments():
    return sorted(ENVIRONMENT_PRESETS.keys())


def resolve_object_labels(explicit_labels=None, environment="generic", max_object_types=5):
    labels = list(explicit_labels or [])
    if not labels:
        labels = list(
            ENVIRONMENT_PRESETS.get(environment, ENVIRONMENT_PRESETS["generic"])["object_labels"]
        )

    deduped = []
    seen = set()
    for label in labels:
        normalized = str(label).strip()
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)

    return deduped[: max(1, int(max_object_types))]


def describe_environment(environment):
    preset = ENVIRONMENT_PRESETS.get(environment, ENVIRONMENT_PRESETS["generic"])
    return {
        "name": environment if environment in ENVIRONMENT_PRESETS else "generic",
        "description": preset["description"],
        "object_labels": list(preset["object_labels"]),
    }
