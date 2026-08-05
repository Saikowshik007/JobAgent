from collections import defaultdict
import yaml
import config


class Prompts:
    """
    A class to load and manage prompt templates and extractor descriptions from a YAML configuration file.
    """

    lookup = None
    descriptions = None

    @classmethod
    def initialize(cls):
        """
        Initialize the Prompts class by loading the YAML files and setting up the lookup dictionary.
        """
        cls.lookup = cls._load_prompts(config.get("files.prompts_yaml"))
        cls.descriptions = cls._load_descriptions(config.get("files.descriptions_yaml"))

    @staticmethod
    def _load_prompts(yaml_path: str) -> dict:
        """
        Load provider-neutral prompt definitions from a YAML file.

        :param yaml_path: Path to the YAML file containing prompt configurations.
        :return: A dictionary with prompt types as keys and lists of message templates as values.
        """
        with open(yaml_path, "r") as file:
            prompts_data = yaml.safe_load(file)

        return prompts_data

    @classmethod
    def render_messages(cls, prompt_type: str, **values) -> list[dict[str, str]]:
        """Render a prompt as provider-neutral Responses API input messages."""
        if cls.lookup is None:
            cls.initialize()
        prompt = cls.lookup[prompt_type]
        safe_values = defaultdict(str, values)
        user_content = "\n\n".join(
            template.format_map(safe_values)
            for template in (
                prompt["job_posting_template"],
                prompt.get("resume_template", ""),
                prompt["instruction_message"],
                prompt["criteria_message"],
                prompt["steps_message"],
            )
            if template
        )
        return [
            {"role": "developer", "content": prompt["system_message"]},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _load_descriptions(yaml_path: str) -> dict:
        """
        Load descriptions from a YAML file.

        :param yaml_path: Path to the YAML file containing descriptions.
        :return: A dictionary with descriptions.
        """
        with open(yaml_path, "r") as file:
            descriptions_data = yaml.safe_load(file)
        return descriptions_data
