class ReleaseNoteGenerator:
    def __init__(self, llm):
        self.llm = llm

    def build_prompt(
            self,
            artifacts: str,
            version_from: str,
            version_to: str,
            ex_artifacts: str,
            ex_version_from: str,
            ex_version_to: str,
            ex_release_notes: str
    ) -> str:
        """
        Build a release notes prompt including a few-shot example release.
        """
        return f"""
    You are an expert software technical writer.

    TASK:
    - Generate high-quality release notes for a software release.
    - Merge related items into concise, user-relevant sentences.
    - Group changes into meaningful sections like "New Features" and "Bug fixes and minor improvements".
    - Avoid repeating trivial implementation details.
    - Include optional short examples (e.g., CLI usage) if relevant to users.
    - Abstract away low-level technical details, focusing on what matters to end users.

    TARGET AUDIENCE:
    End users with a technical background.

    GROUNDING RULES:
    - Use ONLY the information explicitly provided in the input.
    - Do NOT invent, assume, or infer changes that are not present.
    - If a section has no relevant information, omit it entirely.

    OUTPUT FORMAT:
    - Markdown with headings for sections.
    - Each bullet should describe **why the change matters to the user**.
    - Avoid listing every low-level commit verbatim.
    - Combine related changes into single bullets where appropriate.

    FEW-SHOT EXAMPLE:
    INPUT DATA:
    Software version: **{ex_version_from} → {ex_version_to}**
    The following changes occurred between these versions (grouped by category):

    {ex_artifacts}

    HUMAN RELEASE NOTES:
    {ex_release_notes}

    NOW GENERATE RELEASE NOTES FOR THE FOLLOWING RELEASE:
    INPUT DATA:
    Software version: **{version_from} → {version_to}**
    The following changes occurred between these versions (grouped by category):

    {artifacts}
    """

    def generate(self, prompt: str, temperature: float, top_p: float) -> str:
        return self.llm.generate(prompt, temperature, top_p)
