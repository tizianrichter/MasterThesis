class ReleaseNoteGenerator:
    def __init__(self, llm):
        self.llm = llm

    def build_prompt(self, artifacts: str, version_from: str, version_to: str, project_context: str) -> str:
        """
        Build a release notes prompt for an LLM with:
        - instructions first
        - project context
        - version numbers
        - input artifacts
        """

        return f"""
    You are an expert software technical writer.

    TASK:
    Generate high-quality release notes for a software release.

    GOALS:
    - Summarize user-relevant changes
    - Abstract away low-level implementation details
    - Avoid redundancy
    - Use clear, concise language

    TARGET AUDIENCE:
    End users with technical background.

    GROUNDING RULES (IMPORTANT):
    - Use ONLY the information explicitly provided in the input data.
    - Do NOT invent, assume, or infer changes that are not present.
    - If no information supports a section, OMIT the section entirely.

    OUTPUT FORMAT:
    - Use Markdown
    - Include ONLY these sections if relevant:
      - Added
      - Changed
      - Fixed
      - Removed
      - Security
    - Do NOT include introductions, conclusions, titles, or version numbers
    - Output ONLY the release notes content

    PROJECT CONTEXT:
    {project_context}

    INPUT DATA:
    The software version change is from **{version_from}** to **{version_to}**.
    The following items describe changes between these two versions.
    They may include commits, issues, and pull requests.
    Do NOT list them verbatim.

    NOTE:
    - Commit messages and pull requests may refer to the same change. Treat duplicates as a single change.
    - If a change was reverted within this release, do NOT list it unless the revert itself is user-relevant.

    CHANGES:
    {artifacts}
    """

    def generate(self, prompt: str, temperature: float, top_p: float) -> str:
        return self.llm.generate(prompt, temperature, top_p)
