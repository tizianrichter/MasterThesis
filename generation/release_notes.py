class ReleaseNoteGenerator:
    def __init__(self, llm):
        self.llm = llm

    def build_prompt_rag(
            self,
            artifacts: str,
            version_from: str,
            version_to: str,
            project_context: str,
            related_items: str,
            ex_artifacts1: str,
            ex_version_from1: str,
            ex_version_to1: str,
            ex_release_notes1: str,
            ex_related_items1: str,
    ) -> str:
        """
        Build a release notes prompt using RAG
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

    PROJECT CONTEXT:
    {project_context}

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

    RAG EXAMPLE 1:
    INPUT DATA:
    Software version: **{ex_version_from1} → {ex_version_to1}**
    The following commits occurred between these versions (grouped by category):

    {ex_artifacts1}

    The following PRs/Issues occurred between these versions:

    {ex_related_items1}

    HUMAN WRITTEN RELEASE NOTES:
    {ex_release_notes1}

    NOW GENERATE RELEASE NOTES FOR THE FOLLOWING RELEASE:
    INPUT DATA:
    Software version: **{version_from} → {version_to}**
    The following commits occurred between these versions (grouped by category):

    {artifacts}

    The following PRs/Issues occurred between these versions:

    {related_items}
    """

    def build_prompt_two_shot(
            self,
            artifacts: str,
            version_from: str,
            version_to: str,
            project_context: str,
            related_items: str,
            ex_artifacts1: str,
            ex_version_from1: str,
            ex_version_to1: str,
            ex_release_notes1: str,
            ex_related_items1: str,
            ex_artifacts2: str,
            ex_version_from2: str,
            ex_version_to2: str,
            ex_release_notes2: str,
            ex_related_items2: str
    ) -> str:
        """
        Build a release notes prompt using two-shot
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
    
    PROJECT CONTEXT:
    {project_context}

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

    FEW-SHOT EXAMPLE 1:
    INPUT DATA:
    Software version: **{ex_version_from1} → {ex_version_to1}**
    The following commits occurred between these versions (grouped by category):

    {ex_artifacts1}
    
    The following PRs/Issues occurred between these versions:
    
    {ex_related_items1}
    
    HUMAN WRITTEN RELEASE NOTES:
    {ex_release_notes1}
    
    FEW-SHOT EXAMPLE 2:
    INPUT DATA:
    Software version: **{ex_version_from2} → {ex_version_to2}**
    The following commits occurred between these versions (grouped by category):

    {ex_artifacts2}
    
    The following PRs/Issues occurred between these versions:
    
    {ex_related_items2}

    HUMAN WRITTEN RELEASE NOTES:
    {ex_release_notes2}

    NOW GENERATE RELEASE NOTES FOR THE FOLLOWING RELEASE:
    INPUT DATA:
    Software version: **{version_from} → {version_to}**
    The following commits occurred between these versions (grouped by category):

    {artifacts}
    
    The following PRs/Issues occurred between these versions:
    
    {related_items}
    """

    def build_prompt_one_shot(
            self,
            artifacts: str,
            version_from: str,
            version_to: str,
            project_context: str,
            related_items: str,
            ex_artifacts1: str,
            ex_version_from1: str,
            ex_version_to1: str,
            ex_release_notes1: str,
            ex_related_items1: str,
    ) -> str:
        """
        Build a release notes prompt using one-shot
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

    PROJECT CONTEXT:
    {project_context}

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

    FEW-SHOT EXAMPLE 1:
    INPUT DATA:
    Software version: **{ex_version_from1} → {ex_version_to1}**
    The following commits occurred between these versions (grouped by category):

    {ex_artifacts1}

    The following PRs/Issues occurred between these versions:

    {ex_related_items1}

    HUMAN WRITTEN RELEASE NOTES:
    {ex_release_notes1}

    NOW GENERATE RELEASE NOTES FOR THE FOLLOWING RELEASE:
    INPUT DATA:
    Software version: **{version_from} → {version_to}**
    The following commits occurred between these versions (grouped by category):

    {artifacts}

    The following PRs/Issues occurred between these versions:

    {related_items}
    """

    def build_prompt_zero_shot(
            self,
            artifacts: str,
            version_from: str,
            version_to: str,
            project_context: str,
            related_items: str,
    ) -> str:
        """
        Build a release notes prompt using zero-shot
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

    PROJECT CONTEXT:
    {project_context}

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

    NOW GENERATE RELEASE NOTES FOR THE FOLLOWING RELEASE:
    INPUT DATA:
    Software version: **{version_from} → {version_to}**
    The following commits occurred between these versions (grouped by category):

    {artifacts}

    The following PRs/Issues occurred between these versions:

    {related_items}
    """

    def generate(self, prompt: str, temperature: float, top_p: float) -> str:
        return self.llm.generate(prompt, temperature, top_p)
