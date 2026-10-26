"""Prompt Templates for Rigel Evaluation

Manages prompts for different modalities (video/image) and scenarios (with/without references).
"""

from typing import List, Optional


def create_prompt(
    cand: str,
    refs: Optional[List[str]] = None,
    modality: str = "video"
) -> str:
    """Create evaluation prompt based on modality and references.

    Args:
        cand: Candidate caption to evaluate
        refs: Optional list of reference captions
        modality: "video" or "image"

    Returns:
        Formatted prompt string
    """
    if modality not in ["video", "image"]:
        raise ValueError(f"Invalid modality: {modality}. Must be 'video' or 'image'")

    # Modality-specific terminology
    if modality == "video":
        content_term = "video"
        frames_term = "video frames"
        examine_instruction = "Examine the video frames to understand the main content"
    else:  # image
        content_term = "image"
        frames_term = "image"
        examine_instruction = "Examine the image to understand the main content"

    # Build prompt based on whether references are provided
    if refs and len(refs) > 0:
        refs_text = "\n".join([f"- {ref}" for ref in refs])
        header = (
            f"Evaluate the quality of a {content_term} caption "
            f"based on {frames_term} and reference captions."
        )
        prompt = f"""{header}

Evaluation Criteria:
- Score ranges from 1 to 5
- 1: Completely incorrect or irrelevant
- 5: Perfectly describes the {content_term} content with all key information
- Penalize redundancies and irrelevant information
- Score 1 if the generated caption contains any hallucination

Evaluation Steps:
1. {examine_instruction}
2. Assess how accurately the caption describes the {content_term}
3. Compare the generated caption to both {frames_term} and references
4. Assess coverage of main points and relevance
5. Assign ONE score from 1 to 5

Reference Captions:
{refs_text}

Generated Caption:
{cand}

Please output only a single integer from 1 to 5, without any explanation or formatting.
Your score is"""
    else:
        prompt = f"""Evaluate the quality of a {content_term} caption based on {frames_term}.

Evaluation Criteria:
- Score ranges from 1 to 5
- 1: Completely incorrect or irrelevant
- 5: Perfectly describes the {content_term} content
- Penalize redundancies and irrelevant information
- Score 1 if the generated caption contains any hallucination

Evaluation Steps:
1. {examine_instruction}
2. Assess how accurately the caption describes the {content_term}
3. Compare the generated caption to {frames_term}
4. Assess coverage of main points and relevance
5. Assign ONE score from 1 to 5

Generated Caption:
{cand}

Please output only a single integer from 1 to 5, without any explanation or formatting.
Your score is"""

    return prompt
