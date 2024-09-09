# GPTScore Prompt Template

## Factuality 
Evaluate Factuality in the Generated Response

You will be given a source text and a generated response. Your task is to evaluate the factuality of the generated response by comparing it to the source text.

**Evaluation Criteria**:
- Factuality (1-5): Does the generated response accurately reflect the factual statements found in the source text? A score of 1 means that the response contains multiple inaccuracies or fabricated information, while a score of 5 means that the response is entirely accurate and preserves all factual details from the source text.

**Evaluation Steps**:
1. Carefully read the source text to identify key factual statements and details.
2. Review the generated response and compare it to the source text, focusing on the accuracy and integrity of the facts presented.
3. Assign a score for factuality on a scale of 1 to 5 based on the Evaluation Criteria.

**Important**: Ensure that all responses, including explanations and scores, are generated in Korean.

**Example**:
Source Text:
{source_text}

Generated Response:
{generated_response}
                        
Factuality Score (1-5):

## Consistency 
Evaluate Consistency in the Generated Response

You will be given a source text and a generated response. Your task is to evaluate the consistency of the generated response by comparing it to the source text.

**Evaluation Criteria**:
- Consistency (1-5): Does the generated response consistently provide information that aligns with the source text? A score of 1 means that the response contains contradictory or conflicting information, while a score of 5 means that the response is fully consistent with no discrepancies.

**Evaluation Steps**:
1. Carefully read the source text to understand the key points and details.
2. Review the generated response and compare it to the source text, focusing on the consistency of the information provided.
3. Provide a detailed explanation of your evaluation, noting any inconsistencies or confirming the consistency of the response.
4. Assign a score for consistency on a scale of 1 to 5 based on the Evaluation Criteria.

**Important**: Ensure that all responses, including explanations and scores, are generated in Korean.

**Example**:
Source Text:
{source_text}

Generated Response:
{generated_response}

Evaluation Explanation:
- Provide an analysis of the consistency, highlighting specific aspects where the generated response aligns or diverges from the source text.

Consistency Score (1-5):

## Relevance 
Evaluate Relevance in the Generated Response

You will be given a source text and a generated response. Your task is to evaluate the relevance of the generated response by comparing it to the source text.

**Evaluation Criteria**:
- Relevance (1-5): How well does the generated response relate to the source text? A score of 1 means that the response is largely irrelevant or off-topic, while a score of 5 means that the response is highly relevant and directly addresses the key points of the source text.

**Evaluation Steps**:
1. Carefully read the source text to understand its main topics and key points.
2. Review the generated response and compare it to the source text, focusing on how well it addresses the main topics and key points.
3. Provide a detailed explanation of your evaluation, highlighting specific areas where the generated response is relevant or irrelevant to the source text.
4. Assign a score for relevance on a scale of 1 to 5 based on the Evaluation Criteria.

**Important**: Ensure that all responses, including explanations and scores, are generated in Korean.

**Example**:
Source Text:
{source_text}

Generated Response:
{generated_response}

Evaluation Explanation:
- Provide an analysis of the relevance, highlighting specific aspects where the generated response aligns or diverges from the main topics of the source text.

Relevance Score (1-5):

## Fluency 
Evaluate Fluency in the Generated Response

You will be given a source text and a generated response. Your task is to evaluate the fluency of the generated response by assessing its grammatical correctness and readability.

**Evaluation Criteria**:
- Fluency (1-5): How well is the generated response written in terms of grammar, syntax, and overall readability? A score of 1 means the response is poorly written, with numerous grammatical errors or awkward phrasing, while a score of 5 means the response is highly fluent, with excellent grammar and smooth readability.

**Evaluation Steps**:
1. Carefully read the source text to understand its style and tone.
2. Review the generated response, focusing on its grammatical correctness, sentence structure, and overall readability.
3. Provide a detailed explanation of your evaluation, noting any grammatical errors, awkward phrasing, or confirming the fluency of the response.
4. Assign a score for fluency on a scale of 1 to 5 based on the Evaluation Criteria.

**Important**: Ensure that all responses, including explanations and scores, are generated in Korean.

**Example**:
Source Text:
{source_text}

Generated Response:
{generated_response}

Evaluation Explanation:
- Provide an analysis of the fluency, highlighting specific aspects where the generated response is grammatically correct, easy to read, or where it may have issues with fluency.

Fluency Score (1-5):

## Coherence
Evaluate Coherence in the Generated Response

You will be given a source text and a generated response. Your task is to evaluate the coherence of the generated response by comparing it to the source text.

**Evaluation Criteria**:
- Coherence (1-5): How well does the generated response make sense as a unified piece of text? A score of 1 means that the response is disjointed or lacks logical flow, while a score of 5 means that the response is well-structured and logically consistent throughout.

**Evaluation Steps**:
1. Carefully read the source text to understand its overall structure and key points.
2. Review the generated response and assess its logical flow, structure, and overall coherence.
3. Provide a detailed explanation of your evaluation, noting any logical inconsistencies or confirming the coherence of the response.
4. Assign a score for coherence on a scale of 1 to 5 based on the Evaluation Criteria.

**Important**: Ensure that all responses, including explanations and scores, are generated in Korean.

**Example**:
Source Text:
{source_text}

Generated Response:
{generated_response}

Evaluation Explanation:
- Provide an analysis of the coherence, highlighting specific aspects where the generated response aligns or diverges in its logical flow and structure compared to the source text.

Coherence Score (1-5):

## Accuracy 
Evaluate Accuracy in the Generated Response

You will be given a source text and a generated response. Your task is to evaluate the accuracy of the generated response by comparing it to the source text.

**Evaluation Criteria**:
- Accuracy (1-5): Are there any inaccuracies, omissions, or unfactual content in the generated response? A score of 1 means that the response contains significant inaccuracies or incorrect information, while a score of 5 means that the response is fully accurate and correctly reflects the source text.

**Evaluation Steps**:
1. Carefully read the source text to identify all the key facts and details.
2. Review the generated response and compare it to the source text, focusing on the accuracy of the information provided.
3. Provide a detailed explanation of your evaluation, highlighting any inaccuracies, omissions, or confirming the accuracy of the response.
4. Assign a score for accuracy on a scale of 1 to 5 based on the Evaluation Criteria.

**Important**: Ensure that all responses, including explanations and scores, are generated in Korean.

**Example**:
Source Text:
{source_text}

Generated Response:
{generated_response}

Evaluation Explanation:
- Provide an analysis of the accuracy, highlighting specific aspects where the generated response aligns with or diverges from the facts in the source text.

Accuracy Score (1-5):

## Multidimensional Quality 
Evaluate Multidimensional Quality in the Generated Response

You will be given a source text and a generated response. Your task is to evaluate the overall quality of the generated response across multiple dimensions, including factuality, coherence, relevance, and accuracy.

**Evaluation Criteria**:
- Multidimensional Quality (1-5): How well does the generated response perform across all relevant quality dimensions (factuality, coherence, relevance, accuracy)? A score of 1 means the response performs poorly in most or all dimensions, while a score of 5 means the response performs excellently across all dimensions.

**Evaluation Steps**:
1. Carefully read the source text to understand the main content and quality dimensions.
2. Review the generated response, assessing its performance in terms of factuality, coherence, relevance, and accuracy.
3. Provide a detailed explanation of your evaluation, highlighting the strengths and weaknesses of the response across all quality dimensions.
4. Assign a score for overall multidimensional quality on a scale of 1 to 5 based on the Evaluation Criteria.

**Important**: Ensure that all responses, including explanations and scores, are generated in Korean.

**Example**:
Source Text:
{source_text}

Generated Response:
{generated_response}

Evaluation Explanation:
- Provide an analysis of the overall quality, considering the factuality, coherence, relevance, and accuracy of the generated response.

Multidimensional Quality Score (1-5): 

## Semantic Appropriateness 
Evaluate Semantic Appropriateness in the Generated Response

You will be given a source text and a generated response. Your task is to evaluate the semantic appropriateness of the generated response by comparing it to the source text.

**Evaluation Criteria**:
- Semantic Appropriateness (1-5): How well does the generated response convey meaning that is appropriate and aligned with the context of the source text? A score of 1 means the response is semantically inappropriate or off-context, while a score of 5 means the response is fully appropriate and semantically consistent with the source text.

**Evaluation Steps**:
1. Carefully read the source text to understand its meaning and context.
2. Review the generated response and assess whether the meaning it conveys is appropriate and aligned with the source text.
3. Provide a detailed explanation of your evaluation, highlighting areas where the generated response is semantically appropriate or where it diverges from the intended meaning.
4. Assign a score for semantic appropriateness on a scale of 1 to 5 based on the Evaluation Criteria.

**Important**: Ensure that all responses, including explanations and scores, are generated in Korean.

**Example**:
Source Text:
{source_text}

Generated Response:
{generated_response}

Evaluation Explanation:
- Provide an analysis of the semantic appropriateness, highlighting specific aspects where the generated response aligns or diverges from the meaning and context of the source text.

Semantic Appropriateness Score (1-5):

## Understandability
Evaluate Understandability in the Generated Response

You will be given a source text and a generated response. Your task is to evaluate the understandability of the generated response by considering how clear and easy it is to comprehend.

**Evaluation Criteria**:
- Understandability (1-5): How easy is it to understand the generated response? A score of 1 means that the response is confusing, unclear, or difficult to comprehend, while a score of 5 means that the response is very clear, straightforward, and easy to understand.

**Evaluation Steps**:
1. Carefully read the source text to understand its main points and context.
2. Review the generated response, focusing on how clearly the information is presented and whether it is easy to follow.
3. Provide a detailed explanation of your evaluation, noting any areas where the response is particularly clear or where it may be confusing.
4. Assign a score for understandability on a scale of 1 to 5 based on the Evaluation Criteria.

**Important**: Ensure that all responses, including explanations and scores, are generated in Korean.

**Example**:
Source Text:
{source_text}

Generated Response:
{generated_response}

Evaluation Explanation:
- Provide an analysis of the understandability, highlighting specific aspects where the generated response is clear or where it may be difficult to comprehend.

Understandability Score (1-5):
