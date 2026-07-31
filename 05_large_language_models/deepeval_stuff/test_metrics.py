from deepeval.evaluate import assert_test
from deepeval.metrics import GEval, AnswerRelevancyMetric, ExactMatchMetric, PatternMatchMetric, JsonCorrectnessMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from langchain_ollama import OllamaLLM
from pydantic import BaseModel

test_model: OllamaLLM = OllamaLLM(model="mistral")

eval_model: OllamaModel = OllamaModel(
    model="mistral",
    base_url="http://localhost:11434",
    temperature=0
)

def test_exact_answer() -> None:
    prompt: str = "How many dwarfs are in the story of Snow White? Answer only with the number, nothing else"
    expected_output: str = "7"
    actual_output: str = test_model.invoke(prompt)

    test_case: LLMTestCase = LLMTestCase(
        input=prompt,
        expected_output=expected_output,
        actual_output=actual_output
    )

    metric: ExactMatchMetric = ExactMatchMetric(threshold=0.5)

    assert_test(test_case, [metric])

def test_generate_pattern() -> None:
    prompt: str = "Generate a email adress, return only the email adress, nothing else"
    actual_output = test_model.invoke(prompt)

    test_case: LLMTestCase = LLMTestCase(
        input=prompt,
        actual_output=actual_output,
    )

    metric: PatternMatchMetric = PatternMatchMetric(
        pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$",
        ignore_case=False,
        threshold=1.0
    )

    assert_test(test_case, [metric])


def test_generate_json() -> None:
    prompt: str = "Generate a JSON object with name and age fields only, return only the JSON, nothing else"
    actual_output = test_model.invoke(prompt)

    test_case: LLMTestCase = LLMTestCase(
        input=prompt,
        actual_output=actual_output,
    )

    class ExampleSchema(BaseModel):
        name: str
        age: int

    metric: JsonCorrectnessMetric = JsonCorrectnessMetric(
        model=eval_model,
        expected_schema=ExampleSchema,
        include_reason=True
    )

    assert_test(test_case, [metric])

def test_relevance_simple() -> None:
    prompt: str = "Tell me about the eiffel tower."
    expected: str = "Eiffel Tower"
    actual_output: str = test_model.invoke(prompt)

    test_case: LLMTestCase = LLMTestCase(
        input= prompt,
        expected_output=expected,
        actual_output=actual_output
    )

    relevance = AnswerRelevancyMetric(model=eval_model,)

    assert_test(test_case, [relevance])
