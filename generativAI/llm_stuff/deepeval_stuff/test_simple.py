from deepeval.evaluate import assert_test
from deepeval.models import OllamaModel
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from langchain_ollama import OllamaLLM

test_model: OllamaLLM = OllamaLLM(model="mistral")

eval_model: OllamaModel = OllamaModel(
    model="mistral",
    base_url="http://localhost:11434",
    temperature=0
)

answer_relevancy_metric: AnswerRelevancyMetric = AnswerRelevancyMetric(model=eval_model, threshold=0.5, include_reason=True,  verbose_mode=True)

prompt: str = "What is 6 divided by 3?"

test_case: LLMTestCase = LLMTestCase(
  input=prompt,
  expected_output="2",
  actual_output=test_model.invoke(prompt)
)

assert_test(test_case, [answer_relevancy_metric])

answer_relevancy_metric.measure(test_case)
print(answer_relevancy_metric.statements)
print(answer_relevancy_metric.score)
print(answer_relevancy_metric.reason)

