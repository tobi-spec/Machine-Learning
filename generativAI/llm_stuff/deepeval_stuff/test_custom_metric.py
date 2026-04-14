from deepeval.evaluate import assert_test
from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from langchain_ollama import OllamaLLM

test_model: OllamaLLM = OllamaLLM(model="mistral")

eval_model: OllamaModel = OllamaModel(
    model="mistral",
    base_url="http://localhost:11434",
    temperature=0
)

correctness: GEval = GEval(model=eval_model,
                    name="correctness",
                    criteria="Determine if the correct answer 2 is clearly present in the actual output, even if there is additional explanation.",
                    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT])

prompt: str = "What is 6 divided by 3?"
test_case: LLMTestCase = LLMTestCase(input=prompt,
                        expected_output="2",
                        actual_output=test_model.invoke(prompt))

assert_test(test_case, [correctness])