import importlib
import os
import unittest


class LocalRAGTests(unittest.TestCase):
    def test_llm_local_imports_without_api_key(self):
        os.environ.pop("GEMINI_API_KEY", None)

        module = importlib.import_module("llm_local")

        self.assertTrue(callable(module.generate_answer))

    def test_generate_answer_uses_context_without_external_api(self):
        from llm_local import generate_answer

        context = (
            "Employees can reset their password from the IT portal. "
            "Open the password reset page, verify your identity, and choose a new password.\n\n"
            "Vacation requests are submitted through the HR system."
        )

        answer = generate_answer("How do I reset my password?", context, prefer_local_model=False)

        self.assertIn("password", answer.lower())
        self.assertIn("IT portal", answer)


if __name__ == "__main__":
    unittest.main()
