from google import genai
from google.genai import errors
from dotenv import load_dotenv
import os

load_dotenv()

class GeminiClient:
    models = (
        "gemini-3.6-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
    )

    def __init__(self):
        api_key=os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

        self.client = genai.Client(api_key=api_key)

    def _model_candidates(self):
        candidates = list(self.models)

        try:
            available_models = self.client.models.list()
            supported_models = []
            for model in available_models:
                actions = getattr(model, "supported_actions", ()) or ()
                if "generateContent" not in actions:
                    continue

                name = getattr(model, "name", "")
                if name:
                    supported_models.append(name.removeprefix("models/"))

            candidates = [model for model in candidates if model in supported_models]
            candidates.extend(model for model in supported_models if model not in candidates)
        except errors.APIError:
            pass

        return candidates

    def generate(self, prompt:str) -> str:
        last_error = None

        for model in self._model_candidates():
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                return response.text
            except errors.APIError as error:
                last_error = error
                print(f"Model {model} failed ({error.code}); trying the next model.")

        raise RuntimeError("All configured Gemini models failed.") from last_error