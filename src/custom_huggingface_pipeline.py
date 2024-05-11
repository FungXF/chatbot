from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.runnables import RunnableConfig, ensure_config
from typing import Any, Optional, List, Generator

class CustomHuggingFacePipeline(HuggingFacePipeline):
    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        config = ensure_config(config)
        llm_result = self.generate_prompt(
            [self._convert_input(input)],
            stop=stop,
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            **kwargs,
        ).generations[0][0].text
        for generation in llm_result:
            yield generation