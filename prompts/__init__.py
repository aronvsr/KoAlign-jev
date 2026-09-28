"""
prompts/ — 프롬프트 설정 관리 패키지

각 파일이 하나의 PromptConfig 인스턴스를 `config` 이름으로 export합니다.
run_evaluation.py에서 --prompt <name> 으로 선택합니다.

현재 제공되는 설정:
  default      : 한국어 기본 (KoAlign, apply_chat_template + [class] suffix)
  default_en   : 영어 (social_chem 등, apply_chat_template + [class] suffix)
  deepseek_r1  : DeepSeek-R1-Distill 전용 (system→user 병합, suffix 없음)

새 설정 추가 방법:
  1. prompts/<name>.py 파일 생성
  2. PromptConfig를 상속한 클래스를 만들고
  3. `config = MyConfig()` 를 파일 최하단에 선언
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.training.dataset import DelphiExample
    from src.training.config import OutputMode


class PromptConfig:
    """프롬프트 설정 기본 클래스.

    서브클래스에서 build_messages와 apply_template을 구현합니다.

    Attributes:
        name: 설정 식별자 (파일명과 일치)
    """

    name: str = "base"

    # ── 메시지 구성 ──────────────────────────────────────────────────────────

    def build_messages(
        self,
        situation: str,
        output_mode: "OutputMode",
        few_shot_examples: Optional[List["DelphiExample"]] = None,
        is_instruct: bool = False,
    ) -> List[Dict[str, str]]:
        """상황(situation)으로부터 chat 메시지 리스트를 반환합니다.

        Args:
            situation: 평가 대상 상황 텍스트
            output_mode: freeform / yes_no
            few_shot_examples: few-shot 예시 리스트 (없으면 None)
            is_instruct: Instruct 모델 여부 (few-shot 포맷 분기에 사용)

        Returns:
            [{"role": "system", "content": ...}, {"role": "user", "content": ...}, ...]
        """
        raise NotImplementedError

    # ── 프롬프트 문자열 변환 ─────────────────────────────────────────────────

    def apply_template(
        self,
        tokenizer,
        messages: List[Dict[str, str]],
        is_instruct: bool = False,
        enable_thinking: bool = False,
        model_name: Optional[str] = None,
    ) -> str:
        """messages 리스트를 모델 입력 프롬프트 문자열로 변환합니다.

        Args:
            tokenizer: HuggingFace tokenizer
            messages: build_messages가 반환한 메시지 리스트
            is_instruct: chat_template 사용 여부 결정에 사용
            enable_thinking: thinking/reasoning 모드 활성화 여부
            model_name: 모델 이름 (모델별 특수 처리에 사용 가능)

        Returns:
            모델에 직접 입력할 프롬프트 문자열
        """
        raise NotImplementedError


# ── 레지스트리 ────────────────────────────────────────────────────────────────

def load_prompt_config(name: str) -> PromptConfig:
    """prompts/<name>.py 에서 PromptConfig 인스턴스를 로드합니다.

    각 파일은 반드시 `config = <SomePromptConfig>()` 를 export해야 합니다.

    Args:
        name: 설정 이름 (파일명에서 .py 제거)

    Returns:
        해당 PromptConfig 인스턴스

    Raises:
        ValueError: 파일을 찾을 수 없는 경우
        AttributeError: 파일에 `config` 가 없는 경우
        TypeError: `config` 가 PromptConfig 인스턴스가 아닌 경우
    """
    # 파일 경로로 직접 로드 (이름에 점이 포함된 경우에도 동작, e.g. '260629_v0.1')
    _prompts_dir = Path(__file__).parent
    _file = _prompts_dir / f"{name}.py"

    if _file.exists():
        spec = importlib.util.spec_from_file_location(f"prompts._dyn_{name}", _file)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise ImportError(f"Failed to load prompts/{name}.py: {exc}") from exc
    else:
        # fallback: 일반 import (패키지로 설치된 경우 등)
        try:
            module = importlib.import_module(f"prompts.{name}")
        except ModuleNotFoundError:
            raise ValueError(
                f"Prompt config '{name}' not found. "
                f"Make sure prompts/{name}.py exists in the project root."
            )

    if not hasattr(module, "config"):
        raise AttributeError(
            f"prompts/{name}.py must export a module-level `config` instance of PromptConfig."
        )

    cfg = module.config
    if not isinstance(cfg, PromptConfig):
        raise TypeError(
            f"prompts/{name}.py: `config` must be a PromptConfig instance, got {type(cfg)}"
        )
    return cfg
