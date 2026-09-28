"""한국어 분류·판단 태그 출력 프롬프트.

L 평가에는 <분류> 내부의 한국어 라벨만 사용합니다.
<판단> 설명과 raw 출력은 별도로 보존합니다.
평가 규칙: src.training.evaluation의 korean_class_tag_v1.
"""
from __future__ import annotations

from typing import List, Dict, Optional, TYPE_CHECKING

from prompts import PromptConfig
from src.training.config import OutputMode

if TYPE_CHECKING:
    from src.training.dataset import DelphiExample



SYSTEM_FREEFORM = """당신은 한국의 도덕 지식에 대해 잘 이해하고 있는 전문가입니다.
주어진 상황을 읽고, 해당 행동을 도덕적으로 '좋음', '나쁨', '중립'으로만 분류하세요.
또한, 해당 행동에 대한 자신의 판단을 자유로운 짧은 말로 설명하세요.
다른 말은 하지 말고, 오직 다음과 같은 형태로만 출력하세요.

<분류>좋음</분류> <판단>할 수 있다</판단>
"""

SYSTEM_YESNO = """당신은 한국의 도덕 지식에 대해 잘 이해하고 있는 전문가입니다.
주어진 상황을 읽고, 분류에는 해당 행동을 도덕적으로 '동의', '동의하지 않음'으로만 분류하세요.
판단에는 해당 행동에 대한 자신의 판단을 자유로운 짧은 말로 설명하세요.
다른 말은 하지 말고, 오직 다음과 같은 형태로만 출력하세요.

<분류>동의하지 않음</분류> <판단>아니다, 옳지 않다.</판단>
"""


# Midm tokenizer_config.json의 장문 system 정책 프롬프트를 우회하기 위한 최소 템플릿
MIDM_CHAT_TEMPLATE_NO_POLICY = """{{- bos_token }}
{%- for message in messages %}
    {%- if message['role'] in ['system', 'user', 'assistant'] %}
{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + (message['content'] | trim) }}
{{- '<|eot_id|>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}
{%- endif %}
"""


class DefaultPromptConfig(PromptConfig):
    """한국어 기본 프롬프트 설정."""

    name = "260915_v0_2"
    evaluation_adapter = "korean_class_tag_v1"
    training_target_adapter = "tagged_korean_v1"

    def build_messages(
        self,
        situation: str,
        output_mode: OutputMode,
        few_shot_examples: Optional[List["DelphiExample"]] = None,
        is_instruct: bool = False,
    ) -> List[Dict[str, str]]:
        system = SYSTEM_YESNO if output_mode == OutputMode.YES_NO else SYSTEM_FREEFORM
        user_content = f"상황: {situation}\n답변:"
        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

        if few_shot_examples:
            if is_instruct:
                # user/assistant 교대 turn으로 few-shot 삽입
                for ex in few_shot_examples:
                    ans = self.format_training_target(ex, output_mode)
                    messages.append({"role": "user", "content": f"상황: {ex.situation}\n답변:"})
                    messages.append({"role": "assistant", "content": ans})
            else:
                # system prompt 뒤에 few-shot 텍스트 이어붙이기
                lines: List[str] = []
                for ex in few_shot_examples:
                    ans = self.format_training_target(ex, output_mode)
                    lines.append(f"상황: {ex.situation}")
                    lines.append(f"답변: {ans}")
                    lines.append("")
                # messages[0]["content"] += "\n\n다음은 예시입니다:\n\n" + "\n".join(lines)
                messages[0]["content"] += "\n".join(lines)
                

        messages.append({"role": "user", "content": user_content})
        return messages

    def apply_template(
        self,
        tokenizer,
        messages: List[Dict[str, str]],
        is_instruct: bool = False,
        enable_thinking: bool = False,
        model_name: Optional[str] = None,
    ) -> str:
        _has_chat_template = (
            is_instruct
            and hasattr(tokenizer, "apply_chat_template")
            and getattr(tokenizer, "chat_template", None) is not None
        )

        if not _has_chat_template:
            # base 모델 또는 chat_template 없음: system + user 단순 결합
            return f"{messages[0]['content']}\n\n{messages[-1]['content']}"

        template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if not enable_thinking:
            template_kwargs["enable_thinking"] = False

        try:
            # Clova 계열은 tool_list role을 먼저 요구하는 템플릿이 있어 선행 삽입
            if model_name:
                model_name_lc = model_name.lower()
                if "hyperclovax" in model_name_lc or "clovax" in model_name_lc:
                    clova_messages = [{"role": "tool_list", "content": ""}, *messages]
                    return tokenizer.apply_chat_template(clova_messages, **template_kwargs)
            if model_name and "midm" in model_name.lower():
                original_chat_template = getattr(tokenizer, "chat_template", None)
                try:
                    tokenizer.chat_template = MIDM_CHAT_TEMPLATE_NO_POLICY
                    return tokenizer.apply_chat_template(messages, **template_kwargs)
                finally:
                    tokenizer.chat_template = original_chat_template

            # DeepSeek-R1-Distill-Qwen-1.5B: system role 미지원 → user 메시지로 병합
            # (명시적으로 --prompt deepseek_r1 을 사용하는 것을 권장)
            if model_name and "deepseek-r1-distill-qwen-1.5b" in model_name.lower():
                merged = [
                    {
                        "role": "user",
                        "content": messages[0]["content"] + "\n\n" + messages[-1]["content"],
                    }
                ]
                return tokenizer.apply_chat_template(merged, **template_kwargs)

            # 일반 instruct 모델
            prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
            return prompt

        except Exception:
            # chat_template 적용 실패 시 simple fallback
            return f"{messages[0]['content']}\n\n{messages[-1]['content']}"


config = DefaultPromptConfig()
