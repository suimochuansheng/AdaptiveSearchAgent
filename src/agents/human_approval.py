from langgraph.graph import END
from langgraph.types import interrupt

from src.state import AgentState


async def human_approval_node(state: AgentState) -> dict:
    """人工审批节点，使用 interrupt 挂起图等待外部输入"""
    # 如果需要审批（比如置信度达标后询问是否继续）
    if state.get("need_approval", True):
        # 产生中断，等待前端传回 yes/no
        user_choice = interrupt(
            {
                "question": f"搜索结果置信度为 {state['confidence_score']:.2f}，是否生成报告？",
                "options": ["yes", "no"],
            }
        )
        if user_choice == "no":
            return {"final_report": "用户取消了生成。", "should_end": True}
    # 如果不需要审批或用户选择 yes，继续走到 writer
    return {}


async def after_approval_node(state: AgentState) -> str:
    from typing import cast

    # cast(bool, ...) 告诉 mypy “相信我，这个值一定是 bool 类型”
    approved = cast(bool, state.get("human_approved", False))
    return "writer" if approved else cast(str, END)
