import logging

from langgraph.graph import END
from langgraph.types import interrupt

from src.state import AgentState

logger = logging.getLogger(__name__)


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
            logger.info("用户拒绝: human_approved=False")
            return {"final_report": "用户取消了生成。", "human_approved": False}
        # 用户批准：设置 human_approved=True，让 after_approval_node 路由到 writer
        logger.info("用户批准: human_approved=True")
        return {"human_approved": True}
    # 如果不需要审批，继续走到 writer
    logger.info("无需审批，直接通过")
    return {"human_approved": True}


async def after_approval_node(state: AgentState) -> str:
    from typing import cast

    approved = cast("bool", state.get("human_approved", False))
    logger.info(
        "审批后路由: human_approved=%s, has_final_report=%s, search_results=%d",
        approved,
        "final_report" in state,
        len(state.get("search_results", [])),
    )
    return "writer" if approved else cast("str", END)
