"""
客服回复幻觉分类体系

定义幻觉类型、严重程度、检测规则，作为检测工具的基础分类框架。
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


class Severity(str, Enum):
    """幻觉严重程度"""
    CRITICAL = "严重"   # 可能造成安全风险或实际损失
    HIGH = "高危"       # 严重误导用户决策
    MEDIUM = "中等"     # 部分错误但不致严重后果
    LOW = "轻微"        # 信息遗漏或不精确


class HallucinationCategory(str, Enum):
    """幻觉大类"""
    FACT_FABRICATION = "事实编造"       # 编造不存在的产品参数、信息
    POLICY_DISTORTION = "政策歪曲"     # 扭曲或编造政策、优惠
    CAPABILITY_OVERCLAIM = "能力越界"  # 假装具备不具备的系统能力
    SAFETY_MISGUIDANCE = "安全误导"    # 给出可能危害用户安全的建议
    INFORMATION_OMISSION = "信息遗漏"  # 遗漏关键限定条件或风险提示


@dataclass
class HallucinationSubType:
    """幻觉子类型定义"""
    name: str
    category: HallucinationCategory
    severity: Severity
    description: str
    detection_hint: str  # 检测提示词


# ============================================================
# 幻觉子类型定义表
# ============================================================
HALLUCINATION_TYPES: List[HallucinationSubType] = [
    HallucinationSubType(
        name="安全误导",
        category=HallucinationCategory.SAFETY_MISGUIDANCE,
        severity=Severity.CRITICAL,
        description="回复给出可能危害用户健康或安全的建议，无视知识库中的风险提示",
        detection_hint="知识库有安全警告/风险提示（如孕妇慎用、咨询医生），"
                       "但回复却说'可以放心使用'或类似肯定性表述，忽略或弱化了风险",
    ),
    HallucinationSubType(
        name="能力越界",
        category=HallucinationCategory.CAPABILITY_OVERCLAIM,
        severity=Severity.CRITICAL,
        description="客服系统不具备某项操作能力（查物流/退款进度/修改订单/升级工单），"
                    "回复却声称已执行该操作并给出具体结果",
        detection_hint="知识库明确说'未接入XX接口'或'不具备XX功能'，"
                       "但回复声称'已帮您XX'或给出了需要该接口才能获取的具体信息",
    ),
    HallucinationSubType(
        name="参数编造",
        category=HallucinationCategory.FACT_FABRICATION,
        severity=Severity.HIGH,
        description="编造产品参数（材质、规格、功能、接口等），与知识库中的实际参数矛盾",
        detection_hint="回复中的数值、型号、材质、功能等参数与知识库不一致；"
                       "或知识库未标注但回复给出肯定性参数描述",
    ),
    HallucinationSubType(
        name="信息编造",
        category=HallucinationCategory.FACT_FABRICATION,
        severity=Severity.HIGH,
        description="编造不存在的实体信息（地址、门店、品牌关联、联系人等）",
        detection_hint="回复给出了具体地址/店名/人名/品牌关系等实体信息，"
                       "但知识库中无此信息或明确说明不存在",
    ),
    HallucinationSubType(
        name="政策编造",
        category=HallucinationCategory.POLICY_DISTORTION,
        severity=Severity.MEDIUM,
        description="编造完全不存在或严重偏离实际的政策条款",
        detection_hint="回复中的政策条款（退货天数、优惠规则等）与知识库核心参数矛盾，"
                       "如天数、金额等关键数字被改变",
    ),
    HallucinationSubType(
        name="优惠编造",
        category=HallucinationCategory.POLICY_DISTORTION,
        severity=Severity.MEDIUM,
        description="编造不存在的优惠/折扣/活动，或承诺给予不存在的权益",
        detection_hint="回复承诺的优惠券/折扣/活动在知识库中不存在，"
                       "或条件/金额与知识库不匹配",
    ),
    HallucinationSubType(
        name="政策偏差",
        category=HallucinationCategory.POLICY_DISTORTION,
        severity=Severity.MEDIUM,
        description="回复部分正确但有关键错误：错误的服务选项、流程指引等",
        detection_hint="回复部分内容与知识库一致，但存在个别关键点矛盾（如支持的服务类型、"
                       "操作入口、多选项中的一个错误等）",
    ),
    HallucinationSubType(
        name="信息遗漏",
        category=HallucinationCategory.INFORMATION_OMISSION,
        severity=Severity.LOW,
        description="知识库中有影响用户决策的重要限定信息，回复未提及或给出相反的概括性结论",
        detection_hint="知识库包含重要的限定条件/比例/风险提示（如'30%用户反馈偏大'），"
                       "但回复给出绝对化结论（如'尺码标准'）而未提及该信息",
    ),
]


def get_type_by_name(name: str) -> Optional[HallucinationSubType]:
    """根据类型名称获取子类型定义"""
    for t in HALLUCINATION_TYPES:
        if t.name == name:
            return t
    return None


def get_category_summary() -> str:
    """生成分类体系的文本摘要，用于 LLM prompt"""
    lines = ["## 幻觉分类体系\n"]
    for cat in HallucinationCategory:
        lines.append(f"### {cat.value}")
        subtypes = [t for t in HALLUCINATION_TYPES if t.category == cat]
        for st in subtypes:
            lines.append(f"- **{st.name}** (严重程度: {st.severity.value})")
            lines.append(f"  - {st.description}")
            lines.append(f"  - 检测要点: {st.detection_hint}")
        lines.append("")
    return "\n".join(lines)
