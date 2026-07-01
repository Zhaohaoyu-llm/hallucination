"""
LLM 幻觉检测器 — 使用 Anthropic API 检测客服回复中的幻觉

支持两种模式：
- real: 使用 Anthropic API 进行 LLM 检测
- mock: 使用规则引擎模拟检测（无需 API key）
"""

import json
import os
import re
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

# 添加 src 目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier import (
    HALLUCINATION_TYPES,
    HallucinationCategory,
    Severity,
    get_category_summary,
    get_type_by_name,
)


@dataclass
class DetectionResult:
    """单条检测结果"""
    id: str
    is_hallucination: bool
    hallucination_type: Optional[str] = None
    severity: Optional[str] = None
    category: Optional[str] = None
    detail: str = ""
    confidence: float = 0.0  # 0.0 ~ 1.0


class MockDetector:
    """基于规则引擎的 Mock 检测器

    使用关键字匹配、模式识别等启发式方法模拟 LLM 检测行为。
    虽然不如真实 LLM 准确，但可以作为快速原型验证。
    """

    # 能力越界特征词：知识库出现这些 → 系统无此能力
    CAPABILITY_NEGATION_PATTERNS = [
        r"未接入.*接口",
        r"不具备.*功能",
        r"无[（(]客服系统",
        r"无[（(].*接口",
        r"需.*人工",
        r"不可.*告知",
    ]

    # 回复中"已执行操作"的特征词
    CAPABILITY_CLAIM_PATTERNS = [
        r"已帮您\w+",
        r"已(为|给)您\w+",
        r"已经(将|在|为)",
        r"我帮您\w+了",
        r"直接发到",
        r"已修改为",
    ]

    # 安全风险词（知识库中有但回复忽略的）
    SAFETY_WARNING_PATTERNS = [
        r"孕妇.*建议.*咨询",
        r"不建议.*使用",
        r"咨询医生",
        r"可能.*过敏",
        r"慎用",
    ]

    def __init__(self):
        self.type_defs = {t.name: t for t in HALLUCINATION_TYPES}

    def detect_all(self, replies: List[dict]) -> List[DetectionResult]:
        """批量检测所有回复"""
        return [self.detect_one(r) for r in replies]

    def detect_one(self, reply: dict) -> DetectionResult:
        """检测单条回复"""
        rid = reply["id"]
        system_reply = reply["system_reply"]
        knowledge_base = reply.get("knowledge_base", "")

        # 逐层检测，优先级从高到低
        checks = [
            self._check_safety_misguidance,
            self._check_capability_overclaim,
            self._check_param_fabrication,
            self._check_info_fabrication,
            self._check_policy_fabrication,
            self._check_coupon_fabrication,
            self._check_policy_deviation,
            self._check_information_omission,
        ]

        for check_fn in checks:
            result = check_fn(system_reply, knowledge_base)
            if result:
                return DetectionResult(
                    id=rid,
                    is_hallucination=True,
                    hallucination_type=result["type"],
                    severity=self.type_defs[result["type"]].severity.value,
                    category=self.type_defs[result["type"]].category.value,
                    detail=result["detail"],
                    confidence=result.get("confidence", 0.85),
                )

        # 未检测到幻觉
        return DetectionResult(
            id=rid,
            is_hallucination=False,
            detail="回复与知识库一致，未检测到明显的幻觉问题",
            confidence=0.75,
        )

    def _kb_is_empty(self, kb: str) -> bool:
        """知识库是否表示无相关信息"""
        empty_patterns = [
            r"^无[（(]",
            r"^无$",
            r"^无[。.]?$",
            r"未标注",
            r"无.*相关",
        ]
        kb_clean = kb.strip()
        return any(re.search(p, kb_clean) for p in empty_patterns)

    def _check_safety_misguidance(self, reply: str, kb: str) -> Optional[dict]:
        """检测安全误导"""
        has_warning = any(re.search(p, kb) for p in self.SAFETY_WARNING_PATTERNS)
        if not has_warning:
            return None

        # 知识库有安全警告，回复是否弱化/忽略了？
        reassurance_patterns = [
            r"可以放心使用",
            r"孕妇可以",
            r"没有.*风险",
            r"绝对安全",
            r"完全没问题",
            r"所有人都可以用",
            r"可以.*使用",
        ]
        for pat in reassurance_patterns:
            if re.search(pat, reply):
                return {
                    "type": "安全误导",
                    "detail": f"知识库包含安全风险提示，但回复给出肯定性安全承诺，"
                              f"可能误导用户忽视潜在风险",
                    "confidence": 0.90,
                }
        return None

    def _check_capability_overclaim(self, reply: str, kb: str) -> Optional[dict]:
        """检测能力越界"""
        if not self._kb_is_empty(kb):
            return None

        has_claim = any(re.search(p, reply) for p in self.CAPABILITY_CLAIM_PATTERNS)
        if not has_claim:
            return None

        # 确认 kb 确实表示无此能力
        has_negation = any(re.search(p, kb) for p in self.CAPABILITY_NEGATION_PATTERNS)
        if has_negation:
            return {
                "type": "能力越界",
                "detail": f"知识库表明系统不具备该操作能力，但回复声称已执行了该操作",
                "confidence": 0.92,
            }

        # kb 为空（"无"）但没有显式否定 → 也是能力越界
        return {
            "type": "能力越界",
            "detail": f"知识库中无相关信息（系统未接入该接口），"
                      f"但回复给出了具体操作结果",
            "confidence": 0.88,
        }

    def _check_param_fabrication(self, reply: str, kb: str) -> Optional[dict]:
        """检测参数编造 — 对比产品参数"""
        if self._kb_is_empty(kb):
            # kb 说"未标注"，但回复肯定地说"支持XX功能"
            affirmative_patterns = [
                r"支持的.*功能",
                r"采用.*版本",
                r"支持.*连接",
            ]
            for pat in affirmative_patterns:
                if re.search(pat, reply):
                    return {
                        "type": "参数编造",
                        "detail": f"知识库未标注该功能/参数，但回复给出肯定性参数描述",
                        "confidence": 0.82,
                    }
            return None

        # 参数矛盾检测：对比知识库中的参数和回复中的参数
        param_pairs = [
            # (类别, 知识库模式, 回复矛盾模式, 参数名)
            ("蓝牙版本", r"蓝牙\s*(\d+\.\d+)", r"蓝牙\s*(\d+\.\d+)", "蓝牙版本"),
            ("材质", r"材质[：:]\s*(.+?)[，。\n]", r"(头层牛皮|真皮|纯棉|纯羊毛|羊绒)", "材质"),
            ("保修期", r"保修期[：:]\s*(\d+)\s*个?月", r"保修期[为是]\s*(\w+)", "保修期"),
            ("接口类型", r"接口类型[：:]\s*(.+?)[，。\n]", r"(Type-C|USB-C|Lightning|Micro.?USB)", "接口类型"),
        ]

        for category, kb_pat, reply_pat, param_name in param_pairs:
            kb_match = re.search(kb_pat, kb)
            reply_match = re.search(reply_pat, reply)
            if kb_match and reply_match:
                kb_val = kb_match.group(1).strip()
                reply_val = reply_match.group(1).strip()
                # 归一化比较
                if self._normalize(kb_val) != self._normalize(reply_val):
                    return {
                        "type": "参数编造",
                        "detail": f"{param_name}不一致：知识库为'{kb_val}'，回复为'{reply_val}'",
                        "confidence": 0.90,
                    }

        return None

    def _check_info_fabrication(self, reply: str, kb: str) -> Optional[dict]:
        """检测信息编造 — 编造地址、门店、品牌关系等"""
        # 检测回复中是否包含具体地址
        address_patterns = [
            r"[省市区]\S+路\S+号",
            r"[省市区]\S+大道\S+",
            r"邮编\d{6}",
        ]
        has_address = any(re.search(p, reply) for p in address_patterns)

        # 检测知识库是否禁止给出地址
        kb_forbids_address = any(
            p in kb for p in ["不可.*告知", "需由.*系统", "以短信方式"]
        )

        if has_address and kb_forbids_address:
            return {
                "type": "信息编造",
                "detail": f"回复给出了具体地址/收件人信息，但知识库明确要求此类信息"
                          f"需由系统自动匹配后发送，不可口头告知",
                "confidence": 0.95,
            }

        # 检测门店编造
        store_patterns = [
            r"(北京|上海|广州|深圳|成都|杭州).*(体验店|门店|专卖店|专柜)",
            r"线下.*店",
            r"官网.*门店查询",
        ]
        has_store_info = any(re.search(p, reply) for p in store_patterns)
        kb_denies_store = any(
            p in kb for p in ["纯线上", "无线下", "无.*门店", "无.*实体"]
        )

        if has_store_info and kb_denies_store:
            return {
                "type": "信息编造",
                "detail": f"回复声称有线下门店，但知识库明确品牌为纯线上电商",
                "confidence": 0.95,
            }

        # 检测品牌关系编造
        brand_patterns = [
            r"旗下.*子品牌",
            r"同属.*集团",
            r"共享.*供应链",
            r"姐妹品牌",
        ]
        has_brand_relation = any(re.search(p, reply) for p in brand_patterns)
        kb_no_brand = "未提及" in kb or self._kb_is_empty(kb)

        if has_brand_relation and kb_no_brand:
            return {
                "type": "信息编造",
                "detail": f"回复编造了品牌关联关系，知识库中无相关信息",
                "confidence": 0.88,
            }

        return None

    def _check_policy_fabrication(self, reply: str, kb: str) -> Optional[dict]:
        """检测政策编造 — 退货政策、发货时间等（完全编造新的政策条款）"""
        if self._kb_is_empty(kb):
            return None

        # 退货天数：如果知识库和回复提到的天数相差 2 倍以上 → 政策编造
        # 否则 → 政策偏差（在后续检查中处理）
        kb_days = re.search(r"(\d+)天.*退", kb)
        reply_days = re.search(r"(\d+)天.*退", reply)
        if kb_days and reply_days:
            kb_d = int(kb_days.group(1))
            reply_d = int(reply_days.group(1))
            if kb_d != reply_d:
                # 天数差距超过 2 倍 → 编造；差距小但不同 → 偏差
                if reply_d > kb_d * 2 or reply_d < kb_d / 2:
                    return {
                        "type": "政策编造",
                        "detail": f"退货天数严重不一致：知识库为{kb_d}天，"
                                  f"回复为{reply_d}天",
                        "confidence": 0.92,
                    }
                else:
                    return {
                        "type": "政策偏差",
                        "detail": f"退货天数不一致：知识库为{kb_d}天，"
                                  f"回复为{reply_d}天",
                        "confidence": 0.88,
                    }

        # 发货时间：差距超过 2 倍才算编造，否则是偏差
        kb_ship = re.search(r"(\d+)小时.*发", kb)
        reply_ship = re.search(r"(\d+)小时.*发", reply)
        if kb_ship and reply_ship:
            kb_s = int(kb_ship.group(1))
            reply_s = int(reply_ship.group(1))
            if kb_s != reply_s:
                if reply_s > kb_s * 2 or reply_s < kb_s / 2:
                    return {
                        "type": "政策编造",
                        "detail": f"发货时间严重不一致：知识库为{kb_s}小时，"
                                  f"回复为{reply_s}小时",
                        "confidence": 0.92,
                    }
                else:
                    return {
                        "type": "政策偏差",
                        "detail": f"发货时间不一致：知识库为{kb_s}小时，"
                                  f"回复为{reply_s}小时。此外快递公司也可能存在偏差",
                        "confidence": 0.88,
                    }

        return None

    def _check_coupon_fabrication(self, reply: str, kb: str) -> Optional[dict]:
        """检测优惠编造"""
        if self._kb_is_empty(kb):
            return None

        # 提取回复中的优惠数字
        reply_amounts = set(re.findall(r"满(\d+)减(\d+)", reply))

        # 提取知识库中的优惠，排除否定模式（如"无满300减50"）
        # 先找出被否定的优惠
        negated_patterns = set(re.findall(r"(?:无|没有|不存在)\s*满(\d+)减(\d+)", kb))
        # 再找出所有优惠
        all_kb_patterns = set(re.findall(r"满(\d+)减(\d+)", kb))
        # 有效优惠 = 所有 - 否定
        kb_amounts = all_kb_patterns - negated_patterns

        # 回复中有但知识库没有的优惠
        fabricated = reply_amounts - kb_amounts
        if fabricated:
            examples = [f"满{a}减{b}" for a, b in fabricated]
            return {
                "type": "优惠编造",
                "detail": f"回复声称存在'{'; '.join(examples)}'优惠，"
                          f"但知识库中无此优惠活动",
                "confidence": 0.93,
            }

        # 检测学生优惠：需区分"kb真的没有" 和 "kb有但是否定的"
        if "学生" in reply and ("优惠" in reply or "折扣" in reply):
            # KB 否定学生优惠（如"当前无学生优惠"）→ 编造
            if re.search(r"无.*学生", kb):
                return {
                    "type": "优惠编造",
                    "detail": f"回复声称有学生优惠，但知识库明确'无学生优惠政策'",
                    "confidence": 0.92,
                }
            # KB 完全没有提及学生 → 也是编造
            if "学生" not in kb:
                return {
                    "type": "优惠编造",
                    "detail": f"回复声称有学生优惠，但知识库中无相关优惠政策",
                    "confidence": 0.90,
                }

        return None

    def _check_policy_deviation(self, reply: str, kb: str) -> Optional[dict]:
        """检测政策偏差 — 部分正确但有错误"""
        if self._kb_is_empty(kb):
            return None

        # 发票类型偏差
        if "发票" in reply and "发票" in kb:
            kb_e = "电子发票" in kb
            kb_paper = "纸质发票" in kb
            reply_e = "电子发票" in reply
            reply_paper = "纸质发票" in reply

            if kb_e and not kb_paper and reply_paper:
                return {
                    "type": "政策偏差",
                    "detail": f"回复声称支持纸质发票，但知识库明确'暂不支持纸质发票'",
                    "confidence": 0.88,
                }

        # 申请入口偏差
        if "备注" in reply and "发票" in reply:
            if "订单详情页" in kb and "备注" not in kb:
                return {
                    "type": "政策偏差",
                    "detail": f"发票申请入口指引错误：回复让用户在备注填写，"
                              f"但知识库要求通过订单详情页申请",
                    "confidence": 0.85,
                }

        # 快递公司偏差
        kb_couriers = set(re.findall(r"(顺丰|中通|韵达|圆通|申通|极兔|京东)", kb))
        reply_couriers = set(re.findall(r"(顺丰|中通|韵达|圆通|申通|极兔|京东)", reply))
        if kb_couriers and reply_couriers and kb_couriers != reply_couriers:
            return {
                "type": "政策偏差",
                "detail": f"快递公司信息不一致：知识库为{', '.join(kb_couriers)}，"
                          f"回复为{', '.join(reply_couriers)}",
                "confidence": 0.87,
            }

        return None

    def _check_information_omission(self, reply: str, kb: str) -> Optional[dict]:
        """检测信息遗漏"""
        if self._kb_is_empty(kb):
            return None

        # 知识库中有比例/统计数据，回复给出绝对化结论
        kb_has_percentage = bool(re.search(r"(\d+)%", kb))
        reply_absolute = any(
            p in reply for p in ["标准", "正常", "不偏", "都一样", "没问题"]
        )

        if kb_has_percentage and reply_absolute and "反馈" in kb:
            return {
                "type": "信息遗漏",
                "detail": f"知识库包含用户反馈统计数据，但回复给出绝对化结论，"
                          f"遗漏了可能影响用户决策的关键信息",
                "confidence": 0.72,
            }

        return None

    @staticmethod
    def _normalize(text: str) -> str:
        """归一化文本用于比较"""
        return re.sub(r"\s+", "", text).lower()


class LLMDetector:
    """使用 Anthropic API 进行幻觉检测"""

    SYSTEM_PROMPT = """你是一个客服质量审核专家，专门检测智能客服回复中的"幻觉"问题。

你的任务是：将客服回复与知识库进行对比，判断回复是否存在幻觉，并对幻觉进行分类。

## 幻觉分类体系

{category_summary}

## 检测原则

1. **严格对比知识库**：一切以知识库为基准。知识库没有的信息 → 可能是编造。
2. **注意隐含否定**：知识库说"无（系统未接入XX）"= 系统不具备该能力。
3. **安全优先**：知识库有风险提示但回复弱化/忽略 → 安全误导（严重级）。
4. **区分编造和偏差**：核心参数完全不同 → 编造；部分对部分错 → 偏差。
5. **不要过度检测**：如果知识库有相应信息且回复准确 → 无幻觉。
6. **知识库为"无"时的判断**：
   - 回复给出了具体操作结果（如"已帮您修改地址"）→ 能力越界
   - 回复给出了具体信息（如包裹位置、退款时间）→ 能力越界
   - 回复给出了肯定性结论（如"支持NFC"）但KB说"未标注"→ 参数编造

## 输出格式

必须只输出一个 JSON 对象：
{{
  "is_hallucination": true/false,
  "hallucination_type": "类型名称，从上述分类中选择。非幻觉则为 null",
  "severity": "严重/高危/中等/轻微。非幻觉则为 null",
  "confidence": 0.0~1.0,
  "detail": "简要说明检测依据，指出知识库与回复的具体矛盾点。非幻觉也要说明理由"
}}

只输出 JSON，不要输出其他内容。"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-5"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.category_summary = get_category_summary()

    def detect_all(self, replies: List[dict]) -> List[DetectionResult]:
        """批量检测"""
        results = []
        for reply in replies:
            result = self.detect_one(reply)
            results.append(result)
            flag = "[!] HALLUCINATION" if result.is_hallucination else "[OK] NORMAL"
            print(f"  [{result.id}] {flag} "
                  f"-> {result.hallucination_type or '-'} "
                  f"(confidence: {result.confidence:.0%})")
        return results

    def detect_one(self, reply: dict) -> DetectionResult:
        """检测单条回复"""
        import anthropic

        user_prompt = self._build_user_prompt(reply)

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.SYSTEM_PROMPT.format(
                    category_summary=self.category_summary
                ),
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Extract text from response, handling thinking blocks
            text_blocks = [
                block.text for block in message.content
                if hasattr(block, 'text') and getattr(block, 'type', '') == 'text'
            ]
            if not text_blocks:
                raise ValueError(f"No text block in response: {message.content}")
            response_text = text_blocks[0].strip()
            # 清理可能的 markdown code block 包装
            response_text = self._clean_json_response(response_text)
            parsed = json.loads(response_text)

            return DetectionResult(
                id=reply["id"],
                is_hallucination=parsed.get("is_hallucination", False),
                hallucination_type=parsed.get("hallucination_type"),
                severity=parsed.get("severity"),
                category=self._infer_category(parsed.get("hallucination_type")),
                detail=parsed.get("detail", ""),
                confidence=parsed.get("confidence", 0.5),
            )

        except Exception as e:
            print(f"  [{reply['id']}] LLM 调用失败: {e}，回退到 Mock 模式")
            mock = MockDetector()
            return mock.detect_one(reply)

    def _build_user_prompt(self, reply: dict) -> str:
        """构建用户提示"""
        return f"""请检测以下客服回复是否存在幻觉：

**用户问题：**
{reply['user_question']}

**客服回复：**
{reply['system_reply']}

**知识库参考：**
{reply.get('knowledge_base', '无相关知识库条目')}

请对照知识库逐一核查回复中的每个事实性陈述，判断是否存在幻觉并分类。"""

    def _infer_category(self, type_name: Optional[str]) -> Optional[str]:
        """从子类型推断大类"""
        if not type_name:
            return None
        t = get_type_by_name(type_name)
        return t.category.value if t else None

    @staticmethod
    def _clean_json_response(text: str) -> str:
        """清理 LLM 响应中的 markdown 包装"""
        # 移除 ```json ... ``` 包装
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        return text.strip()


def create_detector(mode: str = "real", api_key: Optional[str] = None) -> Any:
    """工厂函数：根据模式创建检测器"""
    if mode == "mock":
        return MockDetector()
    return LLMDetector(api_key=api_key)
